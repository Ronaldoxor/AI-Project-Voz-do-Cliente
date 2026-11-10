"""Entrada da Lambda: recebe {question, filters, k} e devolve resposta + fontes."""
import json

import boto3

from voc import config, rag

# Clientes criados fora do handler: reaproveitados entre invocações quentes
_bedrock = boto3.client("bedrock-runtime", region_name=config.REGION)
_s3v = boto3.client("s3vectors", region_name=config.REGION)

_ALLOWED_FILTERS = {"state", "category", "max_score", "topic"}


def _response(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body, ensure_ascii=False)}


def handler(event, context):
    try:
        payload = json.loads(event["body"]) if "body" in event else event
        question = str(payload.get("question", "")).strip()
        if not 5 <= len(question) <= 500:
            return _response(400, {"error": "question deve ter entre 5 e 500 caracteres"})
        k = min(max(int(payload.get("k", 8)), 1), 20)
        filters = {key: value for key, value in (payload.get("filters") or {}).items()
                   if key in _ALLOWED_FILTERS and value not in (None, "")}
        return _response(200, rag.answer(_bedrock, _s3v, question, k, **filters))
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        return _response(400, {"error": f"requisição inválida: {exc}"})
    except Exception as exc:  # erro do Bedrock/S3 Vectors: registra e não vaza detalhes
        print(f"ERRO: {exc!r}")
        return _response(500, {"error": "erro interno"})