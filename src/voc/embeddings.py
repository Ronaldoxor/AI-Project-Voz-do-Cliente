"""Embeddings com Amazon Titan Text Embeddings V2."""
import json

from voc import config


def embed_text(bedrock, text: str) -> list[float]:
    """Gera o vetor (1024 dimensões, normalizado) de um texto."""
    resp = bedrock.invoke_model(
        modelId=config.EMBED_MODEL_ID,
        body=json.dumps({"inputText": text, "dimensions": config.EMBED_DIM, "normalize": True}),
    )
    return json.loads(resp["body"].read())["embedding"]