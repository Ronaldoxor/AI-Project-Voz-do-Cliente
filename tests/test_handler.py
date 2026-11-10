import json
import sys
import types
from unittest import mock

# A Lambda real tem boto3; no teste substituímos os clientes por mocks.
fake_boto3 = types.SimpleNamespace(client=lambda *a, **k: mock.Mock())
sys.modules.setdefault("boto3", fake_boto3)

from voc import lambda_handler, rag  # noqa: E402


def _call(body):
    return lambda_handler.handler({"body": json.dumps(body)}, None)


def test_pergunta_curta_demais_retorna_400():
    assert _call({"question": "oi"})["statusCode"] == 400


def test_filtros_desconhecidos_sao_ignorados():
    with mock.patch.object(rag, "answer", return_value={"answer": "ok", "sources": []}) as m:
        r = _call({"question": "como está a entrega?", "k": 99,
                   "filters": {"state": "SP", "hack": "x", "category": ""}})
    assert r["statusCode"] == 200
    args, kwargs = m.call_args
    assert args[3] == 20 and kwargs == {"state": "SP"}   # k limitado a 20; só filtros válidos


def test_erro_interno_nao_vaza_detalhe():
    with mock.patch.object(rag, "answer", side_effect=RuntimeError("segredo")):
        r = _call({"question": "como está a entrega?"})
    assert r["statusCode"] == 500 and "segredo" not in r["body"]