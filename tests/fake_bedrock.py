import json

import pandas as pd
import pytest

from voc.enrich import analyze_review, parse_insight, pick_sample

OK = {"sentiment": "negativo", "topic": "entrega_atraso", "severity": 4,
      "wants_refund": False, "summary": "Entrega atrasou 10 dias."}


def test_parse_aceita_json_com_texto_em_volta():
    ins = parse_insight("Claro!\n```json\n" + json.dumps(OK) + "\n```")
    assert ins.topic == "entrega_atraso" and ins.severity == 4


def test_parse_rejeita_valor_fora_do_enum():
    with pytest.raises(ValueError):
        parse_insight(json.dumps({**OK, "topic": "invencao"}))


def test_parse_rejeita_sem_json():
    with pytest.raises(ValueError):
        parse_insight("não sei")


class FakeBedrock:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def converse(self, **kwargs):
        self.calls += 1
        return {"output": {"message": {"content": [{"text": self.replies.pop(0)}]}}}


def test_retry_recupera_de_resposta_ruim():
    client = FakeBedrock(["desculpe", json.dumps(OK)])
    assert analyze_review(client, "m", "texto").sentiment == "negativo"
    assert client.calls == 2


def test_retry_desiste_depois_do_limite():
    client = FakeBedrock(["x", "y", "z"])
    with pytest.raises(ValueError):
        analyze_review(client, "m", "texto", retries=2)
    assert client.calls == 3


def test_pick_sample_prioriza_notas_baixas():
    df = pd.DataFrame({"review_id": range(100), "review_score": [1] * 20 + [5] * 80})
    s = pick_sample(df, 20)
    assert len(s) == 20 and (s["review_score"] <= 2).sum() >= 10   # metade reservada às notas 1-2
    assert s["review_id"].is_unique