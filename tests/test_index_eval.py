from voc.evaluation import precision_at_k
from voc.index_vectors import to_vector


def test_to_vector_formato_s3vectors():
    v = to_vector({"review_id": "r1", "customer_state": "SP", "product_category": "toys",
                   "review_score": 2, "sentiment": "negativo", "topic": "embalagem",
                   "review_text": "caixa rasgada"}, [0.1, 0.2])
    assert v["key"] == "r1" and v["data"] == {"float32": [0.1, 0.2]}
    assert v["metadata"]["score"] == 2 and v["metadata"]["text"] == "caixa rasgada"


def test_precision_at_k():
    hits = [{"topic": "a"}, {"topic": "a"}, {"topic": "b"}, {"topic": "a"}]
    assert precision_at_k(hits, "a") == 0.75
    assert precision_at_k([], "a") == 0.0