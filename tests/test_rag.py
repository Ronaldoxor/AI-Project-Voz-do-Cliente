from voc import rag


def test_filtro_vazio_vira_none():
    assert rag.build_filter() is None


def test_filtro_simples_e_composto():
    assert rag.build_filter(state="SP") == {"state": {"$eq": "SP"}}
    assert rag.build_filter(state="SP", max_score=2) == {
        "$and": [{"state": {"$eq": "SP"}}, {"score": {"$lte": 2}}]}


def test_prompt_tem_ids_curtos_e_pergunta():
    hits = [{"id": "abcdef1234567890", "score": 1, "state": "SP", "category": "toys",
             "text": "Chegou quebrado"}]
    p = rag.build_prompt("O que reclamam?", hits)
    assert '<avaliacao id="abcdef12"' in p and "Chegou quebrado" in p
    assert p.endswith("Pergunta: O que reclamam?")


class FakeS3V:
    def query_vectors(self, **kw):
        self.kw = kw
        return {"vectors": [{"key": "k1", "distance": 0.1,
                             "metadata": {"score": 1, "state": "SP", "category": "toys",
                                          "topic": "produto_defeito", "text": "quebrado"}}]}


class FakeBedrock:
    def invoke_model(self, **kw):
        import io, json
        return {"body": io.BytesIO(json.dumps({"embedding": [0.0] * 1024}).encode())}


def test_retrieve_passa_filtro_e_devolve_metadados():
    s3v = FakeS3V()
    hits = rag.retrieve(FakeBedrock(), s3v, "pergunta", 3, state="SP")
    assert s3v.kw["topK"] == 3 and s3v.kw["filter"] == {"state": {"$eq": "SP"}}
    assert hits[0]["id"] == "k1" and hits[0]["topic"] == "produto_defeito"