from voc.pii import mask_pii


def test_mascara_email_cpf_e_telefone():
    texto = "Falei com joao@mail.com, CPF 123.456.789-09, tel (11) 98765-4321."
    out = mask_pii(texto)
    assert "joao@mail.com" not in out and "123.456.789-09" not in out
    assert "98765-4321" not in out
    assert "[EMAIL]" in out and "[CPF]" in out and "[TELEFONE]" in out


def test_nao_mexe_em_texto_normal():
    assert mask_pii("Chegou 3 dias depois do prazo") == "Chegou 3 dias depois do prazo"