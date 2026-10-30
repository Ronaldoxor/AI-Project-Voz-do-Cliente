"""Mascara dados pessoais antes de enviar texto a um LLM (boa prática de LGPD)."""
import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_CPF = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
# (11) 98765-4321, 11 98765 4321, 98765-4321, 011987654321...
_PHONE = re.compile(r"(?<!\d)(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?9?\d{4}[-\s]?\d{4}(?!\d)")


def mask_pii(text: str) -> str:
    text = _EMAIL.sub("[EMAIL]", text)
    text = _CPF.sub("[CPF]", text)
    text = _PHONE.sub("[TELEFONE]", text)
    return text