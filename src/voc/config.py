"""Configuração central: tudo vem de variáveis de ambiente (.env no desenvolvimento)."""
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # na Lambda não existe .env: as variáveis vêm da própria função
    pass

REGION = os.getenv("AWS_REGION", "us-east-1")

# Bucket S3 comum (prefixos: raw/, curated/, insights/, athena-results/)
BUCKET = os.getenv("VOC_BUCKET", "")
GLUE_DB = os.getenv("VOC_GLUE_DB", "voc")
ATHENA_OUTPUT = f"s3://{BUCKET}/athena-results/"

# Bucket de vetores (S3 Vectors) e índice
VECTOR_BUCKET = os.getenv("VOC_VECTOR_BUCKET", "")
VECTOR_INDEX = os.getenv("VOC_VECTOR_INDEX", "reviews")

# Modelos do Bedrock
EMBED_MODEL_ID = os.getenv("VOC_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
EMBED_DIM = 1024
LLM_MODEL_ID = os.getenv("VOC_LLM_MODEL_ID", "")  # ID ou inference profile copiado do console

# Quantas avaliações processar com LLM/embeddings (controla o custo)
SAMPLE_SIZE = int(os.getenv("VOC_SAMPLE_SIZE", "3000"))