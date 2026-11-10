import boto3
from voc import rag

b = boto3.client("bedrock-runtime", region_name="us-east-1")
s = boto3.client("s3vectors", region_name="us-east-1")
r = rag.answer(b, s, "O que reclamam sobre atraso na entrega?", k=6)
print(r["answer"])
print([x["id"][:8] for x in r["sources"]])