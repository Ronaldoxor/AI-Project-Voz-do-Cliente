import os

import boto3
from dotenv import load_dotenv

load_dotenv()
s3v = boto3.client("s3vectors", region_name=os.getenv("AWS_REGION", "us-east-1"))
args = dict(vectorBucketName=os.environ["VOC_VECTOR_BUCKET"],
            indexName=os.getenv("VOC_VECTOR_INDEX", "reviews"), maxResults=500)
total, token = 0, None
while True:
    if token:
        args["nextToken"] = token
    resp = s3v.list_vectors(**args)
    total += len(resp["vectors"])
    token = resp.get("nextToken")
    if not token:
        break
print(f"{total} vetores no índice")