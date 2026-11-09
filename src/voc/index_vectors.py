"""Cria o índice no S3 Vectors e grava um vetor por avaliação, com metadados filtráveis."""
from concurrent.futures import ThreadPoolExecutor

from voc import config
from voc.embeddings import embed_text

BATCH = 100  # put_vectors aceita lotes grandes; 100 mantém as requisições leves


def to_vector(review: dict, embedding: list[float]) -> dict:
    """Formato esperado pelo put_vectors. `text` fica como metadado NÃO filtrável."""
    return {
        "key": review["review_id"],
        "data": {"float32": embedding},
        "metadata": {
            "state": review["customer_state"],
            "category": review["product_category"],
            "score": int(review["review_score"]),
            "sentiment": review["sentiment"],
            "topic": review["topic"],
            "text": review["review_text"],
        },
    }


def ensure_index(s3v) -> None:
    """Idempotente: ignora o erro se o bucket/índice já existir."""
    try:
        s3v.create_vector_bucket(vectorBucketName=config.VECTOR_BUCKET)
    except s3v.exceptions.ConflictException:
        pass
    try:
        s3v.create_index(
            vectorBucketName=config.VECTOR_BUCKET,
            indexName=config.VECTOR_INDEX,
            dataType="float32",
            dimension=config.EMBED_DIM,
            distanceMetric="cosine",
            metadataConfiguration={"nonFilterableMetadataKeys": ["text"]},
        )
    except s3v.exceptions.ConflictException:
        pass


def main() -> None:
    import awswrangler as wr
    import boto3

    s3v = boto3.client("s3vectors", region_name=config.REGION)
    bedrock = boto3.client("bedrock-runtime", region_name=config.REGION)
    ensure_index(s3v)

    df = wr.athena.read_sql_query(
        """SELECT f.review_id, f.review_text, f.review_score, f.customer_state,
                  f.product_category, i.sentiment, i.topic
           FROM fact_reviews f JOIN review_insights i ON f.review_id = i.review_id""",
        database=config.GLUE_DB, ctas_approach=False, s3_output=config.ATHENA_OUTPUT,
    )
    reviews = df.to_dict("records")
    print(f"{len(reviews)} avaliações para indexar")

    with ThreadPoolExecutor(5) as pool:
        embeddings = list(pool.map(lambda r: embed_text(bedrock, r["review_text"]), reviews))

    for start in range(0, len(reviews), BATCH):
        chunk = [to_vector(r, e) for r, e in zip(reviews[start:start + BATCH],
                                                 embeddings[start:start + BATCH])]
        s3v.put_vectors(vectorBucketName=config.VECTOR_BUCKET,
                        indexName=config.VECTOR_INDEX, vectors=chunk)
    print("índice atualizado (put_vectors com a mesma key sobrescreve: pode rodar de novo)")


if __name__ == "__main__":
    main()