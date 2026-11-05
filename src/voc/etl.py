"""ETL: CSVs do Olist -> tabela curada `fact_reviews` (Parquet no S3 + catálogo Glue)."""
import sys
from pathlib import Path

import pandas as pd

from voc import config
from voc.pii import mask_pii


def build_fact(reviews, orders, items, products, customers, translation) -> pd.DataFrame:
    """Une as tabelas do Olist em uma linha por avaliação com texto."""
    r = reviews.drop_duplicates("review_id").copy()
    r["review_text"] = (
        r["review_comment_title"].fillna("") + ". " + r["review_comment_message"].fillna("")
    ).str.strip(" .")
    r["review_text"] = r["review_text"].str.replace(r"\s+", " ", regex=True)
    r = r[r["review_text"].str.len() >= 15]  # descarta "ok", "bom"... sem sinal útil
    r["review_text"] = r["review_text"].map(mask_pii)
    r["review_date"] = pd.to_datetime(r["review_creation_date"]).dt.date

    o = orders[orders["order_status"] == "delivered"].copy()
    for col in ["order_purchase_timestamp", "order_delivered_customer_date",
                "order_estimated_delivery_date"]:
        o[col] = pd.to_datetime(o[col])
    o["delivery_days"] = (o["order_delivered_customer_date"] - o["order_purchase_timestamp"]).dt.days
    o["late_delivery"] = o["order_delivered_customer_date"] > o["order_estimated_delivery_date"]

    # Primeiro item do pedido define categoria e vendedor; valor = soma dos itens
    first = items.sort_values(["order_id", "order_item_id"]).drop_duplicates("order_id")
    totals = items.groupby("order_id", as_index=False).agg(
        order_value=("price", "sum"), freight_value=("freight_value", "sum")
    )
    first = (
        first[["order_id", "product_id", "seller_id"]]
        .merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
        .merge(translation, on="product_category_name", how="left")
        .merge(totals, on="order_id")
    )
    first["product_category"] = first["product_category_name_english"].fillna("unknown")

    df = (
        r.merge(o[["order_id", "customer_id", "delivery_days", "late_delivery"]], on="order_id")
        .merge(customers[["customer_id", "customer_state"]], on="customer_id")
        .merge(first[["order_id", "seller_id", "product_category", "order_value",
                      "freight_value"]], on="order_id")
    )
    return df[["review_id", "order_id", "review_score", "review_text", "review_date",
               "customer_state", "product_category", "seller_id", "order_value",
               "freight_value", "delivery_days", "late_delivery"]].reset_index(drop=True)


def load_csvs(folder: Path):
    read = lambda name: pd.read_csv(folder / name)  # noqa: E731
    return (
        read("olist_order_reviews_dataset.csv"),
        read("olist_orders_dataset.csv"),
        read("olist_order_items_dataset.csv"),
        read("olist_products_dataset.csv"),
        read("olist_customers_dataset.csv"),
        read("product_category_name_translation.csv"),
    )


def main(folder: str = "data/olist") -> None:
    import awswrangler as wr  # import tardio: os testes não precisam da AWS

    df = build_fact(*load_csvs(Path(folder)))
    print(f"{len(df):,} avaliações com texto")
    wr.catalog.create_database(config.GLUE_DB, exist_ok=True)
    wr.s3.to_parquet(
        df,
        path=f"s3://{config.BUCKET}/curated/fact_reviews/",
        dataset=True,
        database=config.GLUE_DB,
        table="fact_reviews",
        mode="overwrite",
    )
    print("fact_reviews gravada no S3 e registrada no Glue Data Catalog")


if __name__ == "__main__":
    main(*sys.argv[1:])