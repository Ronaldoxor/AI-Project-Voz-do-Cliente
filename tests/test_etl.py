import pandas as pd

from voc.etl import build_fact


def _tables():
    reviews = pd.DataFrame({
        "review_id": ["r1", "r1", "r2", "r3"],
        "order_id": ["o1", "o1", "o2", "o3"],
        "review_score": [1, 1, 5, 4],
        "review_comment_title": ["Atraso", "Atraso", None, None],
        "review_comment_message": ["Chegou muito depois do prazo", "Chegou muito depois do prazo",
                                  "Ótimo produto, recomendo", "bom"],
        "review_creation_date": ["2018-03-10 00:00:00"] * 4,
    })
    orders = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"], "customer_id": ["c1", "c2", "c3"],
        "order_status": ["delivered", "delivered", "delivered"],
        "order_purchase_timestamp": ["2018-02-01 10:00:00"] * 3,
        "order_delivered_customer_date": ["2018-03-05 10:00:00", "2018-02-10 10:00:00",
                                          "2018-02-10 10:00:00"],
        "order_estimated_delivery_date": ["2018-02-20 00:00:00"] * 3,
    })
    items = pd.DataFrame({
        "order_id": ["o1", "o1", "o2", "o3"], "order_item_id": [1, 2, 1, 1],
        "product_id": ["p1", "p2", "p2", "p2"], "seller_id": ["s1", "s2", "s2", "s2"],
        "price": [100.0, 50.0, 30.0, 10.0], "freight_value": [10.0, 5.0, 3.0, 1.0],
    })
    products = pd.DataFrame({"product_id": ["p1", "p2"],
                             "product_category_name": ["cama_mesa_banho", "esporte_lazer"]})
    customers = pd.DataFrame({"customer_id": ["c1", "c2", "c3"], "customer_state": ["SP", "RJ", "MG"]})
    translation = pd.DataFrame({"product_category_name": ["cama_mesa_banho", "esporte_lazer"],
                                "product_category_name_english": ["bed_bath_table", "sports_leisure"]})
    return reviews, orders, items, products, customers, translation


def test_build_fact():
    df = build_fact(*_tables())
    assert sorted(df["review_id"]) == ["r1", "r2"]        # duplicado e texto curto saem
    r1 = df[df["review_id"] == "r1"].iloc[0]
    assert r1["review_text"] == "Atraso. Chegou muito depois do prazo"
    assert r1["customer_state"] == "SP" and r1["product_category"] == "bed_bath_table"
    assert r1["order_value"] == 150.0 and r1["seller_id"] == "s1"
    assert bool(r1["late_delivery"]) is True and r1["delivery_days"] == 32
    r2 = df[df["review_id"] == "r2"].iloc[0]
    assert bool(r2["late_delivery"]) is False