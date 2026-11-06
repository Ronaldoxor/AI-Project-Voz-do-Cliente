-- Uma linha por avaliação com texto; colunas de IA ficam nulas fora da amostra analisada.
CREATE OR REPLACE VIEW voc.vw_reviews_enriched AS
SELECT f.review_id, f.review_date, f.review_score, f.customer_state, f.product_category,
       f.seller_id, f.order_value, f.freight_value, f.delivery_days, f.late_delivery,
       f.review_text, i.sentiment, i.topic, i.severity, i.wants_refund, i.summary
FROM voc.fact_reviews f
LEFT JOIN voc.review_insights i ON i.review_id = f.review_id;