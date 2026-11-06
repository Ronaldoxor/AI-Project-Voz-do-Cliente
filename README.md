# Voz do Cliente AI

Plataforma que transforma avaliações de clientes em dores priorizadas, com chat RAG que cita as fontes e dashboard no Power BI. Construída na AWS com LLM (Bedrock), busca vetorial (S3 Vectors) e BI.

![Dashboard](docs/powerbi-visao-geral.png)
![Chat](docs/chat.png)

## O que faz
- **Data lake:** ETL em Python: CSVs do Olist viram Parquet no S3, catalogado no Glue e consultado no Athena.
- **Enriquecimento com LLM:** classifica cada avaliação (sentimento, tema, severidade, pedido de reembolso) em JSON validado com Pydantic, com retry e checkpoint.
- **RAG:** embeddings Titan V2, busca vetorial com filtros por metadados e resposta com citação de fontes.
- **API:** AWS Lambda com validação de entrada e IAM de menor privilégio.
- **Dashboard:** Power BI em 4 páginas (Athena via ODBC, DAX, drill-through).
- **Privacidade:** mascaramento de CPF, e-mail e telefone antes de enviar texto ao LLM.

## Arquitetura
![Arquitetura](docs/arquitetura.png)

Dataset Olist → S3 (raw/curated) → Glue + Athena → Bedrock (classificação) → Titan Embeddings → S3 Vectors → Lambda → Streamlit. O Power BI lê a view do Athena.

## Tecnologias
| Área | Ferramentas |
| --- | --- |
| AWS | S3, Glue, Athena, Bedrock, S3 Vectors, Lambda, IAM, CloudWatch |
| IA | Amazon Nova 2 Lite (classificação e respostas), Titan Text Embeddings V2, RAG, LLM-as-judge |
| Dados | Python, pandas, pyarrow, awswrangler, Parquet, SQL |
| Qualidade | Pydantic v2, pytest |
| BI e interface | Power BI (DAX), Streamlit |

## Resultados
| Métrica | Valor |
| --- | --- |
| Avaliações analisadas pelo LLM | [N] |
| Auditoria (amostra de 20 avaliações): sentimento | 100% |
| Auditoria (amostra de 20 avaliações): tema | 95% |
| Auditoria (amostra de 20 avaliações): severidade | 95% |
| Precisão@k da busca (conjunto de teste) | [x] |
| Respostas fundamentadas (LLM-juiz) | [y]% |
| Custo total do projeto | US$ [valor] |

**O que aprendi com a auditoria:**
- Elogios de entrega ("chegou no prazo") eram classificados como `entrega_atraso`. Se esses casos forem contados como erro de tema, a taxa de tema cai para 75%. Por isso a página Dores do Power BI filtra `sentiment = negativo`.
- `wants_refund` vinha como verdadeiro em reclamações graves sem pedido explícito. Restringi o prompt para exigir pedido explícito de reembolso, devolução ou estorno.

## Decisões e trade-offs
- **S3 Vectors** em vez de OpenSearch Serverless ou pgvector: sem custo fixo de cluster; suficiente para este volume.
- **Importar** no Power BI em vez de DirectQuery: dados pequenos e estáticos.
- **Temas fechados (enum):** o dashboard precisa de categorias estáveis.
- **Amostra** com viés para notas baixas: controla custo.

## Limitações
- A auditoria usa só 20 avaliações, sendo 11 de nota 1 ou 2 (a amostragem favorece notas baixas de propósito). Os percentuais são indicativos, não estatisticamente conclusivos.
- Os percentuais de tema da página Dores não representam toda a base, pelo viés da amostra.
- Precisão@k usa rótulos gerados por LLM.
- Mascaramento de dados pessoais é por regex.
- O primeiro item do pedido define a categoria.
- O RAG não responde perguntas numéricas; para isso use o Power BI.

## Como rodar
```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows (Git Bash)
pip install -e ".[dev,app]" kaggle
cp .env.example .env     # ajuste bucket, modelo e conta
pytest -q
python -m voc.etl data/olist
python -m voc.enrich
python -m voc.index_vectors
streamlit run app/streamlit_app.py
```
Detalhes de IAM, Lambda e Athena em `docs/`. Para desligar tudo: `infra/teardown.sh`.

## Dados e licença
Dataset: [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), CC BY-NC-SA 4.0 (uso não comercial, com atribuição). Código sob licença [escolha, ex.: MIT].