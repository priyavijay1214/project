"""
Loads raw transactions into SQLite and computes RFM (Recency, Frequency,
Monetary) features plus category-diversity signals entirely in SQL.
"""
import sqlite3
import pandas as pd

conn = sqlite3.connect("/home/claude/customer-segmentation/transactions.db")
tx = pd.read_csv("/home/claude/customer-segmentation/transactions_raw.csv", parse_dates=["order_date"])
tx.to_sql("transactions", conn, if_exists="replace", index=False)

REF_DATE = tx["order_date"].max().date().isoformat()

RFM_SQL = f"""
CREATE TABLE customer_rfm AS
WITH agg AS (
    SELECT
        customer_id,
        COUNT(*) AS frequency,
        ROUND(SUM(order_amount), 2) AS monetary_total,
        ROUND(AVG(order_amount), 2) AS avg_order_value,
        MIN(order_date) AS first_purchase,
        MAX(order_date) AS last_purchase,
        COUNT(DISTINCT category) AS category_diversity,
        CAST(JULIANDAY('{REF_DATE}') - JULIANDAY(MAX(order_date)) AS INTEGER) AS recency_days,
        CAST(JULIANDAY(MAX(order_date)) - JULIANDAY(MIN(order_date)) AS INTEGER) AS tenure_days
    FROM transactions
    GROUP BY customer_id
),
scored AS (
    SELECT
        *,
        -- quintile scores via window function NTILE (1=worst, 5=best)
        NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,
        NTILE(5) OVER (ORDER BY frequency ASC) AS f_score,
        NTILE(5) OVER (ORDER BY monetary_total ASC) AS m_score,
        -- purchase cadence: avg days between orders
        ROUND(CAST(tenure_days AS FLOAT) / NULLIF(frequency - 1, 0), 1) AS avg_days_between_orders
    FROM agg
)
SELECT
    *,
    (r_score + f_score + m_score) AS rfm_total_score
FROM scored;
"""

conn.executescript("DROP TABLE IF EXISTS customer_rfm;")
conn.executescript(RFM_SQL)
conn.commit()

check = pd.read_sql("""
    SELECT rfm_total_score, COUNT(*) as n, ROUND(AVG(monetary_total),0) as avg_spend
    FROM customer_rfm GROUP BY rfm_total_score ORDER BY rfm_total_score DESC;
""", conn)
print(check)

rfm_df = pd.read_sql("SELECT * FROM customer_rfm", conn)
rfm_df.to_csv("/home/claude/customer-segmentation/customer_rfm.csv", index=False)
print("\nRFM table shape:", rfm_df.shape)
conn.close()
