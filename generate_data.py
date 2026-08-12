"""
Generate synthetic e-commerce transaction data with several latent customer
archetypes baked in (champions, loyal, at-risk, new, dormant, big-spenders)
so clustering has real structure to recover -- not just noise.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(7)
N_CUSTOMERS = 4000
END_DATE = datetime(2026, 7, 1)

archetypes = {
    "champion":     dict(weight=.08, freq_lambda=22, recency_days=(1, 15),   aov_mean=95,  aov_sd=25),
    "loyal":        dict(weight=.20, freq_lambda=13, recency_days=(1, 30),   aov_mean=60,  aov_sd=18),
    "big_spender":  dict(weight=.07, freq_lambda=6,  recency_days=(1, 45),   aov_mean=220, aov_sd=60),
    "new":          dict(weight=.15, freq_lambda=2,  recency_days=(1, 20),   aov_mean=50,  aov_sd=15),
    "at_risk":      dict(weight=.20, freq_lambda=8,  recency_days=(90, 180), aov_mean=55,  aov_sd=15),
    "dormant":      dict(weight=.20, freq_lambda=4,  recency_days=(200, 500),aov_mean=45,  aov_sd=15),
    "one_time":     dict(weight=.10, freq_lambda=1,  recency_days=(100, 600),aov_mean=40,  aov_sd=12),
}

rows = []
customer_id = 1
for name, cfg in archetypes.items():
    n = int(N_CUSTOMERS * cfg["weight"])
    for _ in range(n):
        freq = max(1, np.random.poisson(cfg["freq_lambda"]))
        recency = np.random.randint(cfg["recency_days"][0], cfg["recency_days"][1] + 1)
        last_purchase = END_DATE - timedelta(days=int(recency))
        aov = max(8, np.random.normal(cfg["aov_mean"], cfg["aov_sd"]))
        tenure_days = np.random.randint(recency + 10, recency + 900)
        first_purchase = END_DATE - timedelta(days=int(tenure_days))

        # generate individual transactions across their lifetime
        span = max((last_purchase - first_purchase).days, 1)
        purchase_offsets = sorted(np.random.choice(range(span), size=min(freq, span), replace=False))
        for off in purchase_offsets:
            tx_date = first_purchase + timedelta(days=int(off))
            amount = max(5, np.random.normal(aov, aov_sd_local := cfg["aov_sd"] * 0.6))
            category = np.random.choice(
                ["electronics", "apparel", "home_goods", "beauty", "sports", "books", "grocery"],
                p=[.18, .22, .16, .14, .1, .1, .1]
            )
            rows.append((customer_id, tx_date.date().isoformat(), round(amount, 2), category, name))
        customer_id += 1

tx = pd.DataFrame(rows, columns=["customer_id", "order_date", "order_amount", "category", "true_segment"])
tx.to_csv("/home/claude/customer-segmentation/transactions_raw.csv", index=False)
print(tx.shape)
print(tx["true_segment"].value_counts())
print(tx.groupby("true_segment")["order_amount"].mean())
