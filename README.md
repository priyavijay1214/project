# Customer Segmentation (RFM + K-Means)

Segments e-commerce customers into actionable marketing groups using
SQL-computed RFM (Recency, Frequency, Monetary) features and K-Means clustering.

## Pipeline
1. **`generate_data.py`** — synthetic transaction log (~30K transactions, ~4,000 customers) built from 7 latent behavioral archetypes (champions, loyal, big spenders, new, at-risk, dormant, one-time) so clustering has real structure to recover.
2. **`sql_rfm.py`** — loads transactions into SQLite and computes RFM features entirely in **SQL**: recency/frequency/monetary aggregates, category diversity, purchase cadence, and quintile scores via `NTILE(5) OVER (...)` window functions.
3. **`cluster.py`** — scales + log-transforms features, evaluates k=2 to 9 via elbow method and silhouette score, selects k=6 for business interpretability (silhouette-maximizing k=3 collapses distinct segments like Champions and Big Spenders together), and labels each cluster using a rank-based heuristic across recency/frequency/monetary/AOV so every segment gets a distinct, human-readable name. PCA visualization of the resulting clusters.

## Results
- 6 segments identified: **Champions, Big Spenders, Loyal Regulars, At Risk, New Customers, Dormant**.
- Silhouette score 0.328, Davies-Bouldin index 0.934 at k=6.
- Segment sizes range from 248 (Big Spenders) to 1,142 (Loyal Regulars) customers — realistic, actionable group sizes for targeted marketing campaigns.
- Full cluster profiles in `results_summary.txt`, visualization in `segmentation_results.png`.

## Stack
Python, SQLite (SQL window functions), pandas, scikit-learn (K-Means, PCA), matplotlib.


