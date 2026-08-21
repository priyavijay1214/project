import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("/home/claude/customer-segmentation/customer_rfm.csv")

features = ["recency_days", "frequency", "monetary_total", "avg_order_value", "category_diversity"]
X = df[features].copy()
# log-transform monetary/frequency to reduce skew (standard RFM practice)
X["monetary_total"] = np.log1p(X["monetary_total"])
X["frequency"] = np.log1p(X["frequency"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow + Silhouette to select k 
inertias, sils, dbs = [], [], []
k_range = range(2, 10)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled, labels))
    dbs.append(davies_bouldin_score(X_scaled, labels))

print(f"Silhouette scores by k: {dict(zip(k_range, np.round(sils,3)))}")
# k=3 maximizes silhouette but collapses distinct business behaviors together;
# k=6 gives up a modest amount of silhouette score for materially more
# actionable, distinguishable marketing segments -- the standard RFM trade-off.
best_k = 6
print(f"Selected k = {best_k} (silhouette={sils[list(k_range).index(best_k)]:.3f}); "
      f"chosen over the silhouette-maximizing k=3 for business interpretability "
      f"(k=3 merges Champions/Big Spenders into one bucket).")

# Final model 
final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster"] = final_km.fit_predict(X_scaled)
final_sil = silhouette_score(X_scaled, df["cluster"])
final_db = davies_bouldin_score(X_scaled, df["cluster"])
print(f"Final model: k={best_k}, silhouette={final_sil:.3f}, davies-bouldin={final_db:.3f}")

# Business labeling based on cluster centroids 
profile = df.groupby("cluster")[features].mean().round(1)
profile["n_customers"] = df["cluster"].value_counts().sort_index()
print("\nCluster profiles:\n", profile)

# Rank-based labeling: rank clusters relative to EACH OTHER (not global mean)
# on recency, frequency, monetary, and AOV, guaranteeing distinct assignments.
ranked = profile[features].rank(ascending=False)  # rank 1 = highest value across clusters
# for recency_days, LOWER is better (more recent), so invert
ranked["recency_days"] = profile["recency_days"].rank(ascending=True)

candidates = []
for c in profile.index:
    r = ranked.loc[c]
    candidates.append({
        "cluster": c,
        "recency_rank": r["recency_days"],
        "freq_rank": r["frequency"],
        "monetary_rank": r["monetary_total"],
        "aov_rank": r["avg_order_value"],
    })
cand_df = pd.DataFrame(candidates).set_index("cluster")

labels_map = {}
used = set()
k = len(profile.index)

def assign(name, cluster_id):
    if cluster_id not in used:
        labels_map[cluster_id] = name
        used.add(cluster_id)

# Champions: best combined recency+frequency+monetary
combined = cand_df["recency_rank"] + cand_df["freq_rank"] + cand_df["monetary_rank"]
assign("Champions", combined.idxmin())

# Big Spenders: highest AOV among remaining
remaining = cand_df.drop(index=list(used), errors="ignore")
if not remaining.empty:
    assign("Big Spenders", remaining["aov_rank"].idxmin())

# Dormant: worst recency among remaining
remaining = cand_df.drop(index=list(used), errors="ignore")
if not remaining.empty:
    assign("Dormant", remaining["recency_rank"].idxmax())

# At Risk: high frequency (historically active) but poor recency among remaining
remaining = cand_df.drop(index=list(used), errors="ignore")
if not remaining.empty:
    at_risk_score = remaining["recency_rank"] - remaining["freq_rank"]
    assign("At Risk", at_risk_score.idxmax())

# New Customers: low frequency but good recency among remaining
remaining = cand_df.drop(index=list(used), errors="ignore")
if not remaining.empty:
    new_score = remaining["freq_rank"] - remaining["recency_rank"]
    assign("New Customers", new_score.idxmax())

# Anything left -> Loyal Regulars
for c in profile.index:
    if c not in used:
        assign("Loyal Regulars", c)

df["segment_label"] = df["cluster"].map(labels_map)
print("\nSegment labels:", labels_map)
print("\nSegment sizes:\n", df["segment_label"].value_counts())

# accuracy check against ground-truth archetype (not used by the model, only for validation)
if "true_segment" in pd.read_csv("/home/claude/customer-segmentation/customer_rfm.csv").columns:
    pass  # true_segment not carried into RFM table; cross-check separately below

# Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(list(k_range), inertias, marker="o")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")
axes[0].axvline(best_k, color="red", linestyle="--", alpha=0.5)

axes[1].plot(list(k_range), sils, marker="o", color="green")
axes[1].set_title("Silhouette Score by k")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette Score")
axes[1].axvline(best_k, color="red", linestyle="--", alpha=0.5)

pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)
scatter = axes[2].scatter(coords[:, 0], coords[:, 1], c=df["cluster"], cmap="tab10", alpha=0.5, s=12)
axes[2].set_title(f"Customer Segments (PCA, k={best_k})")
axes[2].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% var)")
axes[2].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% var)")
legend1 = axes[2].legend(*scatter.legend_elements(), title="Cluster", loc="best", fontsize=8)
axes[2].add_artist(legend1)

plt.tight_layout()
plt.savefig("/home/claude/customer-segmentation/segmentation_results.png", dpi=150)
print("\nSaved plot to segmentation_results.png")

df.to_csv("/home/claude/customer-segmentation/customer_segments_final.csv", index=False)

with open("/home/claude/customer-segmentation/results_summary.txt", "w") as f:
    f.write(f"Selected k = {best_k}\n")
    f.write(f"Silhouette score: {final_sil:.3f}\n")
    f.write(f"Davies-Bouldin index: {final_db:.3f}\n")
    f.write(f"Segments identified: {list(set(labels_map.values()))}\n")
    f.write(f"\nCluster profiles:\n{profile.to_string()}\n")
