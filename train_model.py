# train_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

print("="*60)
print("CUSTOMER SEGMENTATION MODEL TRAINING")
print("="*60)

# ==========================================================
# 1. LOAD DATA
# ==========================================================

print("\nLoading dataset...")

df = pd.read_excel("Online_Retail.xlsx")

print("Dataset Shape:", df.shape)

# ==========================================================
# 2. DATA CLEANING
# ==========================================================

print("\nCleaning data...")

# Remove cancelled invoices
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# Remove missing CustomerID
df = df.dropna(subset=['CustomerID'])

# Convert CustomerID
df['CustomerID'] = df['CustomerID'].astype(int)

# Convert InvoiceDate
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Remove invalid values
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

print("Cleaned Dataset Shape:", df.shape)

# ==========================================================
# 3. FEATURE ENGINEERING (RFM)
# ==========================================================

print("\nCreating RFM features...")

df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalAmount': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

print("RFM Shape:", rfm.shape)

# ==========================================================
# 4. LOG TRANSFORMATION
# ==========================================================

print("\nApplying log transformation...")

rfm_log = rfm.copy()

rfm_log['Recency'] = np.log1p(rfm_log['Recency'])
rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
rfm_log['Monetary'] = np.log1p(rfm_log['Monetary'])

# ==========================================================
# 5. SCALING
# ==========================================================

print("\nScaling data...")

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(rfm_log)

joblib.dump(scaler, "scaler.pkl")

print("Scaler saved.")

# ==========================================================
# 6. FIND BEST K USING SILHOUETTE
# ==========================================================

print("\nFinding optimal K...")

best_k = None
best_score = -1

silhouette_results = []

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(rfm_scaled)

    sil = silhouette_score(
        rfm_scaled,
        labels,
        sample_size=min(2000, len(rfm_scaled)),
        random_state=42
    )

    silhouette_results.append({
        'K': k,
        'Silhouette': sil
    })

    print(f"K={k} | Silhouette={sil:.4f}")

    if sil > best_score:
        best_score = sil
        best_k = k

print("\nBest K Selected:", best_k)
print("Best Silhouette:", round(best_score, 4))

with open("best_k.txt", "w") as f:
    f.write(str(best_k))

# ==========================================================
# 7. FINAL KMEANS MODEL
# ==========================================================

print("\nTraining Final K-Means...")

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

kmeans_labels = final_kmeans.fit_predict(rfm_scaled)

rfm['Cluster'] = kmeans_labels

joblib.dump(final_kmeans, "kmeans_model.pkl")

print("KMeans model saved.")

# ==========================================================
# 8. HIERARCHICAL CLUSTERING
# ==========================================================

print("\nTraining Hierarchical Clustering...")

hierarchical = AgglomerativeClustering(
    n_clusters=best_k,
    linkage='ward'
)

hier_labels = hierarchical.fit_predict(rfm_scaled)

# ==========================================================
# 9. DBSCAN
# ==========================================================

print("\nTraining DBSCAN...")

dbscan = DBSCAN(
    eps=0.7,
    min_samples=5
)

db_labels = dbscan.fit_predict(rfm_scaled)

# ==========================================================
# 10. ALGORITHM COMPARISON
# ==========================================================

print("\nEvaluating algorithms...")

results = []

# ---------- KMeans ----------
kmeans_sil = silhouette_score(
    rfm_scaled,
    kmeans_labels,
    sample_size=min(2000, len(rfm_scaled)),
    random_state=42
)

kmeans_dbi = davies_bouldin_score(
    rfm_scaled,
    kmeans_labels
)

results.append({
    'Algorithm': 'K-Means',
    'Silhouette Score': round(kmeans_sil, 4),
    'Davies-Bouldin Index': round(kmeans_dbi, 4)
})

# ---------- Hierarchical ----------
hier_sil = silhouette_score(
    rfm_scaled,
    hier_labels,
    sample_size=min(2000, len(rfm_scaled)),
    random_state=42
)

hier_dbi = davies_bouldin_score(
    rfm_scaled,
    hier_labels
)

results.append({
    'Algorithm': 'Hierarchical',
    'Silhouette Score': round(hier_sil, 4),
    'Davies-Bouldin Index': round(hier_dbi, 4)
})

# ---------- DBSCAN ----------
n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)

if n_clusters_db > 1:

    db_sil = silhouette_score(
        rfm_scaled,
        db_labels,
        sample_size=min(2000, len(rfm_scaled)),
        random_state=42
    )

    db_dbi = davies_bouldin_score(
        rfm_scaled,
        db_labels
    )

    results.append({
        'Algorithm': 'DBSCAN',
        'Silhouette Score': round(db_sil, 4),
        'Davies-Bouldin Index': round(db_dbi, 4)
    })

else:

    results.append({
        'Algorithm': 'DBSCAN',
        'Silhouette Score': np.nan,
        'Davies-Bouldin Index': np.nan
    })

comparison_df = pd.DataFrame(results)

comparison_df.to_csv(
    "algorithm_comparison.csv",
    index=False
)

print("\nAlgorithm Comparison:")
print(comparison_df)

# ==========================================================
# 11. PCA
# ==========================================================

print("\nApplying PCA...")

pca = PCA(n_components=2)

pca_data = pca.fit_transform(rfm_scaled)

joblib.dump(pca, "pca.pkl")

print("PCA saved.")

pca_df = pd.DataFrame(
    pca_data,
    columns=['PC1', 'PC2']
)

pca_df['Cluster'] = kmeans_labels

pca_df.to_csv(
    "pca_data.csv",
    index=False
)

# ==========================================================
# 12. CLUSTER PROFILES
# ==========================================================

print("\nCreating cluster profiles...")

cluster_profiles = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean'
}).round(2)

cluster_profiles['Segment'] = 'Customer Segment'

# Automatic segment naming
sorted_clusters = cluster_profiles.sort_values(
    by='Monetary',
    ascending=False
).index.tolist()

segment_names = [
    'VIP Customers',
    'Loyal Customers',
    'Potential Loyalists',
    'At Risk Customers',
    'Occasional Buyers',
    'Inactive Customers',
    'High Potential Customers',
    'Regular Customers',
    'New Customers',
    'Price Sensitive'
]

for i, cluster in enumerate(sorted_clusters):

    if i < len(segment_names):
        cluster_profiles.loc[
            cluster,
            'Segment'
        ] = segment_names[i]

cluster_profiles.to_csv(
    "cluster_profiles.csv"
)

print(cluster_profiles)

# ==========================================================
# 13. SAVE RFM DATASET
# ==========================================================

rfm.to_csv(
    "rfm_dataset.csv"
)

print("\nRFM dataset saved.")

# ==========================================================
# 14. ELBOW PLOT
# ==========================================================

print("\nGenerating Elbow Plot...")

inertia = []

for k in range(2, 11):

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    km.fit(rfm_scaled)

    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    inertia,
    marker='o'
)

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.savefig(
    "elbow_plot.png",
    bbox_inches='tight'
)

plt.close()

print("Elbow plot saved.")

# ==========================================================
# 15. SILHOUETTE PLOT
# ==========================================================

sil_df = pd.DataFrame(silhouette_results)

plt.figure(figsize=(8, 5))

plt.plot(
    sil_df['K'],
    sil_df['Silhouette'],
    marker='o'
)

plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")

plt.savefig(
    "silhouette_plot.png",
    bbox_inches='tight'
)

plt.close()

print("Silhouette plot saved.")

# ==========================================================
# COMPLETE
# ==========================================================

print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("="*60)

print("\nGenerated Files:")

files = [
    "scaler.pkl",
    "kmeans_model.pkl",
    "pca.pkl",
    "best_k.txt",
    "algorithm_comparison.csv",
    "cluster_profiles.csv",
    "rfm_dataset.csv",
    "pca_data.csv",
    "elbow_plot.png",
    "silhouette_plot.png"
]

for file in files:
    print(file saved)

print("\nOptimal K:", best_k)
print("Project Ready for Streamlit Deployment!")
