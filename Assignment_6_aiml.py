import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)

# Load dataset
data = load_wine()
X = data.data
y_true = data.target   # used only for external validation (ARI)

# Feature scaling
X_scaled = StandardScaler().fit_transform(X)

# Select number of clusters for K-Means
print("K-MEANS: SELECTION OF NUMBER OF CLUSTERS")
print("-" * 60)

for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    print(f"k={k}  Inertia={km.inertia_:.2f}  "
          f"Silhouette={silhouette_score(X_scaled, km.labels_):.4f}  "
          f"DBI={davies_bouldin_score(X_scaled, km.labels_):.4f}")

# Create clustering models
models = {
    "K-Means": KMeans(n_clusters=3, n_init=10, random_state=42),
    "Hierarchical": AgglomerativeClustering(n_clusters=3, linkage="ward"),
    "GMM": GaussianMixture(n_components=3, random_state=42),
    "DBSCAN": DBSCAN(eps=2.3, min_samples=5)
}

results = []
all_labels = {}

# Fit and evaluate each model
for name, model in models.items():
    labels = model.fit_predict(X_scaled)
    all_labels[name] = labels

    mask = labels != -1   # ignore DBSCAN noise points
    n_clusters = len(set(labels[mask]))
    n_noise = int((~mask).sum())

    results.append({
        "Algorithm": name,
        "Clusters": n_clusters,
        "Noise": n_noise,
        "Silhouette": round(silhouette_score(X_scaled[mask], labels[mask]), 4),
        "Davies-Bouldin": round(davies_bouldin_score(X_scaled[mask], labels[mask]), 4),
        "Calinski-Harabasz": round(calinski_harabasz_score(X_scaled[mask], labels[mask]), 2),
        "ARI": round(adjusted_rand_score(y_true, labels), 4)
    })

# Display comparison table
df = pd.DataFrame(results)

print("\nCLUSTERING ALGORITHM COMPARISON")
print("-" * 85)
print(df.to_string(index=False))

# Cluster sizes of K-Means
print("\nK-MEANS CLUSTER SIZES:",
      dict(sorted(pd.Series(all_labels["K-Means"]).value_counts().items())))

# Display the best results
print("\nPERFORMANCE SUMMARY")
print("Highest Silhouette Score:", df.loc[df["Silhouette"].idxmax(), "Algorithm"])
print("Lowest Davies-Bouldin Index:", df.loc[df["Davies-Bouldin"].idxmin(), "Algorithm"])
print("Highest ARI:", df.loc[df["ARI"].idxmax(), "Algorithm"])

print("\nProgram executed successfully!")

# Visual inspection using PCA (2-D projection)
X_2d = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

for ax, (name, labels) in zip(axes, all_labels.items()):
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="viridis", s=20)
    ax.set_title(name)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

plt.tight_layout()
plt.savefig("clustering_comparison.png", dpi=150)
plt.show()