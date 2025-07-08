import kagglehub #pip install kagglehub
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Download latest version
path = kagglehub.dataset_download("vitaliikaplan/employee-engagement-survey")

print("Path to dataset files:", path)

# ------------------------------------------------------------------
# 1. Read the four CSV files
# ------------------------------------------------------------------
files = [
    path + "/EES-2022-10.csv",
    path + "/EES-2023-05.csv",
    path + "/EES-2023-10.csv",
    path + "/EES-2024-05.csv",
]

dfs = [pd.read_csv(f) for f in files]

# ------------------------------------------------------------------
# 2. Concatenate into one DataFrame
# ------------------------------------------------------------------
data = pd.concat(dfs, ignore_index=True)

# --------------------------------------------------------------
# 3. Drop the non-numerical column
# --------------------------------------------------------------
data = data.drop(columns=["Which department do you work in?"])

# ------------------------------------------------------------------
# 4. Fit K-means with 5 clusters
# ------------------------------------------------------------------
kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
X = data.values
kmeans.fit(X)

# ------------------------------------------------------------------
# 5. Print centroids
# ------------------------------------------------------------------
centroids_std = kmeans.cluster_centers_         

print("Cluster centroids (original feature scale):")
for idx, c in enumerate(centroids_std, start=1):
    print(f"Cluster {idx}: {c}")
