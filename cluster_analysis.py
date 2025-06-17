"""
cluster_analysis.py
--------------------
This script performs KMeans clustering on precipitation climatology data using monthly precipitation means.
It generates a cluster assignment per station and visualizes the result spatially.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

data_path = "data/glob_1993a2023_mo_join_filtered.xlsx"
output_path = "outputs/clustered_data.xlsx"
n_clusters = 4

df = pd.read_excel(data_path)
df.columns = df.columns.str.replace(" ", "_")

monthly_clim = df.groupby(['latitude', 'longitude', 'month_x'])['pr_x'].mean().unstack()
monthly_clim = monthly_clim.fillna(0)

scaler = StandardScaler()
X = scaler.fit_transform(monthly_clim)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

df['cluster'] = df.set_index(['latitude', 'longitude']).index.map(
    dict(zip(monthly_clim.index, clusters))
)

os.makedirs("outputs", exist_ok=True)
df.to_excel(output_path, index=False)

sns.scatterplot(data=df.drop_duplicates(['latitude', 'longitude']),
                x='longitude', y='latitude', hue='cluster', palette='tab10')
plt.title("Clusters based on precipitation climatology")
plt.savefig("outputs/cluster_map.png", dpi=300)
plt.close()
