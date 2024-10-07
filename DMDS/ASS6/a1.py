import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('CC GENERAL.csv')

# Preprocessing: Handle missing values (if any)
data.fillna(data.mean(), inplace=True)

# Select features for clustering
features = data.iloc[:, 1:]  # Exclude 'CUST_ID' or any identifier column

# Scale the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(scaled_features)

# Add cluster labels to the original data
data['Cluster'] = kmeans.labels_

# Visualization of the clusters (example using two principal components)
from sklearn.decomposition import PCA

pca = PCA(2)
pca_features = pca.fit_transform(scaled_features)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=data['Cluster'], palette='viridis')
plt.title('K-means Clustering')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.show()

# Print the cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)
