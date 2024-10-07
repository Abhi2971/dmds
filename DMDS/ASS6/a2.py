import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the dataset
data = pd.read_csv('Customer.csv')

# Preprocessing: Handle missing values (if any)
data.fillna(data.mean(), inplace=True)

# Select features for clustering (excluding non-numeric columns)
features = data.select_dtypes(include=[np.number])

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Perform Hierarchical Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
data['Cluster'] = agg_clustering.fit_predict(scaled_features)

# Visualization of the clusters using dendrogram
plt.figure(figsize=(10, 7))
linkage_matrix = linkage(scaled_features, method='ward')
dendrogram(linkage_matrix)
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Visualization using scatter plot (example with first two features)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=scaled_features[:, 0], y=scaled_features[:, 1], hue=data['Cluster'], palette='viridis')
plt.title('Hierarchical Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
