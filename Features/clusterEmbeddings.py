import json
from sklearn.cluster import KMeans
import numpy as np
embeddings_path = r'E:\IR Project\Embeddings\embeddings_indexed.json'
# Create cluster embedding labels for my Lucene find similar feature

with open(embeddings_path, 'r') as f:
    embeddings_dict = json.load(f)

# Extract embeddings to an array
embeddings = np.array([value for key, value in embeddings_dict.items()])

num_clusters = 100 # Approximately 40 subfields of CS papers, so I will try 100 to make it more fine-grained 

kmeans = KMeans(n_clusters=num_clusters, random_state=123) # K-Means clustering
kmeans.fit(embeddings)

# Get cluster labels
cluster_labels = kmeans.labels_ # labels_ is the cluster assignment for each document after training

# Save the clusterIDs, each doc's ID is its index in the folder
document_clusters = {i: int(cluster_labels[i]) for i in range(len(cluster_labels))}

with open('embeddings_clusters.json', 'w') as f:
    json.dump(document_clusters, f, indent=4)