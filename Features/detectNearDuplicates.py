import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embeddings_path = r'E:\IR Project\Embeddings\embeddings_indexed.json'
clusters_path = r'E:\IR Project\Embeddings\embeddings_clusters.json'
ignore_list_path = 'ignore_list.txt'

with open(embeddings_path, 'r') as f:
    embeddings_dict = json.load(f)

with open(clusters_path, 'r') as f:
    document_clusters = json.load(f)

# Extract embeddings into array
embeddings = {key: np.array(value) for key, value in embeddings_dict.items()}

clusters = {}
for doc_id, cluster_id in document_clusters.items(): # Group embeddings by cluster
    if cluster_id not in clusters:
        clusters[cluster_id] = []
    clusters[cluster_id].append(doc_id)

# I think min should be 0.90? 
# 0.95: 41/10k. 0.9: 202/10k. 0.85: 865/10k. 0.8: 3220/10k 
similarity_threshold = 0.85 
near_duplicates = []

# Scan over all clusters and get cosine simlarities within each clutser
for cluster_id, doc_ids in clusters.items():
    if len(doc_ids) < 2:
        continue  
    cluster_embeddings = np.array([embeddings[doc_id] for doc_id in doc_ids]) # Get embedding for each doc in cluster
    similarity_matrix = cosine_similarity(cluster_embeddings) # Get cosine similarity score of each doc pair

    # Check each document in cluster for similarity threshold
    num_docs = len(doc_ids)
    for i in range(num_docs):
        for j in range(i + 1, num_docs):
            if similarity_matrix[i, j] > similarity_threshold:
                print(f"Near Duplicate: Doc1: {doc_ids[i]}, Doc2: {doc_ids[j]}, Score: {similarity_matrix[i,j]}")
                near_duplicates.append((doc_ids[i], doc_ids[j], similarity_matrix[i, j]))

with open(ignore_list_path, 'w') as f:
    for dup in near_duplicates:
        f.write(f"{dup[0]} {dup[1]} {dup[2]:.2f}\n")

print(f"Found {len(near_duplicates)} near-duplicates.")
print(f"Ignore list saved to '{ignore_list_path}'")