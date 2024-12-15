import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
import imageio
import matplotlib.animation as animation
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

# Define paths
embeddings_dir = r"E:\IR Project\Embeddings\TSNE"  # Replace with the directory where embeddings are saved
output_dir = "tsne_visualizations"
os.makedirs(output_dir, exist_ok=True)

# Load embedding files
def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

# List of embedding files (add paths to the embeddings you have created)
embedding_files = [
    os.path.join(embeddings_dir, "document_embeddings_sbert2.pkl"),
]

# Load embeddings from a single file
emb = load_embeddings(embedding_files[0])
# Ensure all embeddings are the same size
if len(emb) > 0 and isinstance(emb[0], (list, np.ndarray)):
    embedding_size = len(emb[0])
    emb = [e for e in emb if len(e) == embedding_size]

# Convert to numpy array and normalize
embeddings = np.array(emb)
embeddings = normalize(embeddings, axis=1)

# Perform clustering to assign colors
# There are ~40 CS subcategories, so I chose 40
num_clusters = 40  # You can adjust the number of clusters based on your data
kmeans = KMeans(n_clusters=num_clusters, random_state=123)
cluster_labels = kmeans.fit_predict(embeddings)

# Hyperparameters to iterate over
perplexities = [5, 10, 30, 50, 100]
learning_rates = [50, 50, 100, 200, 500]
n_iters = [250, 500, 1000]

fig, ax = plt.subplots(figsize=(10, 6))

# Function to update the plot for each frame
def update(frame):
    ax.clear()
    perplexity = perplexities[frame % len(perplexities)]
    learning_rate = learning_rates[(frame // len(perplexities)) % len(learning_rates)]
    n_iter = n_iters[(frame // (len(perplexities) * len(learning_rates))) % len(n_iters)]

    tsne_params = {
        'n_components': 2,
        'perplexity': perplexity,
        'learning_rate': learning_rate,
        'max_iter': n_iter,
        'random_state': 123
    }

    tsne = TSNE(**tsne_params)
    embeddings_2d = tsne.fit_transform(embeddings)

    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
    ax.set_title(f't-SNE Visualization (Perplexity={perplexity}, Learning Rate={learning_rate}, Iterations={n_iter})')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.grid(True)
    if frame == 0:
        plt.colorbar(scatter, ax=ax, label='Cluster Label')

# Create animation
num_frames = len(perplexities) * len(learning_rates) * len(n_iters)
ani = animation.FuncAnimation(fig, update, frames=num_frames, repeat=True)

# Save animation as GIF
output_gif_path = os.path.join(output_dir, 'tsne_animation.gif')
ani.save(output_gif_path, writer='pillow', fps=2)

print(f"Animation saved at {output_gif_path}")
