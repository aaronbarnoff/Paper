import os
import glob
import re
import pickle
import numpy as np
from collections import defaultdict
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from scipy.sparse import lil_matrix

# I thought I should qickly add this in last-minute since most of the work was 
# already done from other features, but I didn't have time to compare it against other embeddings.

# Remove stopwords, and using the punkt tokenizer 
# nltk.download()
nltk.download('punkt')
nltk.download('stopwords')

# Parameters
docs_folder = "E:\\IR Project\\Crawler\\arxivCrawler2\\arxivPapers"
docs_name = "docs.pkl"
embeddings_name = "document_embeddings_svd.pkl"
k = 10000              # Number of Documents to Process
window_size = 5        # Co-occurrence window size
embedding_dim = 300    # SVD dimension, similar size to w2vec
max_vocab_size = 10000 
batch_size = 1000      

# Initialize stopwords
stop_words = set(stopwords.words('english'))

def remove_latex(text):
    text = re.sub(r'\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^}]*\})?', '', text)
    text = re.sub(r'\$.*?\$', '', text) 
    text = re.sub(r'\\\((.*?)\\\)', '', text) 
    text = re.sub(r'\\\[(.*?)\\\]', '', text) 
    text = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', '', text, flags=re.DOTALL)
    return text


def process_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        arxivID, title, abstract = "", "", ""
        
        for line in file:
            if "Arxiv_ID:" in line:
                arxivID = remove_latex(line.removeprefix("Arxiv_ID:").strip().removeprefix("http://arxiv.org/abs/")).lower()
            elif "Title:" in line:
                title = remove_latex(line.removeprefix("Title:").strip()).lower()
            elif "Abstract:" in line:
                abstract = remove_latex(line.removeprefix("Abstract:").strip()).lower()
        
        doc = title + " " + abstract
        return doc

# Build tokens 
def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return tokens

def build_vocabulary(documents, max_vocab_size=10000):
    freq = defaultdict(int)
    for doc in documents:
        tokens = tokenize(doc)
        for token in tokens:
            freq[token] += 1
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True) # Sort by frequency
    vocab = [token for token, count in sorted_tokens[:max_vocab_size]] # Get vocabulary size
    return vocab

def build_COOC_matrix(documents, vocab, window_size=5):
    vocab_size = len(vocab)
    word_to_id = {word: idx for idx, word in enumerate(vocab)}
    cooc_matrix = lil_matrix((vocab_size, vocab_size), dtype=np.float32)
    
    for doc_idx, doc in enumerate(documents):
        tokens = tokenize(doc)
        token_ids = [word_to_id[token] for token in tokens if token in word_to_id]
        for i, token_id in enumerate(token_ids):
            start = max(i - window_size, 0)
            end = min(i + window_size + 1, len(token_ids))
            for j in range(start, end):
                if i == j:
                    continue
                context_id = token_ids[j]
                cooc_matrix[token_id, context_id] += 1.0
    return cooc_matrix.tocsr()

def get_document_embeddings(documents, word_vectors, word_to_id):
    embeddings = []
    for doc_idx, doc in enumerate(documents):
        tokens = tokenize(doc)
        vectors = [word_vectors[word_to_id[token]] for token in tokens if token in word_to_id]
        if vectors:
            doc_embedding = np.mean(vectors, axis=0)
        else:
            doc_embedding = np.zeros(word_vectors.shape[1], dtype=np.float32)
        embeddings.append(doc_embedding)
    return embeddings

def save_data(documents, document_embeddings, output_paths):
    docs_path, embeddings_path = output_paths

    with open(docs_path, "wb") as f:
        pickle.dump(documents, f)

    with open(embeddings_path, "wb") as f:
        pickle.dump(document_embeddings, f)

def main():
    documents = []
    for i, filePath in enumerate(glob.glob(os.path.join(docs_folder, "*.txt"))):
        if i >= k:
            break
        document = process_document(filePath)
        documents.append(document)

    vocab = build_vocabulary(documents, max_vocab_size=max_vocab_size)
    print(f"Vocabulary size: {len(vocab)}") # NumDocs

    cooc_matrix = build_COOC_matrix(documents, vocab, window_size=window_size)

    svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
    word_vectors = svd.fit_transform(cooc_matrix)

    word_vectors = normalize(word_vectors, axis=1) # Normalize word vectors
    word_to_id = {word: idx for idx, word in enumerate(vocab)}

    document_embeddings = get_document_embeddings(documents, word_vectors, word_to_id)
    document_embeddings = np.array(document_embeddings)
    save_data(documents, document_embeddings, (docs_name, embeddings_name))

    print(f"Saved {docs_name}, {embeddings_name}")

if __name__ == "__main__":
    main()