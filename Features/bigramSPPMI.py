import os
import re
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, save_npz
from collections import Counter
from itertools import tee
from scipy.sparse import lil_matrix

# Parameters
docs_folder = r"E:\IR Project\Crawler\arxivCrawler2\arxivPapersBig"
output_path = r"E:\IR Project\Embeddings\bigram_terms.json"
window_size = 10  # PMI window, probably want 6-10 size to capture bigrams like "neural network" and "artificial intelligence"
min_word_frequency = 40 
shift = 5  # SPPMI shift

def remove_latex(text):
    text = re.sub(r'\\[a-zA-Z]+(?:\\[[^\\]]*\\])?(?:\\{[^}]*\\})?', '', text)
    text = re.sub(r'\$.*?\$', '', text)
    text = re.sub(r'\\\((.*?)\\\)', '', text)
    text = re.sub(r'\\\[(.*?)\\\]', '', text)
    text = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', '', text, flags=re.DOTALL)
    return text

documents = []

#def ngrams(tokens, n):
#    tokens, tokens_copy = tee(tokens)
#    return zip(*[tokens_copy] * n)

for file_name in os.listdir(docs_folder):
    if file_name.endswith(".txt"):
        file_path = os.path.join(docs_folder, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            arxivID, title, abstract = "", "", ""
            for line in file:
                if "Title:" in line:
                    title = remove_latex(line.removeprefix("Title:").strip()).lower()
                elif "Abstract:" in line:
                    abstract = remove_latex(line.removeprefix("Abstract:").strip()).lower()
            doc = title + " " + abstract
            documents.append(doc)

# Tokenize and create matrix
vectorizer = CountVectorizer(stop_words='english', min_df=min_word_frequency, ngram_range=(2, 2))  # Only get bigrams
X = vectorizer.fit_transform(documents)
vocab = vectorizer.get_feature_names_out()

vocab_size = len(vocab)
co_occurrence_matrix = lil_matrix((vocab_size, vocab_size), dtype=np.float64) # They say csr_matrix is bad to use; this one is sparse

dbg_cnt = 0
max_cnt = 25000
for doc in documents:
    # print(doc)
    if dbg_cnt == max_cnt:
        break
    bigrams = vectorizer.build_analyzer()(doc)
    token_indices = [np.where(vocab == bigram)[0][0] for bigram in bigrams if bigram in vocab]
    for idx, token_idx in enumerate(token_indices):
        start = max(idx - window_size, 0)
        end = min(idx + window_size + 1, len(token_indices))
        
        for neighbor_position in range(start, end):
            if neighbor_position != idx:  # Ensure we're not comparing the bigram to itself
                neighbor_idx = token_indices[neighbor_position]
                
                # Ignore immediately adjacent bigrams
                if abs(idx - neighbor_position) > 1:
                    co_occurrence_matrix[token_idx, neighbor_idx] += 1

    dbg_cnt += 1

co_occurrence_matrix = co_occurrence_matrix.tocsr() # Need to convert back 
total_occurrences = co_occurrence_matrix.sum()
bigram_frequency = co_occurrence_matrix.sum(axis=1).A1 # Frequency for each bigram

rows, cols = co_occurrence_matrix.nonzero()
SPPMI_matrix = lil_matrix((vocab_size, vocab_size), dtype=np.float64)

for i, j in zip(rows, cols):
    p_ij = co_occurrence_matrix[i, j] / total_occurrences # Joint probablity
    p_i = bigram_frequency[i] / total_occurrences # Individual
    p_j = bigram_frequency[j] / total_occurrences
    pmi = np.log2(p_ij / (p_i * p_j)) - shift # Shift for SPPMI
    SPPMI_matrix[i, j] = max(pmi, 0) # SPPMI = (pmi-shift); keep positive only
SPPMI_matrix = SPPMI_matrix.tocsr()

expanded_query_dict = {}

for i, bigram in enumerate(vocab):
    sppmi_values = SPPMI_matrix.getrow(i).toarray().flatten() # SPPMI for current bigram
    most_similar_indices = sppmi_values.argsort()[-3:][::-1] # Record the top 3 most simlar bigrams
    related_terms = [vocab[j] for j in most_similar_indices if sppmi_values[j] > 0 and vocab[j] != bigram] # Don't add itself to the list
    if related_terms:
        expanded_query_dict[bigram] = related_terms

expanded_query_dict = {k: v for k, v in expanded_query_dict.items() if v} # Remove empty bigrams

with open(output_path, 'w') as f:
    json.dump(expanded_query_dict, f, indent=4)

print(f"Expanded query terms saved to {output_path}")
