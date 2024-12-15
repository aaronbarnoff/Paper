from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import os
import glob
import re
import pickle
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"

# SBERT Model
#modelName = 'sentence-transformers/all-MiniLM-L6-v2' # SBERT2
#modelNameShort = "SBERT1"
modelName = 'sentence-transformers/all-mpnet-base-v2' # SBERT2 https://huggingface.co/sentence-transformers/all-mpnet-base-v2
modelNameShort = "SBERT2"

# Parameters
docs_folder = r"E:\IR Project\Crawler\arxivCrawler2\arxivPapers"
docs_name = "docs.pkl"
embeddings_name = f"doc_emb_{modelNameShort}.pkl"
k = 10000  # Number of documents
batch_size = 4  # Document batch size

# Load SBERT model from Hugging Face (and using their recommended settings and code)
tokenizer = AutoTokenizer.from_pretrained(modelName)
model = AutoModel.from_pretrained(modelName).to(device)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def remove_latex(text):
    # Remove LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+(?:\\[[^\\]]*\\])?(?:\\{[^\\}]*\\})?', '', text)
    # Remove inline math
    text = re.sub(r'\$.*?\$', '', text)
    text = re.sub(r'\\\\\\((.*?)\\\\\\)', '', text)
    # Remove display math
    text = re.sub(r'\\\\\\[(.*?)\\\\\\]', '', text)
    text = re.sub(r'\\\\begin\\{.*?\\}.*?\\\\end\\{.*?\\}', '', text, flags=re.DOTALL)
    return text

# Get title and abstract to embed
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

def batch_embeddings(batch_docs):
    try:
        # Tokenize documents
        encoded_input = tokenizer(batch_docs, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform mean pooling
        batch_doc_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings (L2 norm)
        batch_doc_embeddings = F.normalize(batch_doc_embeddings, p=2, dim=1)

        return batch_doc_embeddings.cpu().numpy()
    except Exception as e:
        print(f"Error processing batch: {e}")
        return None

def last_batch(batch_docs):
    if batch_docs:
        return batch_embeddings(batch_docs)
    return None

# Create emebedding
documents = []
document_embeddings = []
batch_docs = []

for i, filePath in enumerate(glob.glob(os.path.join(docs_folder, "*.txt"))):
    print(f'{i}, {filePath}')
    if i >= k:
        break

    doc = process_document(filePath)
    if doc:
        documents.append(doc)
        batch_docs.append(doc)

    # Process in batches
    if len(batch_docs) == batch_size:
        batch_embeddings = batch_embeddings(batch_docs)
        if batch_embeddings is not None:
            document_embeddings.extend(batch_embeddings)

        batch_docs = []

# Process any remaining documents in last batch
last_embeddings = last_batch(batch_docs)
if last_embeddings is not None:
    document_embeddings.extend(last_embeddings)

# Need embeddings as 2D array
if document_embeddings:
    document_embeddings = np.array(document_embeddings)
else:
    raise ValueError("Embedding failed.")

# Save the documents and embeddings to pickle
with open(docs_name, "wb") as f:
    pickle.dump(documents, f)

with open(embeddings_name, "wb") as f:
    pickle.dump(document_embeddings, f)

print(f"Docs and embeddings saved to {docs_name} and {embeddings_name}")
