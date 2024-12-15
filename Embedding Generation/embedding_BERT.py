from transformers import BertModel, BertTokenizer
import torch
import torch.nn.functional as F
import os
import glob
import re
import pickle
import numpy as np

# Parameters
docs_folder = "E:\\IR Project\\Crawler\\arxivCrawler2\\arxivPapers"
docs_name = "docs.pkl"
embeddings_name = "document_embeddings_bert.pkl"
k = 10000              # Docs
layers_option = 'last'  # last, first, average
pooling_option = 'cls'  # cls, mean
batch_size = 8  

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Freeze layers to speed up embedding process
for param in model.parameters():
    param.requires_grad = False

def remove_latex(text):
    text = re.sub(r'\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^\}]*\})?', '', text)
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

def embedding_batch(documents, model, tokenizer, pooling='cls', layers='last'):
    inputs = tokenizer(documents, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    if layers == 'last':
        hidden_states = outputs.last_hidden_state
    elif layers == 'first':
        hidden_states = outputs.hidden_states[0]
    elif layers == 'average':
        hidden_states = torch.mean(torch.stack(outputs.hidden_states), dim=0)
    else:
        raise ValueError("Invalid layer.")
    
    if pooling == 'cls':
        cls_embeddings = hidden_states[:, 0, :]
        return cls_embeddings.cpu().numpy()
    elif pooling == 'mean':
        mean_embeddings = torch.mean(F.normalize(hidden_states, p=2, dim=2), dim=1) # Normalize
        return mean_embeddings.cpu().numpy()
    else:
        raise ValueError("Invalid pooling.")

def process_batches(docs_folder, k, batch_size):
    documents = []
    document_embeddings = []
    batch_docs = []

    for i, filePath in enumerate(glob.glob(os.path.join(docs_folder, "*.txt"))):
        if i >= k:
            break

        document = process_document(filePath)
        documents.append(document)
        batch_docs.append(document)

        # Process batch if it's full
        if len(batch_docs) == batch_size:
            document_embeddings.extend(process_batch(batch_docs))
            batch_docs = []

    if batch_docs:
        document_embeddings.extend(process_batch(batch_docs))

    return documents, document_embeddings

def process_batch(batch_docs):
    document_embeddings_batch = embedding_batch(batch_docs, model, tokenizer, pooling=pooling_option, layers=layers_option)
    return document_embeddings_batch

def save_data(documents, document_embeddings, output_paths):
    docs_path, embeddings_path = output_paths

    with open(docs_path, "wb") as f:
        pickle.dump(documents, f)

    with open(embeddings_path, "wb") as f:
        pickle.dump(document_embeddings, f)

def main():
    documents, document_embeddings = process_batches(docs_folder, k, batch_size)
    
    if document_embeddings:
        document_embeddings = np.array(document_embeddings) # Need embeddings in 2D array
    else:
        raise ValueError("Embedding failed.")

    save_data(documents, document_embeddings, (docs_name, embeddings_name))

    print(f"Docs and embeddings saved to {docs_name} and {embeddings_name}")

if __name__ == "__main__":
    main()
