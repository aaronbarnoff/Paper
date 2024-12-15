from transformers import AutoTokenizer, AutoModel
import torch
import os
import glob
import re
import pickle
import numpy as np
from sklearn.preprocessing import normalize

# Stella: https://huggingface.co/dunzhang/stella_en_400M_v5
docs_folder = r"E:\\IR Project\\Crawler\\arxivCrawler2\\arxivPapers"
docs_path = "docs_stella.pkl"
embedding_path = "doc_emb_stella.pkl"
k = 10000
batch_size = 8  
vector_dim = 256 # Default 1024, others exist too https://huggingface.co/dunzhang/stella_en_400M_v5/tree/main/2_Dense_256

device = "cuda" if torch.cuda.is_available() else "cpu"
model_identifier = r"dunzhang/stella_en_400M_v5"

tokenizer = AutoTokenizer.from_pretrained(model_identifier, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModel.from_pretrained(model_identifier, trust_remote_code=True, output_hidden_states=True).to(device)

vector_linear = torch.nn.Linear(in_features=model.config.hidden_size, out_features=vector_dim)
vector_linear_path = os.path.join(model_identifier, r"E:\IR Project\Embeddings\stella\test\2_Dense_256\pytorch_model.bin") # I used 2_Dense_256
vector_linear_dict = torch.load(vector_linear_path)
vector_linear_dict_modified = {k.replace("linear.", ""): v for k, v in vector_linear_dict.items()} # Labels were wrong
vector_linear.load_state_dict(vector_linear_dict_modified)
vector_linear.to(device)

def remove_latex(text):
    text = re.sub(r'\\[a-zA-Z]+(?:\\[[^\]]*\])?(?:\\{[^}]*\\})?', '', text)
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

def embed_documents(batch_docs, model, tokenizer):
    inputs = tokenizer(batch_docs, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']

        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        batch_embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        batch_embeddings = vector_linear(batch_embeddings)
        batch_embeddings = normalize(batch_embeddings.cpu().numpy())

    return batch_embeddings

def process_batches(docs_folder, k, batch_size):
    documents = []
    document_embeddings = []
    current_batch = []

    for i, file_path in enumerate(glob.glob(os.path.join(docs_folder, "*.txt"))):
        if i >= k:
            break

        document = process_document(file_path)
        documents.append(document)
        current_batch.append(document)

        if len(current_batch) == batch_size:
            document_embeddings.extend(embed_documents(current_batch, model, tokenizer))
            current_batch = []

    if current_batch:
        document_embeddings.extend(embed_documents(current_batch, model, tokenizer))

    return documents, document_embeddings

def save_data(documents, document_embeddings, output_paths):
    docs_path, embedding_path = output_paths

    with open(docs_path, "wb") as f:
        pickle.dump(documents, f)

    with open(embedding_path, "wb") as f:
        pickle.dump(document_embeddings, f)

def main():
    documents, document_embeddings = process_batches(docs_folder, k, batch_size)

    if document_embeddings:
        document_embeddings = np.array(document_embeddings)
    else:
        raise ValueError("Embedding failed.")

    save_data(documents, document_embeddings, (docs_path, embedding_path))

    print(f"Docs and embedding saved to {docs_path} and {embedding_path}")

if __name__ == "__main__":
    main()
