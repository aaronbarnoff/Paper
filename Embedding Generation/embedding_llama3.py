from transformers import AutoTokenizer, AutoModel
import torch
import os
import glob
import re
import pickle
import numpy as np

# https://huggingface.co/meta-llama/Llama-3.2-1B

device = "cuda" if torch.cuda.is_available() else "cpu"
token = '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' # You might need an access token on HuggingFace
model_identifier = 'meta-llama/Llama-3.2-1B'

docs_folder = r"E:\\IR Project\\Crawler\\arxivCrawler2\\arxivPapers"
docs_path = "docs.pkl"
embedding_path = "doc_emb_llama3.pkl"
k = 1000
batch_size = 2
pooling = 'mean'

tokenizer = AutoTokenizer.from_pretrained(model_identifier, token=token)
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
model = AutoModel.from_pretrained(model_identifier, token=token, output_hidden_states=True).to(device)

for param in model.parameters():
    param.requires_grad = False # Freeze model parameters, we aren't training the model (faster)

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
        return title + " " + abstract

# For each batch of documents:
def doc_embedding(documents, model, tokenizer, pooling=pooling):
    # Tokenize the documents to a max length of 512, with padding to ensure equal dimension
    inputs = tokenizer(documents, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad(): # Disable gradient calculations (faster)
        # Pass the tokens to the model and get the hidden states
        outputs = model(**inputs)

    # Get the final layer of the hidden states (this has the semantic relationships)
    hidden_states = outputs.hidden_states[-1]
    if pooling == 'cls':
        cls_embedding = hidden_states[:, 0, :]
        embedding = cls_embedding
    elif pooling == 'mean':
        mean_embedding = hidden_states.mean(dim=1)
        embedding = mean_embedding
    else:
        raise ValueError("Invalid pooling.") # Make IDE happy

    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    return embedding.cpu().numpy()

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
            document_embeddings.extend(doc_embedding(current_batch, model, tokenizer, pooling='mean'))
            current_batch = []

    if current_batch:
        document_embeddings.extend(doc_embedding(current_batch, model, tokenizer, pooling='mean'))

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
        raise ValueError("Embedding error.")

    save_data(documents, document_embeddings, (docs_path, embedding_path))

    print(f"Docs and embeddings saved to {docs_path} and {embedding_path}")

if __name__ == "__main__":
    main()
