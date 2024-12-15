from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os
import glob
import re
import pickle
import numpy as np
import torch
import torch.nn.functional as F
# SBERT Model
#modelName = 'sentence-transformers/all-MiniLM-L6-v2' # SBERT2
#modelNameShort = "SBERT1"
modelName = 'sentence-transformers/all-mpnet-base-v2' # SBERT2 https://huggingface.co/sentence-transformers/all-mpnet-base-v2
modelNameShort = "SBERT2"

# Parameters
docs_folder = r"E:\IR Project\Crawler\arxivCrawler2\arxivPapers"
output_pickle_path_docs = "docs.pkl"
k = 10000  # Number of documents
batch_size = 16  # Batch size
learning_rate = 2e-5  
epochs = 5  
output_path = f"./fine_tuned_{modelNameShort}"

# Load SBERT model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(modelName, device=device)

def remove_latex(text):
    text = re.sub(r'\\[a-zA-Z]+(?:\\[[^\\]]*\\])?(?:\\{[^\\}]*\\})?', '', text)  
    text = re.sub(r'\$.*?\$', '', text) 
    text = re.sub(r'\\\((.*?)\\\)', '', text) 
    text = re.sub(r'\\\[(.*?)\\\]', '', text) 
    text = re.sub(r'\\begin\\{.*?\\}.*?\\end\\{.*?\\}', '', text, flags=re.DOTALL) 
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

def generate_embeddings(batch_docs):
    try:
        batch_doc_embeddings = model.encode(batch_docs, convert_to_tensor=True, show_progress_bar=False)
        normalized_embeddings = F.normalize(batch_doc_embeddings, p=2, dim=1)  # L2 norm
        return normalized_embeddings.cpu().numpy() 
    except Exception as e:
        print(f"Error processing batch: {e}")
        return None

def process_docs():
    documents = []
    labels = [] 
    batch_docs = []
    
    for i, file_path in enumerate(glob.glob(os.path.join(docs_folder, "*.txt"))):
        if i >= k:
            break
        print(f'{i}, {file_path}')
        
        doc = process_document(file_path)
        if doc:
            documents.append(doc)
            batch_docs.append(doc)
            labels.append(0.5)  # Default similarity score 

        # Process batch 
        if len(batch_docs) == batch_size:
            batch_embeddings = generate_embeddings(batch_docs)
            if batch_embeddings is not None:
                batch_docs = []  # Clear after processing

    # Process remaining docs
    if batch_docs:
        generate_embeddings(batch_docs)
    
    return documents, labels

def prepare_data(documents, labels):
    # Create pairs of docs with their labels and similarity score
    train_examples = [InputExample(texts=[documents[i], documents[i+1]], label=float(labels[i])) for i in range(0, len(documents)-1, 2)]
    # Create batches from the training examples, shuffle helps to avoid overfitting, larger batch size makes training better
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    # They say to use cosine similarity loss function, it tries to minimize the error between predicted/target similarity scores in docs
    train_loss = losses.CosineSimilarityLoss(model)
    return train_dataloader, train_loss

def fine_tune(train_dataloader, train_loss):
    model.fit( # From the sentenceTransformer library, fine-tune SBERT based on dataset and loss function
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100, # They might have recommended 500 warmup steps
        optimizer_params={'lr': learning_rate},
        output_path=output_path,
        show_progress_bar=True
    )

def main():
    # Process documents, then prepare it for training, then fine tune it and save.
    documents, labels = process_docs()
    train_dataloader, train_loss = prepare_data(documents, labels)
    fine_tune(train_dataloader, train_loss)
    model.save(output_path)
    print("Fine-tuning complete.")

if __name__ == "__main__":
    main()
