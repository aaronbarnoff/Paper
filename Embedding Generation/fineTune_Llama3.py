from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import os
import glob
import re
import pickle
import numpy as np
from datasets import load_dataset, Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
token = 'xxx' # Requires meta authentication token
model_identifier = 'meta-llama/Llama-3.2-1B'

torch.cuda.set_device(0)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_identifier, token=token)
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token}) 

# Load the model using AutoModel
model = AutoModelForSequenceClassification.from_pretrained(
    model_identifier,
    token=token,
    num_labels=2,  
    output_hidden_states=True  
).to(device)

# Function to remove LaTeX
def removeLatex(text):
    # Remove LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+(?:\\[[^\\]]*\\])?(?:\\{[^\\}]*\\})?', '', text)
    # Remove inline math
    text = re.sub(r'\$.*?\$', '', text)
    text = re.sub(r'\\\((.*?)\\\)', '', text)
    # Remove display math
    text = re.sub(r'\\\[(.*?)\\\]', '', text)
    text = re.sub(r'\\begin\\{.*?\\}.*?\\end\\{.*?\\}', '', text, flags=re.DOTALL)
    return text

# Parameters
docs_folder = r"E:\IR Project\Crawler\arxivCrawler2\arxivPapers"
k = 100  # Limit the number of documents to process
batch_size = 1  # Reduced batch size to handle padding issues

texts = []
labels = [] 

for i, filePath in enumerate(glob.glob(os.path.join(docs_folder, "*.txt"))):
    if i >= k:
        break
    with open(filePath, 'r', encoding='utf-8') as file:
        print(f"Processing file: {filePath}")
        arxivID, title, abstract = "", "", ""
        for line in file:
            if "Arxiv_ID:" in line:
                arxivID = removeLatex(line.removeprefix("Arxiv_ID:").strip().removeprefix("http://arxiv.org/abs/")).lower()
            elif "Title:" in line:
                title = removeLatex(line.removeprefix("Title:").strip()).lower()
            elif "Abstract:" in line:
                abstract = removeLatex(line.removeprefix("Abstract:").strip()).lower()
        doc = title + " " + abstract
        texts.append(doc)
        labels.append(0)

dataset = Dataset.from_dict({'text': texts, 'label': labels})

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the dataset into training and evaluation sets
dataset = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = dataset['train']
eval_dataset = dataset['test']

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=50,  # Trying every 50, but too low
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5, # Too high, switched to steps
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10,
    gradient_accumulation_steps=8,  # This simulates a larger batch size
    fp16=True  # Reduce memory usage and speed up training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained("./fine_tuned_llama3")
tokenizer.save_pretrained("./fine_tuned_llama3")

print("Fine-tuning complete.")
