import pickle
import numpy as np
import json
# Need to remove python metadata from pickle and convert to json for Lucene

# Load the embeddings from the pickle file
with open(r'E:\IR Project\Embeddings\sbert2\doccument_embeddings_SBERT2.pkl', 'rb') as f:
    embeddings = pickle.load(f)

if isinstance(embeddings, np.ndarray): # Convert numpy to list
    converted_emb = embeddings.tolist()

    # Convert it to a dictionary using numeric indices as keys
    converted_emb_dict = {str(i): converted_emb[i] for i in range(len(converted_emb))}

elif isinstance(embeddings, dict): # Convert dict to list
    # Update keys to be numeric indices starting from 0 to len(embeddings) - 1
    converted_emb_dict = {str(i): value.tolist() if isinstance(value, np.ndarray) else value
                                 for i, (key, value) in enumerate(embeddings.items())}

else:
    raise TypeError("Embeddi")

# Save as json
with open('E:\\IR Project\\Embeddings\\embeddings_indexed.json', 'w') as f:
    json.dump(converted_emb_dict, f, indent=4)