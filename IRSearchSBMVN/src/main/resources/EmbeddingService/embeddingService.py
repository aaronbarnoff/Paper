from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

app = Flask(__name__)
CORS(app)  # Allow all cross-origin requests
model = SentenceTransformer('all-mpnet-base-v2') # SBERT2

logging.basicConfig(level=logging.DEBUG)

@app.route('/generateEmbedding', methods=['GET'])
def generate_embedding():
    text = request.args.get('text') # From HTTP header
    if not text:
        app.logger.debug('No text in request.')
        return jsonify({'error': 'No text'}), 400

    app.logger.debug(f'Generating embedding for: "{text}"')

    try:
        embedding = model.encode(text).tolist()
        # We need to normalize the query embedding for cosine similarity against the normalized doc. embedding
        norm = np.linalg.norm(embedding) 
        if norm > 0:
            embedding = embedding / norm
        
        # Py np array to JSON list 
        embedding_list = embedding.tolist()
        
        app.logger.debug(f'Embedding: {embedding[:5]}...')
        return jsonify({"embedding": embedding_list})
    except Exception as e:
        app.logger.error(f'Error generating embedding: {e}')
        return jsonify({'error': 'Failed to generate embedding'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) # Listen on port 5000 for the query from java