import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load everything
index = faiss.read_index('pulse_survey_faiss.index')
with open('pulse_survey_metadata.pkl', 'rb') as f:
    metadata_list = pickle.load(f)
model = SentenceTransformer('all-MiniLM-L6-v2')

while True:
    query = input("\nEnter your search query (or 'quit' to exit): ")
    if query.lower() == 'quit':
        break
    
    q_emb = model.encode([query])
    q_emb = np.array(q_emb).astype('float32')
    
    D, I = index.search(q_emb, k=10)
    
    print(f"\nTop results for: '{query}'")
    for i in range(len(I[0])):
        if I[0][i] != -1:
            meta = metadata_list[I[0][i]]
            print(f"{i+1}. Score: {D[0][i]:.3f} | {meta['answer']} → {meta['question_text']} (User {meta['user_id']}, {meta['created_at']})")