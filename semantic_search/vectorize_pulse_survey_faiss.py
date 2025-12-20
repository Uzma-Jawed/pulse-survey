import mysql.connector
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import os
import pickle
from datetime import datetime

# ==================== CONFIG ====================
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'december25*',
    'database': 'osprmuti_pulse_survey'
}

FAISS_INDEX_FILE = 'pulse_survey_faiss.index'
METADATA_FILE = 'pulse_survey_metadata.pkl'
LAST_TIMESTAMP_FILE = 'last_timestamp.txt'

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # 384-dimensional, fast & good
DIMENSION = 384

# ==================== LOAD MODEL ====================
model = SentenceTransformer(EMBEDDING_MODEL)

# ==================== CONNECT TO MYSQL ====================
conn = mysql.connector.connect(**MYSQL_CONFIG)
cursor = conn.cursor()

# ==================== LOAD OR CREATE FAISS INDEX ====================
if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(METADATA_FILE):
    print("Loading existing FAISS index...")
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(METADATA_FILE, 'rb') as f:
        metadata_list = pickle.load(f)
else:
    print("Creating new FAISS index...")
    index = faiss.IndexFlatL2(DIMENSION)  # L2 distance (you can change to IP for cosine)
    metadata_list = []

# ==================== GET LAST PROCESSED TIMESTAMP ====================
if os.path.exists(LAST_TIMESTAMP_FILE):
    with open(LAST_TIMESTAMP_FILE, 'r') as f:
        last_timestamp = f.read().strip()
else:
    last_timestamp = '2025-01-01 00:00:00'  # Initial run

# ==================== FETCH NEW ANSWERS ====================
query = """
SELECT 
    psa.id AS answer_id,
    psa.company_id,
    psa.user_id,
    psa.question_id,
    psq.name AS question_text,
    psq.type AS question_type,
    psa.scale_rating,
    psa.binary_answer,
    psa.open_ended_answer,
    psa.nps_style_rating,
    psa.created_at
FROM pulse_survey_answers psa
JOIN pulse_survey_questions psq ON psa.question_id = psq.id
WHERE psa.created_at > %s
ORDER BY psa.created_at
"""

cursor.execute(query, (last_timestamp,))
rows = cursor.fetchall()

if not rows:
    print("No new answers to process.")
else:
    df = pd.DataFrame(rows, columns=[
        'answer_id', 'company_id', 'user_id', 'question_id',
        'question_text', 'question_type', 'scale_rating',
        'binary_answer', 'open_ended_answer', 'nps_style_rating', 'created_at'
    ])

    print(f"Found {len(df)} new answers. Processing...")

    # ==================== CREATE TEXT FOR EMBEDDING ====================
    
    def create_search_text(row):
     answer = (
        row['binary_answer'] or
        row['scale_rating'] or
        row['open_ended_answer'] or
        row['nps_style_rating'] or
        "No response"
    )
    # Better formatting for semantic understanding
     return (
        f"Employee survey response: "
        f"Question - {row['question_text']}. "
        f"Answer given - {answer}. "
        f"Question type: {row['question_type']}. "
        f"Date: {row['created_at'].date()}"
    )

    df['search_text'] = df.apply(create_search_text, axis=1)

    # ==================== GENERATE EMBEDDINGS ====================
    texts = df['search_text'].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # ==================== ADD TO FAISS INDEX ====================
    index.add(embeddings)
    print(f"Added {len(embeddings)} vectors to FAISS index. Total vectors: {index.ntotal}")

    # ==================== SAVE METADATA ====================
    for _, row in df.iterrows():
        metadata = {
            'answer_id': int(row['answer_id']),
            'company_id': int(row['company_id']),
            'user_id': int(row['user_id']),
            'question_text': row['question_text'],
            'question_type': row['question_type'],
            'answer': row['binary_answer'] or row['scale_rating'] or row['open_ended_answer'] or row['nps_style_rating'],
            'created_at': str(row['created_at']),
            'search_text': row['search_text']
        }
        metadata_list.append(metadata)

    # ==================== SAVE INDEX & METADATA ====================
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(metadata_list, f)

    # ==================== UPDATE LAST TIMESTAMP ====================
    latest_timestamp = df['created_at'].max()
    with open(LAST_TIMESTAMP_FILE, 'w') as f:
        f.write(str(latest_timestamp))

    print(f"Updated index. Latest timestamp: {latest_timestamp}")

# ==================== TEST SEARCH ====================
print("\n" + "="*50)
print("TESTING VECTOR SEARCH")
print("="*50)

query_text = "employees satisfied with work"  # Change this to test
query_embedding = model.encode([query_text])
query_embedding = np.array(query_embedding).astype('float32')

# Search top 5 similar responses
k = 5
distances, indices = index.search(query_embedding, k)

print(f"Query: '{query_text}'\n")
for i, idx in enumerate(indices[0]):
    if idx != -1:  # Valid index
        meta = metadata_list[idx]
        dist = distances[0][i]
        print(f"Rank {i+1} (Distance: {dist:.4f})")
        print(f"   Question: {meta['question_text']}")
        print(f"   Answer: {meta['answer']}")
        print(f"   Type: {meta['question_type']} | User: {meta['user_id']}")
        print(f"   Date: {meta['created_at']}\n")
    else:
        print(f"Rank {i+1}: No result")

cursor.close()
conn.close()
print("Done!")