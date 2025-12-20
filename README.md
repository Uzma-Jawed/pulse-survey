# Pulse Survey Semantic Search Engine

A powerful **semantic search system** built on real employee pulse survey data. This project transforms structured relational data from a MySQL database into a **vector database (FAISS)** to enable natural language search over survey responses — including binary (yes/no), scale ratings, and open-ended answers.

## Features
- Connects to real pulse survey schema
- Combines question text + answer + type + scoring into rich embeddings
- Uses **Sentence Transformers** (`all-MiniLM-L6-v2`)
- Stores and searches vectors locally with **FAISS**
- **Real-time incremental updates** — only processes new responses
- Interactive CLI search interface
- Secure configuration

## Tech Stack
- Python 3.10+
- MySQL
- sentence-transformers
- FAISS (Facebook AI Similarity Search)
- pandas, mysql-connector-python, numpy

## Project Structure
```
├── vectorize_pulse_survey_faiss.py     # Main pipeline: MySQL → Embeddings → FAISS (real-time sync)
├── search_interface.py                 # Interactive semantic search CLI
├── requirements.txt                    # Python dependencies                         
├── README.md                           # This file
└── (Auto-generated files - not committed)
    ├── pulse_survey_faiss.index        # FAISS vector index
    ├── pulse_survey_metadata.pkl       # Stored metadata (question, answer, user, date)
    └── last_timestamp.txt              # Tracks last processed record for incremental updates
```

## Setup Instructions

### 1. Setup MySQL Database
- Install [MySQL Server](https://dev.mysql.com/downloads/mysql/) and [MySQL Workbench](https://dev.mysql.com/downloads/workbench/)
- Create a new database: `osprmuti_pulse_survey`
- Import the full schema (SQL dump from your system) to create all tables.

### 2. Add Sample Data (or use live data)

### 3. Python Environment
```bash
python -m venv venv
venv\Scripts\activate #on windows
pip install -r requirements.txt
```

### 4. Secure Database Connection
**Never commit passwords!**  

### 5. Run the System

**Build/Update Vector Index**
```bash
python vectorize_pulse_survey_faiss.py
```
- First run: Processes all existing answers
- Later runs: Only new responses since last run (real-time ready)

**Interactive Semantic Search**
```bash
python search_interface.py
```
Try queries.

## Real-Time Usage
Schedule `vectorize_pulse_survey_faiss.py` to run every 5–10 minutes using:
- **Windows Task Scheduler**

This keeps the vector index updated as new survey responses arrive.

---

**Author:** Uzma Jawed   

⭐ Feel free to star and fork!  
Contributions and suggestions welcome.

---
