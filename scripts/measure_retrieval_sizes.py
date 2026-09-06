import os
import sys
import sqlite3
import numpy as np

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Force environment config
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(project_root, ".env"))

import app.config as config
from app.retrieval.retriever import Retriever
from app.db.chroma_client import ChromaClient
from app.db import super_node_store

def count_words(text):
    return len(str(text).split())

def measure_sizes():
    # 1. Fetch 10 simple queries
    db_path = os.path.join(project_root, "data", "sqlite", "rag.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT question FROM benchmark_qa_simple LIMIT 10")
    questions = [row[0] for row in c.fetchall()]
    conn.close()

    static_totals = []
    dynamic_totals = []

    for idx, question in enumerate(questions):
        print(f"\n[{idx+1}/10] Query: {question}")
        
        # STATIC
        config.CHROMA_PERSIST_PATH = os.path.join(project_root, "data", "backups", "v1", "chroma")
        config.SQLITE_DB_PATH = os.path.join(project_root, "data", "backups", "v1", "rag.db")
        ChromaClient._client = None
        ChromaClient._collection = None
        ChromaClient._super_nodes_collection = None
        super_node_store._collection = None
        
        static_chunks = Retriever.search(question, top_k=3)
        static_sizes = [count_words(c['text']) for c in static_chunks]
        static_total = sum(static_sizes)
        static_totals.append(static_total)
        
        print("  -> STATIC (Raw Chunks)")
        for i, (chunk, size) in enumerate(zip(static_chunks, static_sizes)):
            print(f"     Chunk {i+1} [ID: {chunk['id'][:15]}...]: {size} words")
        print(f"     Total Static Words: {static_total}")

        # DYNAMIC
        config.CHROMA_PERSIST_PATH = os.path.join(project_root, "data", "chroma")
        config.SQLITE_DB_PATH = os.path.join(project_root, "data", "sqlite", "rag.db")
        ChromaClient._client = None
        ChromaClient._collection = None
        ChromaClient._super_nodes_collection = None
        super_node_store._collection = None
        
        dynamic_chunks = Retriever.search(question, top_k=3)
        dynamic_sizes = [count_words(c['text']) for c in dynamic_chunks]
        dynamic_total = sum(dynamic_sizes)
        dynamic_totals.append(dynamic_total)
        
        print("  -> DYNAMIC (Super Nodes / Mixed)")
        for i, (chunk, size) in enumerate(zip(dynamic_chunks, dynamic_sizes)):
            ctype = "Super Node" if chunk.get('type') == 'super_node' else "Raw Chunk"
            print(f"     Chunk {i+1} [{ctype} | ID: {chunk['id'][:15]}...]: {size} words")
        print(f"     Total Dynamic Words: {dynamic_total}")

    print("\n" + "=" * 50)
    print("AVERAGE SIZES (10 Queries, top_k=3)")
    print("=" * 50)
    print(f"Average Static Words (Raw Chunks):  {np.mean(static_totals):.1f}")
    print(f"Average Dynamic Words (Super Nodes): {np.mean(dynamic_totals):.1f}")

if __name__ == '__main__':
    measure_sizes()
