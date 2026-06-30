import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from app.db.chroma_client import ChromaClient
from app.db.sqlite_client import get_connection
from app.config import CHROMA_DIR, SQLITE_DIR

def run_stats():
    print("=== ChromaDB Stats ===")
    collection = ChromaClient.get_collection()
    count = collection.count()
    print(f"Total Chunks (Embeddings): {count}")
    
    if count > 0:
        results = collection.get(include=["metadatas"])
        urls = set([m.get("source_url") for m in results["metadatas"]])
        print(f"Unique URLs indexed: {len(urls)}")
        
        sections = {}
        for m in results["metadatas"]:
            sec = m.get("section", "Unknown")
            sections[sec] = sections.get(sec, 0) + 1
            
        print("Chunks by Section:")
        for k, v in sections.items():
            print(f"  - {k}: {v} chunks")
            
    chroma_sqlite = os.path.join(CHROMA_DIR, "chroma.sqlite3")
    if os.path.exists(chroma_sqlite):
        size_kb = os.path.getsize(chroma_sqlite) / 1024
        print(f"Chroma DB File Size: {size_kb:.2f} KB")

    print("\n=== Benchmark SQLite Stats ===")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM benchmark_qa")
    qa_count = cursor.fetchone()[0]
    print(f"Total Q&A Pairs: {qa_count}")
    
    cursor.execute("SELECT COUNT(DISTINCT source_url) FROM benchmark_qa")
    unique_docs = cursor.fetchone()[0]
    print(f"Unique Source URLs: {unique_docs}")
    
    rag_db = os.path.join(SQLITE_DIR, "rag.db")
    if os.path.exists(rag_db):
        size_kb = os.path.getsize(rag_db) / 1024
        print(f"Benchmark DB File Size: {size_kb:.2f} KB")

if __name__ == "__main__":
    run_stats()
