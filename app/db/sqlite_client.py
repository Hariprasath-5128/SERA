import sqlite3
from app.config import SQLITE_DB_PATH

def get_connection():
    return sqlite3.connect(SQLITE_DB_PATH)

def init_db():
    """Initialize the SQLite database with the benchmark table."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # We store the Q&A pairs here for future Phase 6 benchmarking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_qa (
            id TEXT PRIMARY KEY,
            document_id TEXT,
            question TEXT,
            answer TEXT,
            source_url TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_benchmark(qa_id: str, document_id: str, question: str, answer: str, source_url: str):
    """Save a single Q&A pair to the benchmark database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO benchmark_qa (id, document_id, question, answer, source_url)
        VALUES (?, ?, ?, ?, ?)
    """, (qa_id, document_id, question, answer, source_url))
    conn.commit()
    conn.close()
