import os
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

# Hack to load .env token if running standalone
env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.strip().split("=")[1]

from app.config import EMBEDDING_MODEL_NAME
from app.ingestion.dataset_loader import stream_medquad
from app.ingestion.web_fetcher import get
from app.ingestion.html_cleaner import clean
from app.ingestion.cleaner import clean_text
from app.ingestion.section_parser import parse_sections
from app.ingestion.chunker import split
from app.ingestion.embedder import encode
from app.db.chroma_client import ChromaClient
from app.db.sqlite_client import init_db, save_benchmark, get_connection

def run_ingestion():
    print(f"=== Starting SERA Master Ingestor ===")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    
    print("1. Initializing SQLite Database...")
    init_db()
    
    # Pre-load already processed Q&A to support resume capability
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, document_id FROM benchmark_qa")
    existing_qas = cursor.fetchall()
    conn.close()
    
    processed_qa_ids = {row[0] for row in existing_qas}
    embedded_doc_ids = {row[1] for row in existing_qas}
    
    print(f"   -> Found {len(processed_qa_ids)} already processed Q&A pairs.")
    print(f"   -> Found {len(embedded_doc_ids)} already embedded documents.")
    
    print("\n2. Processing MedQuAD Stream...")
    total_chunks_embedded = 0
    total_qas_saved = 0
    
    for row in stream_medquad():
        # Step A: Only download and embed the HTML Document ONCE per document_id
        if row.document_id not in embedded_doc_ids:
            try:
                print(f"\n[NEW DOC] Fetching: {row.question_focus} (ID: {row.document_id})")
                print(f"          URL: {row.document_url}")
                raw_html = get(row.document_url)
                text = clean(raw_html)
                cleaned_text = clean_text(text)
                sections = parse_sections(cleaned_text)
                
                all_chunks = []
                for sec_name, sec_text in sections:
                    chunks = split(
                        text=sec_text,
                        document_id=row.document_id,
                        source_url=row.document_url,
                        question_focus=row.question_focus,
                        umls_semantic_group="Disorder", 
                        section=sec_name
                    )
                    all_chunks.extend(chunks)
                    
                if all_chunks:
                    print(f"   -> Encoding {len(all_chunks)} chunks using {EMBEDDING_MODEL_NAME}...")
                    encoded_chunks = encode(all_chunks)
                    
                    print(f"   -> Inserting {len(encoded_chunks)} vectors into ChromaDB...")
                    ChromaClient.insert_chunks(encoded_chunks)
                    
                    total_chunks_embedded += len(all_chunks)
                    
                # Mark document as embedded so we don't fetch it again for the next Q&A pair on the same page
                embedded_doc_ids.add(row.document_id)
                
            except Exception as e:
                print(f"   [ERROR] Failed to process document {row.document_url}: {e}")
                continue
                
        # Step B: Always save the Q&A pair to SQLite (if not already saved)
        if row.question_id not in processed_qa_ids:
            save_benchmark(
                qa_id=row.question_id,
                document_id=row.document_id,
                question=row.question,
                answer=row.answer,
                source_url=row.document_url
            )
            processed_qa_ids.add(row.question_id)
            total_qas_saved += 1
            print(f"   -> Saved Q&A: {row.question[:50]}...")
            
    print("\n" + "="*50)
    print(" INGESTION COMPLETE ")
    print("="*50)
    print(f"Total New Documents Embedded: {total_chunks_embedded} chunks")
    print(f"Total New Q&A Pairs Saved:  {total_qas_saved}")
    print(f"Total Q&A in Database:      {len(processed_qa_ids)}")

if __name__ == "__main__":
    run_ingestion()
