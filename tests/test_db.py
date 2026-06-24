import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Hack to load .env token for standalone testing
env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.strip().split("=")[1]

from app.ingestion.dataset_loader import stream_medquad
from app.ingestion.web_fetcher import get
from app.ingestion.html_cleaner import clean
from app.ingestion.cleaner import clean_text
from app.ingestion.section_parser import parse_sections
from app.ingestion.chunker import split
from app.ingestion.embedder import encode
from app.db.chroma_client import ChromaClient
from app.db.sqlite_client import init_db, save_benchmark, get_connection

def run_test():
    print("1. Initializing SQLite...")
    init_db()
    
    print("2. Fetching 5 articles from MedQuAD...")
    docs_processed = 0
    for row in stream_medquad():
        if docs_processed >= 5:
            break
            
        print(f"Processing: {row.question_focus} (Doc ID: {row.document_id})")
        
        # Run the full pipeline
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
                umls_semantic_group="Disorder", # Mocking semantic group
                section=sec_name
            )
            all_chunks.extend(chunks)
            
        print(f"   -> Embedding {len(all_chunks)} chunks...")
        encoded_chunks = encode(all_chunks)
        
        print("   -> Inserting into ChromaDB...")
        ChromaClient.insert_chunks(encoded_chunks)
        
        print("   -> Saving Q&A to SQLite benchmark table...")
        save_benchmark(
            qa_id=row.question_id,
            document_id=row.document_id,
            question=row.question,
            answer=row.answer,
            source_url=row.document_url
        )
        docs_processed += 1
        
    print("\n" + "="*60)
    print(" VERIFICATION: QUERYING SQLITE (5 samples) ")
    print("="*60)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT document_id, question, answer FROM benchmark_qa LIMIT 5")
    qa_rows = cursor.fetchall()
    
    for i, qa_row in enumerate(qa_rows, 1):
        print(f"\n[SQLite Sample {i}] Doc ID: {qa_row[0]}")
        print(f"Q: {qa_row[1]}")
        safe_ans = qa_row[2].replace('\n', ' ')
        safe_ans = safe_ans.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
        print(f"A snippet: {safe_ans}")
        
    print("\n" + "="*60)
    print(" VERIFICATION: QUERYING CHROMADB (5 chunks) ")
    print("="*60)
    collection = ChromaClient.get_collection()
    
    results = collection.get(limit=5)
    print(f"Total chunks in database: {collection.count()}")
    
    if results['ids']:
        for i in range(len(results['ids'])):
            print(f"\n[ChromaDB Sample {i+1}] Chunk ID: {results['ids'][i]}")
            
            metadata = results['metadatas'][i]
            print(f"Metadata: Focus='{metadata['question_focus']}', Section='{metadata['section']}'")
            
            safe_doc = results['documents'][i].replace('\n', ' ')
            safe_doc = safe_doc.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
            print(f"Stored Vector Text: {safe_doc}")

if __name__ == "__main__":
    run_test()
