import os
import sys
import time
import shutil
from urllib.parse import urlparse
from collections import defaultdict

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
from app.ingestion.custom_extractor import extract_structured_content
from app.ingestion.chunker import split
from app.ingestion.embedder import encode
from app.db.chroma_client import ChromaClient
from app.db.sqlite_client import init_db, save_benchmark, get_connection

class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def process_document(url, domain):
    use_wayback = False
    wb_timestamp = None
    use_playwright = False
    
    if 'ghr.nlm.nih.gov' in domain or 'cancer.gov' in domain or 'nhlbi.nih.gov' in domain:
        use_wayback = False
    elif 'nlm.nih.gov' in domain and 'medlineplus' in urlparse(url).path and 'natural' in urlparse(url).path:
        use_wayback = True
    elif 'niddk.nih.gov' in domain or 'ninds.nih.gov' in domain:
        use_wayback = True
    elif 'rarediseases.info.nih.gov' in domain:
        use_playwright = True
    elif 'cdc.gov' in domain:
        use_wayback = True
        wb_timestamp = '20140101'
        
    raw_html, source = get(url, use_wayback=use_wayback, wb_timestamp=wb_timestamp, use_playwright=use_playwright)
    sections = extract_structured_content(raw_html, domain, url)
    return sections, source

def map_nihseniorhealth_question(q_type):
    q = q_type.lower()
    if q == 'information':
        return '[DEFINITIONS]'
    elif q in ['causes', 'susceptibility', 'prevention']:
        return '[RISK FACTORS AND PREVENTIONS]'
    elif q in ['symptoms', 'exams and tests', 'treatment', 'complications', 'outlook']:
        return '[SYMPTOMS AND DIAGNOSIS AND TREATMENTS]'
    elif q == 'faq':
        return '[FAQ]'
    else:
        return None

def run_ingestion():
    # Setup dual logger
    log_path = os.path.join(project_root, "data", "ingestion_output.log")
    sys.stdout = DualLogger(log_path)
    
    print(f"=== Starting SERA Master Ingestor ===")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    
    chroma_path = os.path.join(project_root, 'data', 'chroma')
    sqlite_path = os.path.join(project_root, 'data', 'sqlite', 'rag.db')
    stats_path = os.path.join(project_root, 'data', 'ingestion_stats.json')
    
    print("1. Initializing SQLite Database...")
    init_db()
    
    # Pre-load already processed Q&A to support resume capability
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, source_url FROM benchmark_qa")
    existing_qas = cursor.fetchall()
    conn.close()
    
    processed_qa_ids = {row[0] for row in existing_qas}
    embedded_doc_urls = {row[1] for row in existing_qas}
    failed_doc_urls = set()
    
    print(f"   -> Resuming with {len(embedded_doc_urls)} URLs already embedded.")
    
    print("\n2. Processing MedQuAD Stream...")
    total_chunks_embedded = 0
    total_qas_saved = 0
    
    seen_urls = set()
    
    # Stats tracking
    stats_path = os.path.join(project_root, 'data', 'ingestion_stats.json')
    stats_dict = {}
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats_dict = json.load(f)
        except Exception:
            pass
            
    stats = defaultdict(lambda: {'live': 0, 'wayback': 0, 'skipped': 0, 'skipped_urls': [], 'topics': 0, 'words': 0, 'samples': 0, 'total_dataset_links': 0})
    for k, v in stats_dict.items():
        stats[k] = v
        
    for row in stream_medquad():
        domain = urlparse(row.document_url).netloc
        
        if row.document_url not in seen_urls:
            stats[domain]['total_dataset_links'] += 1
            seen_urls.add(row.document_url)
        
        if 'nihseniorhealth.gov' in domain:
            try:
                # Use answer as content since URL doesn't exist
                question_type = map_nihseniorhealth_question(row.question_type or 'information')
                content = f"{question_type}\n{row.answer}"
                
                chunks = split(
                    text=content,
                    document_id=row.document_id,
                    source_url=row.document_url,
                    question_focus=row.question_focus,
                    umls_semantic_group="Disorder",
                    section=question_type
                )
                
                if chunks:
                    encoded_chunks = encode(chunks)
                    ChromaClient.insert_chunks(encoded_chunks)
                    total_chunks_embedded += len(chunks)
                    
                    stats[domain]['live'] += 1
                    stats[domain]['samples'] += 1
                    stats[domain]['topics'] += 1
                    stats[domain]['words'] += len(content.split())
            except Exception as e:
                print(f"   [ERROR] Failed to process nihseniorhealth QA {row.question_id}: {e}")
        else:
            if row.document_url not in embedded_doc_urls and row.document_url not in failed_doc_urls:
                try:
                    print(f"\n[NEW DOC] Fetching: {row.question_focus} (ID: {row.document_id})")
                    print(f"          URL: {row.document_url}")
                    sections, source = process_document(row.document_url, domain)
                    
                    all_chunks = []
                    for sec_name, sec_text in sections.items():
                        chunks = split(
                            text=sec_text,
                            document_id=row.document_id,
                            source_url=row.document_url,
                            question_focus=row.question_focus,
                            umls_semantic_group="Disorder", 
                            section=sec_name
                        )
                        all_chunks.extend(chunks)
                        
                    # Rescue Mode: If the page returned 0 chunks (dead/blank), loop through years 2013-2026
                    if not all_chunks:
                        print(f"   [WARN] Page appears blank/dead. Entering Rescue Mode (2013-2026)...")
                        for year in range(2026, 2012, -1):
                            wb_timestamp = f"{year}0101"
                            try:
                                raw_html, source = get(row.document_url, use_wayback=True, wb_timestamp=wb_timestamp, use_playwright=False)
                                sections = extract_structured_content(raw_html, domain, row.document_url)
                                
                                for sec_name, sec_text in sections.items():
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
                                    print(f"   [SUCCESS] Rescued using {year} Wayback snapshot!")
                                    break
                            except Exception:
                                pass
                                
                    if all_chunks:
                        print(f"   -> Encoding {len(all_chunks)} chunks using {EMBEDDING_MODEL_NAME}...")
                        encoded_chunks = encode(all_chunks)
                        
                        print(f"   -> Inserting {len(encoded_chunks)} vectors into ChromaDB...")
                        ChromaClient.insert_chunks(encoded_chunks)
                        total_chunks_embedded += len(all_chunks)
                        
                        if source == 'Live':
                            stats[domain]['live'] += 1
                        elif 'Playwright' in source:
                            stats[domain]['live'] += 1  # Playwright counts as live
                        else:
                            stats[domain]['wayback'] += 1
                            
                        stats[domain]['samples'] += 1
                        stats[domain]['topics'] += len(sections)
                        stats[domain]['words'] += sum(len(text.split()) for text in sections.values())
                    else:
                        print(f"   [ERROR] Failed to rescue document {row.document_url}")
                        failed_doc_urls.add(row.document_url)
                        stats[domain]['skipped'] += 1
                        if row.document_url not in stats[domain]['skipped_urls'] and len(stats[domain]['skipped_urls']) < 5:
                            stats[domain]['skipped_urls'].append(row.document_url)
                        
                    embedded_doc_urls.add(row.document_url)
                except Exception as e:
                    print(f"   [ERROR] Failed to process document {row.document_url}: {e}")
                    failed_doc_urls.add(row.document_url)
                    stats[domain]['skipped'] += 1
                    if row.document_url not in stats[domain]['skipped_urls'] and len(stats[domain]['skipped_urls']) < 5:
                        stats[domain]['skipped_urls'].append(row.document_url)
                    continue
                
        # Always save the Q&A pair to SQLite (if not already saved)
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
            
            try:
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(dict(stats), f, indent=4)
            except Exception:
                pass
                
    # Print Final Report
    print("\n" + "="*50)
    print(" INGESTION COMPLETE - FINAL REPORT ")
    print("="*50)
    print(f"Total New Documents Embedded: {total_chunks_embedded} chunks")
    print(f"Total New Q&A Pairs Saved:  {total_qas_saved}")
    print(f"Total Q&A in Database:      {len(processed_qa_ids)}")
    
    print("\n=== DOMAIN BREAKDOWN ===")
    overall_live = 0
    overall_wayback = 0
    overall_skipped = 0
    overall_topics = 0
    overall_words = 0
    overall_samples = 0
    overall_total_dataset_links = 0
    
    for domain, d_stats in stats.items():
        print(f"\nDOMAIN: {domain}")
        print(f"  - Total Links in Dataset:   {d_stats.get('total_dataset_links', 0)}")
        print(f"  - Accessed directly (Live): {d_stats['live']}")
        print(f"  - Accessed via Wayback:     {d_stats['wayback']}")
        print(f"  - Links Skipped:            {d_stats['skipped']}")
        
        avg_topics = d_stats['topics'] / d_stats['samples'] if d_stats['samples'] > 0 else 0
        avg_words = d_stats['words'] / d_stats['samples'] if d_stats['samples'] > 0 else 0
        print(f"  - Avg Topics per sample:    {avg_topics:.1f}")
        print(f"  - Avg Words per sample:     {avg_words:.1f}")
        
        if d_stats['skipped_urls']:
            print("  - Sample of skipped links:")
            for s_url in d_stats['skipped_urls'][:3]:
                print(f"      * {s_url}")
                
        overall_live += d_stats['live']
        overall_wayback += d_stats['wayback']
        overall_skipped += d_stats['skipped']
        overall_topics += d_stats['topics']
        overall_words += d_stats['words']
        overall_samples += d_stats['samples']
        overall_total_dataset_links += d_stats.get('total_dataset_links', 0)
        
    print("\n=== OVERALL DATASET STATS ===")
    print(f"  - Total Links in Dataset:   {overall_total_dataset_links}")
    print(f"  - Total Accessed directly:  {overall_live}")
    print(f"  - Total Accessed via WB:    {overall_wayback}")
    print(f"  - Total Links Skipped:      {overall_skipped}")
    
    overall_avg_topics = round(overall_topics / overall_samples, 1) if overall_samples > 0 else 0
    overall_avg_words = round(overall_words / overall_samples, 1) if overall_samples > 0 else 0
    print(f"  - Overall Avg Topics/sample: {overall_avg_topics}")
    print(f"  - Overall Avg Words/sample:  {overall_avg_words}")

if __name__ == "__main__":
    run_ingestion()
