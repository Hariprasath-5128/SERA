import os
import sys
import time
import shutil
import chromadb
from sentence_transformers import SentenceTransformer
from tabulate import tabulate
from huggingface_hub import constants

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
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

MODELS = {
    "MiniLM": "all-MiniLM-L6-v2",
    "BGE Base": "BAAI/bge-base-en-v1.5",
    "Nomic": "nomic-ai/nomic-embed-text-v1.5",
    "PubMedBERT": "NeuML/pubmedbert-base-embeddings",
    "MedCPT (Asymmetric)": "ncbi/MedCPT-Article-Encoder",
    "BGE-M3 (Dense)": "BAAI/bge-m3",
    "GTE-Large-v1.5": "Alibaba-NLP/gte-large-en-v1.5",
    "ColBERTv2": "colbert-ir/colbertv2.0"
}

def map_question_to_section(question: str) -> str:
    q = question.lower()
    if "cause" in q or "genetic changes" in q:
        return "Causes"
    if "inherit" in q:
        return "Inheritance"
    if "common" in q or "how many people" in q:
        return "Frequency"
    if "treatment" in q or "manage" in q:
        return "Treatment"
    if "symptom" in q or "signs" in q:
        return "Symptoms"
    return "Description"

def clear_hf_cache(model_id: str):
    """Deletes a specific model from HuggingFace cache to prevent disk full errors."""
    folder_name = f"models--{model_id.replace('/', '--')}"
    cache_path = os.path.join(constants.HF_HUB_CACHE, folder_name)
    if os.path.exists(cache_path):
        print(f"   [Cleanup] Freeing disk space: Deleting {folder_name}...")
        shutil.rmtree(cache_path, ignore_errors=True)

def run_benchmark(num_diseases=10):
    print(f"=== Starting Ultimate Embedding Model Benchmark ({num_diseases} diseases) ===")
    
    documents = {}
    questions = []
    
    print("1. Fetching dataset and chunking...")
    docs_processed = 0
    current_doc_id = None
    
    for row in stream_medquad():
        if row.document_id != current_doc_id:
            if docs_processed >= num_diseases:
                break
            current_doc_id = row.document_id
            docs_processed += 1
            
            try:
                raw_html = get(row.document_url)
                text = clean(raw_html)
                cleaned_text = clean_text(text)
                sections = parse_sections(cleaned_text)
                
                all_chunks = []
                for sec_name, sec_text in sections:
                    chunks = split(sec_text, row.document_id, row.document_url, row.question_focus, "Disorder", sec_name)
                    all_chunks.extend(chunks)
                
                documents[row.document_id] = all_chunks
                print(f"   Processed: {row.question_focus} ({len(all_chunks)} chunks)")
            except Exception as e:
                print(f"   Failed to process {row.document_url}: {e}")
                
        if row.document_id in documents:
            expected_sec = map_question_to_section(row.question)
            questions.append({
                "query": row.question,
                "doc_id": row.document_id,
                "expected_section": expected_sec
            })

    all_chunks_flat = []
    for chunks in documents.values():
        all_chunks_flat.extend(chunks)
    
    print(f"\nCollected {len(questions)} questions. Total chunks to embed: {len(all_chunks_flat)}")
    
    chroma_client = chromadb.Client()
    results_table = []
    
    for model_name, model_id in MODELS.items():
        print(f"\n--- Testing Model: {model_name} ---")
        try:
            correct_at_1 = 0
            correct_at_3 = 0
            correct_at_5 = 0
            mrr_sum = 0.0
            
            if model_name == "ColBERTv2":
                from ragatouille import RAGPretrainedModel
                print(f"Loading {model_id} via RAGatouille...")
                rag = RAGPretrainedModel.from_pretrained(model_id)
                
                texts = [c.text for c in all_chunks_flat]
                ids = [c.chunk_id for c in all_chunks_flat]
                
                print("Indexing ColBERT matrices...")
                t0 = time.time()
                index_name = "colbert_benchmark"
                # RAGatouille auto-stores to disk
                rag.index(collection=texts, document_ids=ids, index_name=index_name)
                print(f"Indexed {len(texts)} chunks in {time.time() - t0:.2f}s")
                
                print("Running benchmark queries...")
                t0 = time.time()
                for i, q in enumerate(questions):
                    expected_prefix = f"{q['doc_id']}_{q['expected_section']}"
                    res = rag.search(q["query"], k=5)
                    
                    rank = 0
                    for r_idx, doc in enumerate(res):
                        if doc["document_id"].startswith(expected_prefix):
                            rank = r_idx + 1
                            break
                    
                    if rank == 1: correct_at_1 += 1
                    if rank > 0 and rank <= 3: correct_at_3 += 1
                    if rank > 0 and rank <= 5: correct_at_5 += 1
                    if rank > 0: mrr_sum += (1.0 / rank)
                    
                query_latency = (time.time() - t0) * 1000 / len(questions)
                del rag
                
            else:
                print(f"Loading {model_id} via SentenceTransformers...")
                doc_embedder = SentenceTransformer(model_id, trust_remote_code=True)
                if model_name == "MedCPT (Asymmetric)":
                    query_embedder = SentenceTransformer("ncbi/MedCPT-Query-Encoder", trust_remote_code=True)
                else:
                    query_embedder = doc_embedder
                    
                collection_name = model_name.lower().replace(" ", "_").replace("/", "_").replace("-", "_").replace("(", "").replace(")", "")
                collection = chroma_client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
                
                texts = [c.text for c in all_chunks_flat]
                ids = [c.chunk_id for c in all_chunks_flat]
                metadatas = [{"section": c.section, "doc_id": c.document_id} for c in all_chunks_flat]
                
                print("Embedding dense vectors...")
                t0 = time.time()
                chunk_embeddings = doc_embedder.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
                print(f"Embedded {len(texts)} chunks in {time.time() - t0:.2f}s")
                
                collection.add(
                    ids=ids,
                    embeddings=chunk_embeddings.tolist(),
                    documents=texts,
                    metadatas=metadatas
                )
                
                print("Running benchmark queries...")
                t0 = time.time()
                query_texts = [q["query"] for q in questions]
                query_embeddings = query_embedder.encode(query_texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
                
                for i, q in enumerate(questions):
                    expected_prefix = f"{q['doc_id']}_{q['expected_section']}"
                    
                    res = collection.query(
                        query_embeddings=[query_embeddings[i].tolist()],
                        n_results=5
                    )
                    
                    retrieved_ids = res["ids"][0]
                    
                    rank = 0
                    for r_idx, r_id in enumerate(retrieved_ids):
                        if r_id.startswith(expected_prefix):
                            rank = r_idx + 1
                            break
                    
                    if rank == 1: correct_at_1 += 1
                    if rank > 0 and rank <= 3: correct_at_3 += 1
                    if rank > 0 and rank <= 5: correct_at_5 += 1
                    if rank > 0: mrr_sum += (1.0 / rank)
                        
                query_latency = (time.time() - t0) * 1000 / len(questions)
                
                del doc_embedder
                if model_name == "MedCPT (Asymmetric)":
                    del query_embedder
                chroma_client.delete_collection(collection_name)
                
            total_q = len(questions)
            r1 = correct_at_1 / total_q
            r3 = correct_at_3 / total_q
            r5 = correct_at_5 / total_q
            mrr = mrr_sum / total_q
            
            results_table.append([model_name, f"{r1:.2f}", f"{r3:.2f}", f"{r5:.2f}", f"{mrr:.2f}", f"{query_latency:.1f}ms"])
            print(f"Finished {model_name} (R@5: {r5:.2f}, MRR: {mrr:.2f})")
            
            # CLEAR DISK CACHE SO OS DOESN'T CRASH!
            clear_hf_cache(model_id)
            if model_name == "MedCPT (Asymmetric)":
                clear_hf_cache("ncbi/MedCPT-Query-Encoder")
                
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            results_table.append([model_name, "ERROR", "ERROR", "ERROR", "ERROR", "ERROR"])
            
    print("\n" + "="*80)
    print("                 ULTIMATE EMBEDDING MODEL BENCHMARK")
    print("="*80)
    print(tabulate(results_table, headers=["Model", "Recall@1", "Recall@3", "Recall@5", "MRR", "Avg Latency"], tablefmt="grid"))

if __name__ == "__main__":
    run_benchmark(num_diseases=100)
