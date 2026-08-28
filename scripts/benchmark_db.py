import os
import sys
import argparse
import json
import logging
import sqlite3
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from openai import OpenAI

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Force environment config
from dotenv import load_dotenv
env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)

import app.config as config
from app.retrieval.retriever import Retriever
from app.generation.generator import Generator
from app.db.chroma_client import ChromaClient
from app.db import super_node_store

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_ground_truth(query):
    """Generate a baseline medical truth using raw LLM knowledge to score against."""
    from app.generation.generator import Generator
    import app.config as config
    client = Generator.get_client()
    try:
        response = client.chat.completions.create(
            model=config.GENERATOR_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a medical expert. Provide a comprehensive, highly detailed medical answer to the following question. Include all relevant symptoms, treatments, side effects, and physiological mechanisms. Do not include conversational filler, just the medical facts."},
                {"role": "user", "content": query}
            ],
            temperature=0.0,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate ground truth: {e}")
        return query

def evaluate_pipeline(num_questions=10):
    logger.info("--- Starting Static vs Dynamic Database Evaluation ---")
    
    # 1. Fetch queries from the database (benchmark_qa table)
    db_path = os.path.join(project_root, "data", "sqlite", "rag.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT question FROM benchmark_qa LIMIT ?", (num_questions,))
    questions = [row[0] for row in c.fetchall()]
    conn.close()

    if not questions:
        logger.error("No questions found in benchmark_qa table!")
        return

    logger.info(f"Loaded {len(questions)} queries from rag.db benchmark_qa table.")

    # Load BioBERT scorer and ROUGE scorer
    logger.info("Initializing BioBERT and ROUGE scorers...")
    from bert_score import BERTScorer
    bert_scorer = BERTScorer(model_type="dmis-lab/biobert-base-cased-v1.1", num_layers=11, device="cpu")
    r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    static_preds = []
    dynamic_preds = []
    refs = []
    results = []

    for idx, question in enumerate(questions):
        logger.info(f"\nEvaluating [{idx+1}/{len(questions)}]: {question}")
        
        # 1. Generate Ground Truth (Reference)
        logger.info("  -> Generating Baseline Reference (Zero-shot LLM)...")
        ground_truth = generate_ground_truth(question)
        refs.append(ground_truth)
        
        # 2. Evaluate STATIC (V1 Backup)
        logger.info("  -> Querying STATIC Database (V1 Backup)...")
        config.CHROMA_PERSIST_PATH = os.path.join(project_root, "data", "backups", "v1", "chroma")
        config.SQLITE_DB_PATH = os.path.join(project_root, "data", "backups", "v1", "rag.db")
        ChromaClient._client = None
        ChromaClient._collection = None
        ChromaClient._super_nodes_collection = None
        super_node_store._collection = None
        
        static_chunks = Retriever.search(question, top_k=3)
        static_ans = Generator.generate_answer(question, static_chunks)
        static_preds.append(static_ans)
        
        # 3. Evaluate DYNAMIC (Current Knowledge Graph)
        logger.info("  -> Querying DYNAMIC Database (Knowledge Graph)...")
        config.CHROMA_PERSIST_PATH = os.path.join(project_root, "data", "chroma")
        config.SQLITE_DB_PATH = os.path.join(project_root, "data", "sqlite", "rag.db")
        ChromaClient._client = None
        ChromaClient._collection = None
        ChromaClient._super_nodes_collection = None
        super_node_store._collection = None
        
        dynamic_chunks = Retriever.search(question, top_k=5)
        dynamic_ans = Generator.generate_answer(question, dynamic_chunks)
        dynamic_preds.append(dynamic_ans)
        
        # ROUGE Scoring against Ground Truth
        s_rouge = r_scorer.score(ground_truth, static_ans)
        d_rouge = r_scorer.score(ground_truth, dynamic_ans)

        results.append({
            "question": question,
            "static_rougeL": s_rouge['rougeL'].fmeasure,
            "dynamic_rougeL": d_rouge['rougeL'].fmeasure
        })

    # BioBERT Scoring
    logger.info("\nCalculating BioBERT Semantic Similarity Scores...")
    
    def truncate_words(text, max_words=200):
        words = text.split()
        return " ".join(words[:max_words]) if len(words) > max_words else text

    static_preds_truncated = [truncate_words(p) for p in static_preds]
    dynamic_preds_truncated = [truncate_words(p) for p in dynamic_preds]
    refs_truncated = [truncate_words(r) for r in refs]

    _, _, static_f1 = bert_scorer.score(static_preds_truncated, refs_truncated)
    _, _, dynamic_f1 = bert_scorer.score(dynamic_preds_truncated, refs_truncated)
    
    avg_s_f1 = static_f1.mean().item()
    avg_d_f1 = dynamic_f1.mean().item()
    
    avg_s_rl = sum(r["static_rougeL"] for r in results) / len(results)
    avg_d_rl = sum(r["dynamic_rougeL"] for r in results) / len(results)
    
    logger.info("=========================================")
    logger.info("          EVALUATION RESULTS             ")
    logger.info("=========================================")
    logger.info("Metric         | Static (Before) | Dynamic (After)")
    logger.info("-----------------------------------------")
    logger.info(f"ROUGE-L F1     | {avg_s_rl:.4f}          | {avg_d_rl:.4f}")
    logger.info(f"BioBERT F1     | {avg_s_f1:.4f}          | {avg_d_f1:.4f}")
    logger.info("=========================================")
    
    # Save results
    out_file = os.path.join(project_root, "benchmark_comparison.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "num_questions": len(questions),
            "static_scores": {"rougeL": avg_s_rl, "biobert": avg_s_f1},
            "dynamic_scores": {"rougeL": avg_d_rl, "biobert": avg_d_f1},
            "details": results
        }, f, indent=2)
    logger.info(f"Detailed results saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Number of queries to pull from rag.db")
    args = parser.parse_args()
    evaluate_pipeline(args.limit)
