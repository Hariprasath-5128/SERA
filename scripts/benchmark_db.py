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
                {"role": "system", "content": "You are a medical expert. Provide a comprehensive, highly detailed medical answer to the following question. Include all relevant symptoms, treatments, side effects, and physiological mechanisms. Use bullet points and bold headers to structure your response. Do not include conversational filler, just the medical facts."},
                {"role": "user", "content": query}
            ],
            temperature=0.0,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate ground truth: {e}")
        return query

import re

STOPWORDS = set([
    "the", "and", "to", "of", "a", "in", "is", "that", "it", "with", "as", "for", "are", "this", 
    "on", "by", "be", "an", "or", "from", "can", "what", "which", "how", "has", "have", "not", 
    "but", "these", "their", "they", "will", "would", "about", "also", "if", "more", "such", 
    "when", "some", "other", "than", "may", "there", "into", "all", "we", "then", "up", "out"
])

def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())
    return set([w for w in words if w not in STOPWORDS])

def evaluate_pipeline(num_questions=10, query_type="complex", limit_mode="none", fixed_limit=500, static_k=5, dynamic_k=2):
    logger.info(f"--- Starting Static vs Dynamic Evaluation ({query_type.upper()} | Static k={static_k}, Dynamic k={dynamic_k}) ---")
    
    db_path = os.path.join(project_root, "data", "sqlite", "rag.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    table_name = "benchmark_qa_complex" if query_type == "complex" else "benchmark_qa_simple"
    c.execute(f"SELECT question FROM {table_name} LIMIT ?", (num_questions,))
    questions = [row[0] for row in c.fetchall()]
    conn.close()

    if not questions:
        logger.error("No questions found in benchmark_qa table!")
        return

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
        
        ground_truth = generate_ground_truth(question)
        refs.append(ground_truth)
        gt_keywords = extract_keywords(ground_truth)
        
        logger.info("  -> Querying STATIC Database...")
        config.CHROMA_PERSIST_PATH = os.path.join(project_root, "data", "backups", "v1", "chroma")
        config.SQLITE_DB_PATH = os.path.join(project_root, "data", "backups", "v1", "rag.db")
        ChromaClient._client = None
        ChromaClient._collection = None
        ChromaClient._super_nodes_collection = None
        super_node_store._collection = None
        
        static_chunks = Retriever.search(question, top_k=static_k)
        
        # Enforce static limit if fixed mode
        static_limit = fixed_limit if limit_mode == "fixed" else None
        static_ans = Generator.generate_answer(question, static_chunks, max_words=static_limit)
        static_preds.append(static_ans)
        
        # Calculate exactly how many words Static actually used
        static_words_used = 0
        static_context_text = ""
        for chunk in static_chunks:
            words = chunk['text'].split()
            if static_limit and static_words_used + len(words) > static_limit:
                allowed = static_limit - static_words_used
                if allowed > 0:
                    static_context_text += " ".join(words[:allowed]) + "\n\n"
                    static_words_used += allowed
                break
            static_context_text += chunk['text'] + "\n\n"
            static_words_used += len(words)
            
        static_context_words = max(1, len(static_context_text.split()))
        static_context_keywords = extract_keywords(static_context_text)
        static_volume = len(static_context_keywords.intersection(gt_keywords))
        
        static_ans_keywords = extract_keywords(static_ans)
        static_recall = len(static_ans_keywords.intersection(gt_keywords)) / max(1, len(gt_keywords))
        
        # Determine Dynamic Limit to perfectly match Static if in dynamic_match mode
        dynamic_limit = static_words_used if limit_mode == "dynamic_match" else fixed_limit
        if limit_mode != "none":
            logger.info(f"     [Enforcing max_words={dynamic_limit} for Dynamic Pipeline]")
        
        logger.info("  -> Querying DYNAMIC Database...")
        config.CHROMA_PERSIST_PATH = os.path.join(project_root, "data", "chroma")
        config.SQLITE_DB_PATH = os.path.join(project_root, "data", "sqlite", "rag.db")
        ChromaClient._client = None
        ChromaClient._collection = None
        ChromaClient._super_nodes_collection = None
        super_node_store._collection = None
        
        # Retrieve many, then filter ONLY Super Nodes, then take dynamic_k
        raw_dynamic = Retriever.search(question, top_k=20)
        dynamic_chunks = [c for c in raw_dynamic if c.get('type') == 'super_node'][:dynamic_k]
        
        # If no super nodes found (e.g. simple queries), fallback to raw chunks so it doesn't crash
        if not dynamic_chunks:
            dynamic_chunks = raw_dynamic[:dynamic_k]
            
        dynamic_ans = Generator.generate_answer(question, dynamic_chunks, max_words=dynamic_limit if limit_mode != "none" else None)
        dynamic_preds.append(dynamic_ans)
        
        # Calculate exactly how many words Dynamic actually used
        dynamic_words_used = 0
        dynamic_context_text = ""
        for chunk in dynamic_chunks:
            words = chunk['text'].split()
            if dynamic_limit and dynamic_words_used + len(words) > dynamic_limit:
                allowed = dynamic_limit - dynamic_words_used
                if allowed > 0:
                    dynamic_context_text += " ".join(words[:allowed]) + "\n\n"
                    dynamic_words_used += allowed
                break
            dynamic_context_text += chunk['text'] + "\n\n"
            dynamic_words_used += len(words)
            
        dynamic_context_words = max(1, len(dynamic_context_text.split()))
        dynamic_context_keywords = extract_keywords(dynamic_context_text)
        dynamic_volume = len(dynamic_context_keywords.intersection(gt_keywords))
        
        dynamic_ans_keywords = extract_keywords(dynamic_ans)
        dynamic_recall = len(dynamic_ans_keywords.intersection(gt_keywords)) / max(1, len(gt_keywords))
        
        # ROUGE Scoring against Ground Truth
        s_rouge = r_scorer.score(ground_truth, static_ans)
        d_rouge = r_scorer.score(ground_truth, dynamic_ans)

        results.append({
            "question": question,
            "static_rougeL": s_rouge['rougeL'].fmeasure,
            "dynamic_rougeL": d_rouge['rougeL'].fmeasure,
            "static_recall": static_recall,
            "dynamic_recall": dynamic_recall,
            "static_volume": static_volume,
            "dynamic_volume": dynamic_volume
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
    
    avg_s_recall = sum(r["static_recall"] for r in results) / len(results)
    avg_d_recall = sum(r["dynamic_recall"] for r in results) / len(results)
    
    avg_s_volume = sum(r["static_volume"] for r in results) / len(results)
    avg_d_volume = sum(r["dynamic_volume"] for r in results) / len(results)
    
    logger.info("=========================================")
    logger.info("          EVALUATION RESULTS             ")
    logger.info("=========================================")
    logger.info("Metric         | Static (Before) | Dynamic (After)")
    logger.info("-----------------------------------------")
    logger.info(f"ROUGE-L F1     | {avg_s_rl:.4f}          | {avg_d_rl:.4f}")
    logger.info(f"BioBERT F1     | {avg_s_f1:.4f}          | {avg_d_f1:.4f}")
    logger.info(f"Concept Recall | {avg_s_recall:.4f}          | {avg_d_recall:.4f}")
    logger.info(f"Info Volume    | {avg_s_volume:.4f}          | {avg_d_volume:.4f}")
    logger.info("=========================================")
    
    # Save results
    out_file = os.path.join(project_root, f"benchmark_comparison_{query_type}.json")
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "num_questions": len(questions),
            "static_scores": {"rougeL": avg_s_rl, "biobert": avg_s_f1, "recall": avg_s_recall, "volume": avg_s_volume},
            "dynamic_scores": {"rougeL": avg_d_rl, "biobert": avg_d_f1, "recall": avg_d_recall, "volume": avg_d_volume},
            "details": results
        }, f, indent=2)
    logger.info(f"Detailed results saved to {out_file}")
    return avg_s_rl, avg_d_rl, avg_s_f1, avg_d_f1, avg_s_recall, avg_d_recall, avg_s_volume, avg_d_volume

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Number of queries to pull from rag.db")
    parser.add_argument("--type", type=str, choices=["complex", "simple", "all"], default="all", help="Type of queries to evaluate")
    parser.add_argument("--limit_mode", type=str, choices=["none", "fixed", "dynamic_match"], default="none", help="Mode to enforce strict word limits.")
    parser.add_argument("--max_words", type=int, default=500, help="Max words to allow when limit_mode is 'fixed'")
    parser.add_argument("--static_k", type=int, default=5, help="Number of nodes to retrieve for Static pipeline")
    parser.add_argument("--dynamic_k", type=int, default=2, help="Number of nodes to retrieve for Dynamic pipeline")
    args = parser.parse_args()
    
    if args.type == "all":
        print("\n\n" + "*" * 50)
        print(f"RUNNING PART 1: SIMPLE QUERIES (Static k={args.static_k}, Dynamic k={args.dynamic_k})")
        print("*" * 50)
        s_rl_simple, d_rl_simple, s_f1_simple, d_f1_simple, s_recall_simple, d_recall_simple, s_volume_simple, d_volume_simple = evaluate_pipeline(args.limit, "simple", args.limit_mode, args.max_words, args.static_k, args.dynamic_k)
        
        print("\n\n" + "*" * 50)
        print(f"RUNNING PART 2: COMPLEX QUERIES (Static k={args.static_k}, Dynamic k={args.dynamic_k})")
        print("*" * 50)
        s_rl_complex, d_rl_complex, s_f1_complex, d_f1_complex, s_recall_complex, d_recall_complex, s_volume_complex, d_volume_complex = evaluate_pipeline(args.limit, "complex", args.limit_mode, args.max_words, args.static_k, args.dynamic_k)
        
        print("\n" + "=" * 80)
        print(f"                   FINAL BENCHMARK COMPARISON (Static k={args.static_k}, Dynamic k={args.dynamic_k})                   ")
        print("================================================================================")
        print(f"               | {'SIMPLE QUERIES (Static vs Dynamic)':^35} | {'COMPLEX QUERIES (Static vs Dynamic)':^35} |")
        print("---------------|-------------------------------------|-------------------------------------|")
        print(f" ROUGE-L F1    |    {s_rl_simple:.4f}    vs    {d_rl_simple:.4f}       |    {s_rl_complex:.4f}    vs    {d_rl_complex:.4f}       |")
        print(f" BioBERT F1    |    {s_f1_simple:.4f}    vs    {d_f1_simple:.4f}       |    {s_f1_complex:.4f}    vs    {d_f1_complex:.4f}       |")
        print("---------------|-------------------------------------|-------------------------------------|")
        print(f" Concept Recall|    {s_recall_simple:.4f}    vs    {d_recall_simple:.4f}       |    {s_recall_complex:.4f}    vs    {d_recall_complex:.4f}       |")
        print(f" Info Volume   |    {s_volume_simple:.4f}    vs    {d_volume_simple:.4f}       |    {s_volume_complex:.4f}    vs    {d_volume_complex:.4f}       |")
        print("================================================================================\n")
    else:
        evaluate_pipeline(args.limit, args.type, args.limit_mode, args.max_words, args.static_k, args.dynamic_k)
