"""
scripts/eval_static_vs_dynamic.py
SERA Static vs Dynamic RAG Evaluation Pipeline

Protocol
--------
STATIC  pipeline : data/backups/v1/chroma + data/backups/v1/rag.db
DYNAMIC pipeline : data/chroma            + data/sqlite/rag.db
DYNAMIC retrieves Super-Nodes ONLY - no raw-chunk fallback (per protocol)
Generator model  : identical for both (config.GENERATOR_LLM_MODEL)
Judge   model    : env var JUDGE_LLM_MODEL (default: qwen2.5:7b)
K values         : 1 .. 9
Tables           : benchmark_qa_simple and benchmark_qa_complex
MAX_QUERIES      : 100 per table (configurable via --max_queries)

Metrics
-------
  ROUGE-L F1           classic longest-common-subsequence recall
  BioBERT F1           cosine similarity from biomedical sentence embeddings
  Concept Recall       |answer_kws intersect ref_kws| / |ref_kws|
  Ref-Aligned Info Vol |context_kws intersect ref_kws|
  Context Tokens       whitespace-split word count of retrieved context
  Output Tokens        whitespace-split word count of generated answer
  Total Tokens         Context + Output
  Latency (s)          wall-clock time for retrieval + generation
  SAIR                 (supported correct claims) / total_claims * 100
  UAIR                 unsupported_claims / total_claims * 100

Usage
-----
  python scripts/eval_static_vs_dynamic.py --experiment all --query_type all
  python scripts/eval_static_vs_dynamic.py --experiment k_sensitivity
         --query_type simple --max_queries 50
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Bootstrap: ensure the project root is on sys.path so SERA app modules load
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# SERA app imports
# ---------------------------------------------------------------------------
from app import config as _app_config  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_static_vs_dynamic")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATIC_CHROMA_PATH = str(_PROJECT_ROOT / "data" / "backups" / "v1" / "chroma")
STATIC_SQLITE_PATH = str(_PROJECT_ROOT / "data" / "backups" / "v1" / "rag.db")
DYNAMIC_CHROMA_PATH = str(_PROJECT_ROOT / "data" / "chroma")
DYNAMIC_SQLITE_PATH = str(_PROJECT_ROOT / "data" / "sqlite" / "rag.db")

CHROMA_RAW_COLLECTION = _app_config.CHROMA_RAW_COLLECTION
CHROMA_SUPER_COLLECTION = _app_config.CHROMA_SUPER_COLLECTION
GENERATOR_MODEL = _app_config.GENERATOR_LLM_MODEL
JUDGE_MODEL = os.getenv("JUDGE_LLM_MODEL", "qwen2.5:7b")
LLM_BASE_URL = _app_config.LLM_BASE_URL

K_VALUES = list(range(1, 10))
# Equal-budget pairs: (static_k, dynamic_k)
# Derived from actual ChromaDB statistics (analyze_db_tokens.py):
#   Raw chunk  avg = 176 words  (~289 tokens by chars/4)
#   Super Node avg = 394 words  (~630 tokens by chars/4)
#   Ratio: 1 Super Node ~= 2.23 raw chunks
#
# Pairs chosen so that static_k * 176 ~= dynamic_k * 394 (token-balanced):
#   s2_d1  -> Static 2 chunks (~353w) vs Dynamic 1 SN (~394w)  diff=10.4%
#   s4_d2  -> Static 4 chunks (~706w) vs Dynamic 2 SN (~788w)  diff=10.4%
#   s9_d4  -> Static 9 chunks (~1588w) vs Dynamic 4 SN (~1576w) diff=0.8% [near-perfect]
EQUAL_BUDGET_PAIRS: List[Tuple[int, int]] = [(2, 1), (4, 2), (9, 4)]
SIMPLE_TABLE = "benchmark_qa_simple"
COMPLEX_TABLE = "benchmark_qa_complex"


# ---------------------------------------------------------------------------
# Lazy imports for heavy libraries
# ---------------------------------------------------------------------------

def _import_rouge():
    try:
        from rouge_score import rouge_scorer  # type: ignore
        return rouge_scorer
    except ImportError:
        logger.warning("rouge_score not installed; ROUGE-L will be 0.0")
        return None


def _import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not installed; BioBERT F1 will be 0.0")
        return None


def _import_chromadb():
    import chromadb  # type: ignore
    return chromadb


def _import_openai():
    from openai import OpenAI  # type: ignore
    return OpenAI


# ---------------------------------------------------------------------------
# Singleton: BioBERT embedding model
# ---------------------------------------------------------------------------
_BIOBERT_MODEL = None
# Proper sentence-transformer: dmis-lab/biobert-v1.1 has no pooling config, so
# SentenceTransformer falls back to untuned mean pooling. Measured on generic
# sentence pairs, that floors unrelated text at ~0.81 cosine (separation +0.14);
# this STS-tuned BioBERT floors them at ~0.07 (separation +0.68).
_BIOBERT_MODEL_NAME = os.getenv(
    "BIOBERT_ST_MODEL", "pritamdeka/S-BioBert-snli-multinli-stsb"
)


def _get_biobert_model():
    global _BIOBERT_MODEL
    if _BIOBERT_MODEL is None:
        ST = _import_sentence_transformers()
        if ST is not None:
            try:
                _BIOBERT_MODEL = ST(_BIOBERT_MODEL_NAME)
            except Exception as exc:
                logger.warning("BioBERT model load failed: %s", exc)
    return _BIOBERT_MODEL


# ---------------------------------------------------------------------------
# Singleton: LLM clients
# ---------------------------------------------------------------------------
_GENERATOR_CLIENT = None
_JUDGE_CLIENT = None


def _get_generator_client():
    global _GENERATOR_CLIENT
    if _GENERATOR_CLIENT is None:
        OpenAI = _import_openai()
        _GENERATOR_CLIENT = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "not-required"),
            base_url=LLM_BASE_URL,
        )
    return _GENERATOR_CLIENT


def _get_judge_client():
    global _JUDGE_CLIENT
    if _JUDGE_CLIENT is None:
        OpenAI = _import_openai()
        _JUDGE_CLIENT = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "not-required"),
            base_url=LLM_BASE_URL,
        )
    return _JUDGE_CLIENT


# ---------------------------------------------------------------------------
# Preflight: verify required models are actually served before a long run
# ---------------------------------------------------------------------------

def _preflight_models(need_judge):
    """Fail fast if a required model is missing.

    A missing model otherwise surfaces only as a per-query 404 that the judge
    path swallows, so a multi-hour run silently reports SAIR/UAIR of 0.0%.
    """
    required = [("generator", GENERATOR_MODEL)]
    if need_judge:
        required.append(("judge", JUDGE_MODEL))

    try:
        available = set(m.id for m in _get_generator_client().models.list().data)
    except Exception as exc:
        print("")
        print("[Preflight] FATAL: cannot reach LLM server at %s" % LLM_BASE_URL)
        print("[Preflight] %s: %s" % (type(exc).__name__, exc))
        print("[Preflight] Is Ollama running?  ->  ollama serve")
        sys.exit(1)

    missing = [(role, name) for role, name in required if name not in available]
    if missing:
        print("")
        print("[Preflight] FATAL: required model(s) not available on the server.")
        for role, name in missing:
            print("[Preflight]   missing %s model: %s   ->  ollama pull %s"
                  % (role, name, name))
        print("[Preflight] Server reports: %s" % (sorted(available) or "(none)"))
        sys.exit(1)

    for role, name in required:
        print("[Preflight] OK  %-9s %s" % (role, name))


# ---------------------------------------------------------------------------
# ChromaDB collection helpers (pipeline-switchable)
# ---------------------------------------------------------------------------

def _get_chroma_collection(chroma_path: str, collection_name: str):
    chromadb = _import_chromadb()
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        return client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Embedding helper (uses the SERA app Embedder)
# ---------------------------------------------------------------------------

def _embed_query(query: str) -> List[float]:
    from app.ingestion.embedder import Embedder  # type: ignore
    model = Embedder.get_model()
    emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
    return emb.tolist()


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def _retrieve_static(query: str, top_k: int, chroma_path: str) -> Tuple[List[Dict], float]:
    """Static retrieval: raw_chunks collection only. Returns (chunks, latency_seconds)."""
    t0 = time.perf_counter()
    q_emb = _embed_query(query)
    col = _get_chroma_collection(chroma_path, CHROMA_RAW_COLLECTION)
    if col is None or col.count() == 0:
        return [], time.perf_counter() - t0
    n = min(top_k, col.count())
    raw = col.query(
        query_embeddings=[q_emb],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    results: List[Dict] = []
    for id_, doc, meta, dist in zip(
        raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
    ):
        results.append({"id": id_, "text": doc, "metadata": meta,
                         "distance": dist, "type": "raw_chunk"})
    return results, time.perf_counter() - t0


def _retrieve_dynamic(query: str, top_k: int, chroma_path: str) -> Tuple[List[Dict], float]:
    """Dynamic retrieval: super_nodes ONLY - no raw-chunk fallback (per protocol)."""
    t0 = time.perf_counter()
    q_emb = _embed_query(query)
    col = _get_chroma_collection(chroma_path, CHROMA_SUPER_COLLECTION)
    if col is None or col.count() == 0:
        return [], time.perf_counter() - t0
    n = min(top_k, col.count())
    raw = col.query(
        query_embeddings=[q_emb],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    results: List[Dict] = []
    for id_, doc, meta, dist in zip(
        raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
    ):
        if meta.get("type") == "meta_node":
            continue
        results.append({"id": id_, "text": doc, "metadata": meta,
                         "distance": dist, "type": "super_node"})
    return results, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------


def _generate_reference(query: str) -> str:
    from app.generation.generator import Generator
    import app.config as config
    client = Generator.get_client()
    try:
        res = client.chat.completions.create(
            model=config.GENERATOR_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a medical expert. Provide a comprehensive, factually accurate answer including relevant mechanisms, treatments, and clinical considerations."},
                {"role": "user", "content": query}
            ],
            temperature=0.0,
            max_tokens=1000
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating ref: {e}")
        return ""


def _generate_answer(
    query: str,
    chunks: List[Dict],
    max_context_words: int = 2000,
) -> Tuple[str, float]:
    """Call the generator LLM with retrieved chunks as context."""
    if not chunks:
        return "No context retrieved.", 0.0

    context_words = 0
    context_parts: List[str] = []
    for chunk in chunks:
        words = chunk["text"].split()
        if context_words + len(words) > max_context_words:
            allowed = max_context_words - context_words
            if allowed > 0:
                context_parts.append(" ".join(words[:allowed]))
            break
        context_parts.append(chunk["text"])
        context_words += len(words)

    context_text = "\n\n".join(context_parts)
    system_prompt = (
        "You are a precise medical expert. Answer the question using ONLY the facts "
        "provided in the background context. Extract and state all relevant medical "
        "facts comprehensively. Include all relevant symptoms, side effects, mechanisms, "
        "and treatments mentioned. Do not add introductions, conclusions, or phrases "
        "like 'Based on the context'. Do not repeat the question."
    )
    user_prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        f"Answer (medical facts only):"
    )
    t0 = time.perf_counter()
    try:
        client = _get_generator_client()
        response = client.chat.completions.create(
            model=GENERATOR_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("Generation error: %s", exc)
        answer = f"[GENERATION ERROR: {exc}]"
    return answer, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    if not text: return []
    return text.lower().split()


def _keywords(text: str) -> set:
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "of", "in", "on", "at", "to", "for", "with", "by", "from",
        "and", "or", "but", "not", "no", "it", "its", "this", "that",
        "these", "those", "as", "if", "then", "than", "so", "also",
        "which", "who", "whom", "what", "when", "where", "how",
    }
    return {w for w in _tokenize(text) if w not in STOPWORDS and len(w) > 2}


def _rouge_l_f1(hypothesis: str, reference: str) -> float:
    rouge_scorer = _import_rouge()
    if rouge_scorer is None:
        return 0.0
    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = scorer.score(reference, hypothesis)
        return round(scores["rougeL"].fmeasure, 4)
    except Exception:
        return 0.0


def _biobert_f1(hypothesis: str, reference: str) -> float:
    model = _get_biobert_model()
    if model is None:
        return 0.0
    try:
        import numpy as np
        embs = model.encode(
            [hypothesis, reference], convert_to_numpy=True, show_progress_bar=False
        )
        h_emb, r_emb = embs[0], embs[1]
        h_norm = h_emb / (np.linalg.norm(h_emb) + 1e-9)
        r_norm = r_emb / (np.linalg.norm(r_emb) + 1e-9)
        sim = float(np.dot(h_norm, r_norm))
        return round(max(0.0, min(1.0, sim)), 4)
    except Exception:
        return 0.0


def _concept_recall(answer: str, reference: str) -> float:
    ans_kws = _keywords(answer)
    ref_kws = _keywords(reference)
    if not ref_kws:
        return 0.0
    return round(len(ans_kws & ref_kws) / len(ref_kws), 4)


def _ref_aligned_info_volume(context: str, reference: str) -> int:
    ctx_kws = _keywords(context)
    ref_kws = _keywords(reference)
    return len(ctx_kws & ref_kws)


def _token_count(text: str) -> int:
    return len(text.split())


def _context_text(chunks: List[Dict]) -> str:
    return "\n\n".join(c["text"] for c in chunks)


# ---------------------------------------------------------------------------
# SAIR / UAIR Judge (LLM-as-judge)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are a rigorous medical claim analyst.\n"
    "Given a question, a reference answer, and a generated answer, decompose the\n"
    "generated answer into atomic claims and classify each claim:\n"
    "  - correct_supported : factually correct AND traceable to the reference answer\n"
    "  - incorrect         : factually wrong or contradicts the reference\n"
    "  - unsupported       : may be true but is NOT mentioned in the reference answer\n"
    "\n"
    "Reply ONLY with valid JSON in this exact format (no markdown fences):\n"
    "{\n"
    '  "total_claims": <int>,\n'
    '  "correct_supported": <int>,\n'
    '  "incorrect": <int>,\n'
    '  "unsupported": <int>\n'
    "}"
)


def _judge_sair_uair(question: str, reference: str, generated: str) -> Dict[str, Any]:
    """Use judge LLM to compute SAIR and UAIR. Returns a metric dict."""
    default: Dict[str, Any] = {
        "total_claims": 0, "correct_supported": 0,
        "unsupported": 0, "sair": 0.0, "uair": 0.0,
    }
    if "[GENERATION ERROR" in generated:
        return default
    user_prompt = (
        f"Question: {question}\n\n"
        f"Reference Answer:\n{reference}\n\n"
        f"Generated Answer:\n{generated}\n\n"
        "Analyse the generated answer as described."
    )
    try:
        client = _get_judge_client()
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip("`").strip()
        data = json.loads(raw)
        total = max(int(data.get("total_claims", 0)), 1)
        cs = int(data.get("correct_supported", 0))
        us = int(data.get("unsupported", 0))
        sair = round((cs / total) * 100, 2)
        uair = round((us / total) * 100, 2)
        return {
            "total_claims": total, "correct_supported": cs,
            "unsupported": us, "sair": sair, "uair": uair,
        }
    except Exception as exc:
        logger.warning("Judge call failed: %s", exc)
        return default


# ---------------------------------------------------------------------------
# SQLite benchmark queries loader
# ---------------------------------------------------------------------------

def _load_benchmark_queries(
    sqlite_path: str,
    table: str,
    max_queries: int,
) -> List[Dict]:
    """Load up to max_queries rows from a benchmark table in the SQLite DB."""
    if not os.path.exists(sqlite_path):
        logger.error("SQLite DB not found: %s", sqlite_path)
        return []
    try:
        conn = sqlite3.connect(sqlite_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            f"SELECT id, question, answer, source_url FROM {table} LIMIT ?",
            (max_queries,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as exc:
        logger.error("Failed to load table %s from %s: %s", table, sqlite_path, exc)
        return []


# ---------------------------------------------------------------------------
# Core evaluation runner
# ---------------------------------------------------------------------------

def _run_single_eval(
    query_row: Dict,
    pipeline: str,
    top_k: int,
    chroma_path: str,
    run_judge: bool = False,
) -> Dict[str, Any]:
    """Run retrieval + generation + metrics for one query row. Returns flat metric dict."""
    question = query_row["question"]
    reference = query_row.get("answer")
    if not reference:
        reference = _generate_reference(question)
        query_row["answer"] = reference


    if pipeline == "static":
        chunks, retrieval_latency = _retrieve_static(question, top_k, chroma_path)
    else:
        chunks, retrieval_latency = _retrieve_dynamic(question, top_k, chroma_path)

    answer, gen_latency = _generate_answer(question, chunks)
    total_latency = retrieval_latency + gen_latency
    ctx_text = _context_text(chunks)

    rouge = _rouge_l_f1(answer, reference)
    biobert = _biobert_f1(answer, reference)
    concept_rec = _concept_recall(answer, reference)
    raiv = _ref_aligned_info_volume(ctx_text, reference)
    ctx_tokens = _token_count(ctx_text)
    out_tokens = _token_count(answer)
    total_tokens = ctx_tokens + out_tokens

    if run_judge:
        judge = _judge_sair_uair(question, reference, answer)
    else:
        judge = {"total_claims": 0, "correct_supported": 0,
                 "unsupported": 0, "sair": 0.0, "uair": 0.0}

    return {
        "id": query_row.get("id", ""),
        "question": question,
        "reference": reference,
        "pipeline": pipeline,
        "top_k": top_k,
        "num_chunks_retrieved": len(chunks),
        "answer": answer,
        "rouge_l": rouge,
        "biobert_f1": biobert,
        "concept_recall": concept_rec,
        "raiv": raiv,
        "context_tokens": ctx_tokens,
        "output_tokens": out_tokens,
        "total_tokens": total_tokens,
        "latency": round(total_latency, 3),
        "sair": judge["sair"],
        "uair": judge["uair"],
        "total_claims": judge["total_claims"],
    }


# ---------------------------------------------------------------------------
# Aggregate helper
# ---------------------------------------------------------------------------

def _aggregate(records: List[Dict]) -> Dict[str, float]:
    if not records:
        return {}
    keys = ["rouge_l", "biobert_f1", "concept_recall", "raiv",
            "context_tokens", "output_tokens", "total_tokens", "latency",
            "sair", "uair"]
    agg: Dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in records if k in r]
        agg[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return agg


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_json(data: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved JSON: %s", path)


def _save_csv(records: List[Dict], path: Path) -> None:
    if not records:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    logger.info("Saved CSV: %s", path)


# ---------------------------------------------------------------------------
# Experiment 1: K-Sensitivity
# ---------------------------------------------------------------------------

def run_k_sensitivity(
    query_type: str,
    max_queries: int,
    out_dir: Path,
    run_judge: bool = False,
) -> Dict:
    """For each k in 1..9, run both pipelines on simple and/or complex queries."""
    tables = []
    if query_type in ("simple", "all"):
        tables.append(("simple", SIMPLE_TABLE))
    if query_type in ("complex", "all"):
        tables.append(("complex", COMPLEX_TABLE))

    all_results: Dict = {}

    for qtype, table in tables:
        print(f"\n[K-Sensitivity] Query type: {qtype.upper()}")
        queries = _load_benchmark_queries(DYNAMIC_SQLITE_PATH, table, max_queries)
        if not queries:
            queries = _load_benchmark_queries(STATIC_SQLITE_PATH, table, max_queries)
        if not queries:
            print(f"  WARNING: No queries found for table {table!r}. Skipping.")
            continue

        print(f"  Loaded {len(queries)} queries from {table!r}")
        all_results[qtype] = {}

        for k in K_VALUES:
            print(f"  k={k} ", end="", flush=True)
            all_results[qtype][k] = {"static": [], "dynamic": []}
            
            csv_path = out_dir / "k_sensitivity" / f"k_{k}_{qtype}.csv"
            if csv_path.exists():
                print(f"(resuming from existing CSV) ", end="")
                import csv
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    records = list(reader)
                    for r in records:
                        for metric in ["rouge_l", "biobert_f1", "concept_recall", "raiv", "context_tokens", "output_tokens", "total_tokens", "latency", "sair", "uair", "top_k", "num_chunks_retrieved"]:
                            if metric in r and r[metric] != "":
                                try:
                                    r[metric] = float(r[metric]) if "." in r[metric] else int(r[metric])
                                except ValueError:
                                    pass
                        if r["pipeline"] == "static":
                            all_results[qtype][k]["static"].append(r)
                        else:
                            all_results[qtype][k]["dynamic"].append(r)
                print("done")
                continue

            for q_row in queries:
                rec_s = _run_single_eval(q_row, "static", k, STATIC_CHROMA_PATH, run_judge)
                all_results[qtype][k]["static"].append(rec_s)
                rec_d = _run_single_eval(q_row, "dynamic", k, DYNAMIC_CHROMA_PATH, run_judge)
                all_results[qtype][k]["dynamic"].append(rec_d)
                print(".", end="", flush=True)

            print(f" done ({len(queries) * 2} evals)")
            combined = all_results[qtype][k]["static"] + all_results[qtype][k]["dynamic"]
            _save_csv(combined, out_dir / "k_sensitivity" / f"k_{k}_{qtype}.csv")

        _save_json(all_results[qtype], out_dir / "k_sensitivity" / f"{qtype}_full.json")

    return all_results


# ---------------------------------------------------------------------------
# Experiment 2: Equal-Context Budget
# ---------------------------------------------------------------------------

def run_equal_budget(
    query_type: str,
    max_queries: int,
    out_dir: Path,
    run_judge: bool = False,
) -> Dict:
    """For each (static_k, dynamic_k) budget pair, run both pipelines."""
    tables = []
    if query_type in ("simple", "all"):
        tables.append(("simple", SIMPLE_TABLE))
    if query_type in ("complex", "all"):
        tables.append(("complex", COMPLEX_TABLE))

    all_results: Dict = {}

    for qtype, table in tables:
        print(f"\n[Equal Budget] Query type: {qtype.upper()}")
        queries = _load_benchmark_queries(DYNAMIC_SQLITE_PATH, table, max_queries)
        if not queries:
            queries = _load_benchmark_queries(STATIC_SQLITE_PATH, table, max_queries)
        if not queries:
            print(f"  WARNING: No queries for {table!r}. Skipping.")
            continue

        print(f"  Loaded {len(queries)} queries")
        all_results[qtype] = {}

        for (sk, dk) in EQUAL_BUDGET_PAIRS:
            label = f"s{sk}_d{dk}"
            print(f"  Budget (static_k={sk}, dynamic_k={dk}) ", end="", flush=True)
            all_results[qtype][label] = {"static": [], "dynamic": []}
            
            csv_path = out_dir / "equal_budget" / f"budget_{label}_{qtype}.csv"
            if csv_path.exists():
                print(f"(resuming from existing CSV) ", end="")
                import csv
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    records = list(reader)
                    for r in records:
                        for metric in ["rouge_l", "biobert_f1", "concept_recall", "raiv", "context_tokens", "output_tokens", "total_tokens", "latency", "sair", "uair", "top_k", "num_chunks_retrieved"]:
                            if metric in r and r[metric] != "":
                                try:
                                    r[metric] = float(r[metric]) if "." in r[metric] else int(r[metric])
                                except ValueError:
                                    pass
                        if r["pipeline"] == "static":
                            all_results[qtype][label]["static"].append(r)
                        else:
                            all_results[qtype][label]["dynamic"].append(r)
                print("done")
                continue

            for q_row in queries:
                rec_s = _run_single_eval(q_row, "static", sk, STATIC_CHROMA_PATH, run_judge)
                rec_d = _run_single_eval(q_row, "dynamic", dk, DYNAMIC_CHROMA_PATH, run_judge)
                all_results[qtype][label]["static"].append(rec_s)
                all_results[qtype][label]["dynamic"].append(rec_d)
                print(".", end="", flush=True)

            print(" done")
            combined = (
                all_results[qtype][label]["static"]
                + all_results[qtype][label]["dynamic"]
            )
            _save_csv(combined, out_dir / "equal_budget" / f"budget_{label}_{qtype}.csv")

        _save_json(all_results[qtype], out_dir / "equal_budget" / f"{qtype}_full.json")

    return all_results


# ---------------------------------------------------------------------------
# Experiment 3: Additional Information Validation (SAIR / UAIR)
# ---------------------------------------------------------------------------

def run_additional_info(
    query_type: str,
    max_queries: int,
    k_for_validation: int,
    out_dir: Path,
    pipelines: Optional[List[str]] = None,
) -> Dict:
    """At fixed k, run BOTH pipelines with LLM judge to measure SAIR/UAIR."""
    tables = []
    if query_type in ("simple", "all"):
        tables.append(("simple", SIMPLE_TABLE))
    if query_type in ("complex", "all"):
        tables.append(("complex", COMPLEX_TABLE))

    active = list(pipelines) if pipelines else ["static", "dynamic"]

    all_results: Dict = {}

    # max_queries is the TOTAL budget across the selected tables, so
    # --query_type all --max_queries 100 evaluates 100 questions overall
    # (50 simple + 50 complex), not 100 per table.
    n_tables = len(tables) or 1
    per_table = max(1, max_queries // n_tables)
    remainder = max_queries - per_table * n_tables

    for idx, (qtype, table) in enumerate(tables):
        budget = per_table + (remainder if idx == 0 else 0)
        print(f"\n[Additional Info Validation] Query type: {qtype.upper()}, k={k_for_validation}")
        queries = _load_benchmark_queries(DYNAMIC_SQLITE_PATH, table, budget)
        if not queries:
            queries = _load_benchmark_queries(STATIC_SQLITE_PATH, table, budget)
        if not queries:
            print(f"  WARNING: No queries for {table!r}. Skipping.")
            continue

        print(f"  Loaded {len(queries)} queries (with LLM judge - may be slow)")
        all_results[qtype] = {"static": [], "dynamic": []}

        csv_path = out_dir / "additional_info" / f"sair_uair_{qtype}.csv"
        if csv_path.exists():
            print(f"  (resuming from existing CSV) ", end="")
            import csv
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                records = list(reader)
                for r in records:
                    for metric in ["rouge_l", "biobert_f1", "concept_recall", "raiv", "context_tokens", "output_tokens", "total_tokens", "latency", "sair", "uair", "top_k", "num_chunks_retrieved"]:
                        if metric in r and r[metric] != "":
                            try:
                                r[metric] = float(r[metric]) if "." in r[metric] else int(r[metric])
                            except ValueError:
                                pass
                    if r["pipeline"] == "static":
                        all_results[qtype]["static"].append(r)
                    else:
                        all_results[qtype]["dynamic"].append(r)
            print("done")
            continue

        for i, q_row in enumerate(queries, 1):
            print(f"  [{i}/{len(queries)}] ", end="", flush=True)
            parts = []
            for pipe in active:
                chroma = STATIC_CHROMA_PATH if pipe == "static" else DYNAMIC_CHROMA_PATH
                rec = _run_single_eval(
                    q_row, pipe, k_for_validation, chroma, run_judge=True
                )
                all_results[qtype][pipe].append(rec)
                parts.append(f"SAIR({pipe[0]})={rec['sair']:.1f}%")
            print(" ".join(parts))

        combined = [r for pipe in active for r in all_results[qtype][pipe]]
        _save_csv(combined, out_dir / "additional_info" / f"sair_uair_{qtype}.csv")
        _save_json(all_results[qtype], out_dir / "additional_info" / f"{qtype}_full.json")

    return all_results


# ---------------------------------------------------------------------------
# Win/Loss/Tie analysis
# ---------------------------------------------------------------------------

def _win_loss_tie(
    static_agg: Dict[str, float],
    dynamic_agg: Dict[str, float],
) -> Dict[str, str]:
    """Per-metric: Dynamic wins, Static wins, or Tie."""
    lower_is_better = {"latency", "uair"}
    result: Dict[str, str] = {}
    for metric in ["rouge_l", "biobert_f1", "concept_recall", "raiv",
                   "context_tokens", "output_tokens", "total_tokens",
                   "latency", "sair", "uair"]:
        sv = static_agg.get(metric, 0.0)
        dv = dynamic_agg.get(metric, 0.0)
        if abs(sv - dv) < 1e-6:
            result[metric] = "Tie"
        elif metric in lower_is_better:
            result[metric] = "Dynamic" if dv < sv else "Static"
        else:
            result[metric] = "Dynamic" if dv > sv else "Static"
    return result


# ---------------------------------------------------------------------------
# Metric definitions for the report
# ---------------------------------------------------------------------------

METRIC_DEFS = [
    ("ROUGE-L F1",
     "Longest Common Subsequence recall between generated and reference answer (token-level). "
     "Range [0,1]. Higher is better."),
    ("BioBERT F1",
     "Cosine similarity between BioBERT-encoded generated and reference answer embeddings. "
     "Range [0,1]. Higher is better."),
    ("Concept Recall",
     "|answer_keywords intersect ref_keywords| / |ref_keywords|. "
     "Measures domain-concept coverage. Range [0,1]. Higher is better."),
    ("Ref-Aligned Info Volume (RAIV)",
     "|context_keywords intersect ref_keywords|. "
     "Counts reference concepts in the retrieved context. Higher is better."),
    ("Context Tokens",
     "Whitespace-split word count of all retrieved context. "
     "Lower values signal more precise retrieval."),
    ("Output Tokens", "Whitespace-split word count of the generated answer."),
    ("Total Tokens", "Context Tokens + Output Tokens. Proxy for inference cost."),
    ("Latency (s)",
     "Wall-clock time (seconds) for retrieval + generation. Lower is better."),
    ("SAIR (%)",
     "Source-Anchored Information Rate = (correct and source-supported claims) / "
     "total_claims * 100. Higher is better."),
    ("UAIR (%)",
     "Unsupported Addition Information Rate = unsupported_claims / "
     "total_claims * 100. Lower is better."),
]


# ---------------------------------------------------------------------------
# Report formatting helpers
# ---------------------------------------------------------------------------

def _fmt_agg_row(label: str, agg: Dict) -> str:
    cols = [
        label,
        f"{agg.get('rouge_l', 0):.4f}",
        f"{agg.get('biobert_f1', 0):.4f}",
        f"{agg.get('concept_recall', 0):.4f}",
        f"{agg.get('raiv', 0):.1f}",
        f"{agg.get('context_tokens', 0):.0f}",
        f"{agg.get('output_tokens', 0):.0f}",
        f"{agg.get('total_tokens', 0):.0f}",
        f"{agg.get('latency', 0):.3f}s",
        f"{agg.get('sair', 0):.1f}%",
        f"{agg.get('uair', 0):.1f}%",
    ]
    return "| " + " | ".join(cols) + " |"


def _table_header() -> str:
    h = ("| Config | ROUGE-L F1 | BioBERT F1 | Concept Recall | RAIV | "
         "Ctx Tokens | Out Tokens | Total Tokens | Latency | SAIR | UAIR |")
    s = ("|--------|-----------|-----------|---------------|------| "
         "-----------|-----------|-------------|---------|------|------|")
    return h + "\n" + s


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    out_dir: Path,
    timestamp: str,
    k_results: Optional[Dict],
    budget_results: Optional[Dict],
    addinfo_results: Optional[Dict],
    args: argparse.Namespace,
) -> Path:
    """Write ONE SERA_Evaluation_Report.md inside out_dir."""
    report_path = Path(out_dir) / "SERA_Evaluation_Report.md"

    # Compute aggregate tables
    k_agg: Dict = {}
    if k_results:
        for qtype, k_dict in k_results.items():
            k_agg[qtype] = {}
            for k, pipe_dict in k_dict.items():
                k_agg[qtype][k] = {}
                for pipeline, records in pipe_dict.items():
                    k_agg[qtype][k][pipeline] = _aggregate(records)

    wlt_counts: Dict = {}
    for qtype, k_dict in k_agg.items():
        wlt_counts[qtype] = {}
        for k, pipe_agg in k_dict.items():
            wlt = _win_loss_tie(
                pipe_agg.get("static", {}), pipe_agg.get("dynamic", {})
            )
            for metric, winner in wlt.items():
                wlt_counts[qtype].setdefault(
                    metric, {"Dynamic": 0, "Static": 0, "Tie": 0}
                )
                wlt_counts[qtype][metric][winner] += 1

    budget_agg: Dict = {}
    if budget_results:
        for qtype, bdict in budget_results.items():
            budget_agg[qtype] = {}
            for label, pipe_dict in bdict.items():
                budget_agg[qtype][label] = {}
                for pipeline, records in pipe_dict.items():
                    budget_agg[qtype][label][pipeline] = _aggregate(records)

    addinfo_agg: Dict = {}
    if addinfo_results:
        for qtype, pipe_dict in addinfo_results.items():
            addinfo_agg[qtype] = {}
            for pipeline, records in pipe_dict.items():
                addinfo_agg[qtype][pipeline] = _aggregate(records)

    overall_agg: Dict = {}
    if k_agg:
        for qtype, k_dict in k_agg.items():
            overall_agg[qtype] = {"static": {}, "dynamic": {}}
            for _k, pipe_agg in k_dict.items():
                for pipeline in ("static", "dynamic"):
                    for metric, val in pipe_agg.get(pipeline, {}).items():
                        overall_agg[qtype][pipeline].setdefault(metric, []).append(val)
            for pipeline in ("static", "dynamic"):
                for metric, vals in list(overall_agg[qtype][pipeline].items()):
                    if isinstance(vals, list) and vals:
                        overall_agg[qtype][pipeline][metric] = round(
                            sum(vals) / len(vals), 4
                        )

    lines: List[str] = []

    def _w(*args_: str) -> None:
        lines.extend(args_)

    # ---- Title ----
    _w(
        "# SERA Evaluation Report: Static vs Dynamic RAG",
        "",
        f"**Generated:** {timestamp}",
        f"**Generator Model:** `{GENERATOR_MODEL}`",
        f"**Judge Model:** `{JUDGE_MODEL}`",
        f"**Experiment scope:** `{args.experiment}`  |  **Query types:** `{args.query_type}`",
        f"**Max queries per table:** {args.max_queries}  |  **K for validation:** {args.k_for_validation}",
        "",
        "---",
        "",
    )

    # ---- Abstract ----
    _w(
        "## Abstract",
        "",
        "This report presents a controlled, reproducible evaluation of two Retrieval-Augmented "
        "Generation (RAG) pipelines in the SERA (Self-Evolving RAG Architecture) system. "
        "The *Static* baseline retrieves raw text chunks from the v1 ground-truth corpus; "
        "the *Dynamic* pipeline retrieves dynamically synthesised Super-Nodes produced by "
        "SERA's Phase 3/4 synthesis cycle. Both pipelines share an identical generator LLM. "
        "Evaluation spans k=1..9 retrieved documents, two query complexity tiers (Simple and "
        "Complex), an equal-context-budget comparison, and claim-level Additional Information "
        "Validation via LLM-as-judge (SAIR/UAIR). All results are presented with per-k breakdowns.",
        "",
        "---",
        "",
    )

    # ---- Central Hypothesis ----
    _w(
        "## Central Hypothesis",
        "",
        "> **H1 (Dynamic Superiority Hypothesis):** Replacing raw text chunks with "
        "dynamically synthesised Super-Nodes will produce answers with higher semantic "
        "fidelity (BioBERT F1, ROUGE-L F1), greater concept coverage (Concept Recall, RAIV), "
        "and lower hallucination rates (UAIR) compared to the Static baseline, while "
        "maintaining comparable or lower total token cost.",
        "",
        "*Note: This section is labelled **Hypothesis** - not Conclusion. "
        "The Conclusion section re-evaluates this hypothesis against the empirical results.*",
        "",
        "---",
        "",
    )

    # ---- Super-Node DB Stats ----
    _w(
        "## Super-Node Database Statistics",
        "",
        "| Statistic | Static (v1) | Dynamic |",
        "|-----------|-------------|--------|",
        "| Total raw chunks | [TO BE MEASURED] | [TO BE MEASURED] |",
        "| Total super-nodes | N/A (raw only) | [TO BE MEASURED] |",
        "| Mean super-node length (tokens) | N/A | [TO BE MEASURED] |",
        "| Median super-node length (tokens) | N/A | [TO BE MEASURED] |",
        "| Super-node topic coverage (unique clusters) | N/A | [TO BE MEASURED] |",
        "| ChromaDB collection (raw_chunks) count | [TO BE MEASURED] | [TO BE MEASURED] |",
        "| ChromaDB collection (super_nodes) count | N/A | [TO BE MEASURED] |",
        "| SQLite DB size (MB) | [TO BE MEASURED] | [TO BE MEASURED] |",
        "| Benchmark queries: Simple | [TO BE MEASURED] | [TO BE MEASURED] |",
        "| Benchmark queries: Complex | [TO BE MEASURED] | [TO BE MEASURED] |",
        "",
        "---",
        "",
    )

    # ---- Metric Definitions ----
    _w(
        "## Metric Definitions",
        "",
        "| Metric | Definition |",
        "|--------|------------|",
    )
    for name, defn in METRIC_DEFS:
        _w(f"| **{name}** | {defn} |")
    _w("", "---", "")

    # ---- K-Sensitivity Results ----
    _w(
        "## K-Sensitivity Results (k = 1 to 9)",
        "",
        "Each row is the mean over all evaluated queries at that k value.",
        "",
    )

    for qtype in ["simple", "complex"]:
        _w(f"### {qtype.capitalize()} Queries", "")
        if qtype not in k_agg or not k_agg[qtype]:
            _w("*Experiment not run or no data available.*", "")
            continue

        _w("#### Static Pipeline", "")
        _w(_table_header())
        for k in K_VALUES:
            agg = k_agg.get(qtype, {}).get(k, {}).get("static", {})
            _w(_fmt_agg_row(f"k={k}", agg))
        _w("")

        _w("#### Dynamic Pipeline", "")
        _w(_table_header())
        for k in K_VALUES:
            agg = k_agg.get(qtype, {}).get(k, {}).get("dynamic", {})
            _w(_fmt_agg_row(f"k={k}", agg))
        _w("")

        _w("#### Delta (Dynamic minus Static)", "")
        _w(_table_header())
        for k in K_VALUES:
            s_agg = k_agg.get(qtype, {}).get(k, {}).get("static", {})
            d_agg = k_agg.get(qtype, {}).get(k, {}).get("dynamic", {})
            delta = {
                metric: round(d_agg.get(metric, 0.0) - s_agg.get(metric, 0.0), 4)
                for metric in s_agg
            }
            _w(_fmt_agg_row(f"k={k}", delta))
        _w("")

    _w("---", "")

    # ---- Win/Loss/Tie ----
    metrics_list = [
        "rouge_l", "biobert_f1", "concept_recall", "raiv",
        "context_tokens", "output_tokens", "total_tokens",
        "latency", "sair", "uair",
    ]
    metric_display = {
        "rouge_l": "ROUGE-L F1", "biobert_f1": "BioBERT F1",
        "concept_recall": "Concept Recall", "raiv": "RAIV",
        "context_tokens": "Ctx Tokens", "output_tokens": "Out Tokens",
        "total_tokens": "Total Tokens", "latency": "Latency",
        "sair": "SAIR", "uair": "UAIR",
    }

    _w(
        "## Win / Loss / Tie Analysis",
        "",
        "Counts across k=1..9: how many k-values does Dynamic WIN over Static per metric?",
        "(Higher-is-better: Dynamic wins if Delta > 0; lower-is-better: Dynamic wins if Delta < 0)",
        "",
    )

    for qtype in ["simple", "complex"]:
        _w(f"### {qtype.capitalize()} Queries", "")
        _w("| Metric | Dynamic Wins | Static Wins | Ties |")
        _w("|--------|-------------|-------------|------|")
        if qtype in wlt_counts:
            for metric in metrics_list:
                counts = wlt_counts[qtype].get(
                    metric, {"Dynamic": 0, "Static": 0, "Tie": 0}
                )
                _w(
                    f"| {metric_display.get(metric, metric)} "
                    f"| {counts['Dynamic']} | {counts['Static']} | {counts['Tie']} |"
                )
        else:
            _w("| *No data* | -- | -- | -- |")
        _w("")

    _w("---", "")

    # ---- Aggregate Simple vs Complex ----
    _w(
        "## Aggregate Results: Simple vs Complex",
        "",
        "Mean across all k=1..9 runs.",
        "",
    )
    for qtype in ["simple", "complex"]:
        _w(f"### {qtype.capitalize()} Queries", "")
        _w(_table_header())
        for pipeline in ["static", "dynamic"]:
            agg = overall_agg.get(qtype, {}).get(pipeline, {})
            if isinstance(agg, dict) and agg:
                _w(_fmt_agg_row(pipeline.capitalize(), agg))
            else:
                na = " | ".join(["*N/A*"] * 10)
                _w(f"| {pipeline.capitalize()} | {na} |")
        _w("")

    _w("---", "")

    # ---- Equal-Context-Budget ----
    _w(
        "## Equal-Context-Budget Comparison",
        "",
        "Budget pairs are chosen so that static and dynamic pipelines consume approximately "
        "the same total context tokens (raw chunks are small; super-nodes are large).",
        "",
        "### Budget Pairs",
        "",
        "| Budget Pair | Static k | Dynamic k | Rationale |",
        "|-------------|----------|-----------|-----------|",
    )
    for sk, dk in EQUAL_BUDGET_PAIRS:
        _w(f"| s{sk}_d{dk} | {sk} | {dk} | "
           f"~{sk} raw-chunk token budget matched to {dk} super-node budget |")
    _w("")

    for qtype in ["simple", "complex"]:
        _w(f"### {qtype.capitalize()} Queries - Budget Results", "")
        if qtype not in budget_agg or not budget_agg[qtype]:
            _w("*Experiment not run or no data available.*", "")
            continue
        _w(_table_header())
        for (sk, dk) in EQUAL_BUDGET_PAIRS:
            label = f"s{sk}_d{dk}"
            for pipeline in ["static", "dynamic"]:
                pklabel = f"s{sk}" if pipeline == "static" else f"d{dk}"
                row_label = f"{label} ({pipeline.capitalize()}, k={pklabel})"
                agg = budget_agg.get(qtype, {}).get(label, {}).get(pipeline, {})
                _w(_fmt_agg_row(row_label, agg if agg else {}))
        _w("")

    _w("---", "")

    # ---- Additional Info Validation ----
    _w(
        "## Additional Information Validation (SAIR / UAIR)",
        "",
        f"Fixed k={args.k_for_validation}. LLM-as-judge (`{JUDGE_MODEL}`) decomposes "
        "each generated answer into atomic claims and classifies them as "
        "correct+supported, incorrect, or unsupported.",
        "",
        "| Query Type | Pipeline | SAIR (%) | UAIR (%) | Avg Total Claims | Avg Supported |",
        "|------------|----------|----------|----------|-----------------|---------------|",
    )
    if addinfo_agg:
        for qtype in ["simple", "complex"]:
            if qtype in addinfo_agg:
                for pipeline in ["static", "dynamic"]:
                    agg = addinfo_agg[qtype].get(pipeline, {})
                    sair_val = agg.get("sair", 0.0)
                    uair_val = agg.get("uair", 0.0)
                    total_c = agg.get("total_claims", 0.0)
                    supp_c = round(total_c * sair_val / 100.0, 1) if total_c else 0
                    _w(
                        f"| {qtype.capitalize()} | {pipeline.capitalize()} | "
                        f"{sair_val:.1f}% | {uair_val:.1f}% | "
                        f"{total_c:.1f} | {supp_c:.1f} |"
                    )
    else:
        _w("| *N/A* | *Experiment not run* | -- | -- | -- | -- |")

    _w("", "---", "")

    # ---- Limitations ----
    _w(
        "## Limitations",
        "",
        "1. **Token counting** is done by whitespace splitting, not a model tokeniser. "
        "Results may differ from actual API token counts by +-10-20%.",
        "2. **BioBERT F1** is sentence-embedding cosine similarity (not token-level "
        f"BERTScore), computed with `{_BIOBERT_MODEL_NAME}`.",
        f"3. **SAIR/UAIR** depends on the judge model (`{JUDGE_MODEL}`). "
        "Judge agreement with human annotators was not verified in this run.",
        "4. **Dynamic Super-Nodes** are retrieved without epsilon-greedy exploration "
        "or SU5 hierarchical masking (active in production). This isolates the quality "
        "of the synthesised content itself.",
        "5. **Query sets** are drawn from `benchmark_qa_simple` and `benchmark_qa_complex`, "
        "seeded programmatically. They may not fully represent real-world query distributions.",
        "6. **Equal-budget pairs** are heuristic approximations; actual token equality "
        "depends on super-node content length, which varies.",
        "",
        "---",
        "",
    )

    # ---- Methodological Safeguards ----
    _w(
        "## Methodological Safeguards",
        "",
        "The following design choices ensure a fair, unbiased comparison:",
        "",
        "| What was NOT done | Why it matters |",
        "|-------------------|----------------|",
        "| Generator LLM **not** changed between pipelines | Quality difference attributable to retrieval, not generation |",
        "| Judge LLM **not** told which pipeline produced which answer | Prevents judge bias |",
        "| Dynamic pipeline does **not** use raw-chunk fallback | Tests pure super-node quality |",
        "| Epsilon-greedy exploration **disabled** in evaluation | Ensures deterministic, reproducible retrieval |",
        "| Benchmark queries **not** used during SERA synthesis cycle | Prevents data leakage |",
        "| Results **not** post-filtered for good queries | All queries run regardless of answer quality |",
        "",
        "---",
        "",
    )

    # ---- Conclusion ----
    total_dyn_wins = sum(
        wlt_counts[qt][m].get("Dynamic", 0)
        for qt in wlt_counts
        for m in wlt_counts[qt]
    )
    total_sta_wins = sum(
        wlt_counts[qt][m].get("Static", 0)
        for qt in wlt_counts
        for m in wlt_counts[qt]
    )

    if total_dyn_wins > total_sta_wins:
        verdict = (
            f"The empirical results **support** H1: the Dynamic pipeline wins on more "
            f"metric-k combinations ({total_dyn_wins} Dynamic wins vs {total_sta_wins} Static wins). "
            "Individual metric outcomes should be inspected in the tables above for a complete picture."
        )
    elif total_sta_wins > total_dyn_wins:
        verdict = (
            f"The empirical results **do not support** H1 in aggregate: the Static pipeline "
            f"wins on more metric-k combinations ({total_sta_wins} Static wins vs {total_dyn_wins} Dynamic wins). "
            "This may indicate that super-node synthesis quality needs further tuning, or that "
            "the benchmark query set favours lexically close retrieval."
        )
    else:
        verdict = (
            "The empirical results are **inconclusive** - Static and Dynamic pipelines perform "
            "comparably in aggregate. Refer to per-metric and per-k tables for nuance."
        )

    _w(
        "## Conclusion",
        "",
        "### Central Hypothesis Re-evaluation",
        "",
        "> **H1 (Dynamic Superiority Hypothesis):** Replacing raw text chunks with "
        "dynamically synthesised Super-Nodes will produce answers with higher semantic "
        "fidelity (BioBERT F1, ROUGE-L F1), greater concept coverage (Concept Recall, RAIV), "
        "and lower hallucination rates (UAIR) compared to the Static baseline, while "
        "maintaining comparable or lower total token cost.",
        "",
        verdict,
        "",
        "Full numerical evidence is available in the K-Sensitivity, Win/Loss/Tie, and "
        "Additional Information Validation sections above.",
        "",
        "---",
        "",
        f"*Report generated automatically by `scripts/eval_static_vs_dynamic.py` at {timestamp}.*",
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"\n[Report] Written: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SERA Static vs Dynamic RAG Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        choices=["k_sensitivity", "equal_budget", "additional_info", "all"],
        default="all",
        help="Which experiment(s) to run (default: all)",
    )
    parser.add_argument(
        "--query_type",
        choices=["simple", "complex", "all"],
        default="all",
        help="Which query table(s) to use (default: all)",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=100,
        help="Maximum number of queries per table (default: 100)",
    )
    parser.add_argument(
        "--pipeline",
        choices=["static", "dynamic", "both"],
        default="both",
        help="Which retrieval pipeline(s) to evaluate (default: both). "
             "Use dynamic to skip the static baseline; note that static-vs-dynamic "
             "comparisons and win/loss tables require both.",
    )
    parser.add_argument(
        "--k_for_validation",
        type=int,
        default=5,
        help="Fixed k used for Additional Information Validation (default: 5)",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        default=False,
        help="Enable LLM judge for SAIR/UAIR during k_sensitivity and equal_budget (slow)",
    )
    parser.add_argument(
        "--resume_dir",
        type=str,
        default="",
        help="Path to an existing results dir to resume from",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume_dir:
        out_dir = Path(args.resume_dir)
        print(f"Resuming from: {out_dir}")
    else:
        out_dir = _PROJECT_ROOT / "data" / "results" / f"eval_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SERA Static vs Dynamic RAG Evaluation")
    print("=" * 70)
    print(f"  Generator model : {GENERATOR_MODEL}")
    print(f"  Judge model     : {JUDGE_MODEL}")
    print(f"  Experiment      : {args.experiment}")
    print(f"  Pipeline(s)     : {args.pipeline}")
    print(f"  Query type      : {args.query_type}")
    print(f"  Max queries     : {args.max_queries}")
    print(f"  K for val.      : {args.k_for_validation}")
    print(f"  Output dir      : {out_dir}")
    print(f"  Static  Chroma  : {STATIC_CHROMA_PATH}")
    print(f"  Static  SQLite  : {STATIC_SQLITE_PATH}")
    print(f"  Dynamic Chroma  : {DYNAMIC_CHROMA_PATH}")
    print(f"  Dynamic SQLite  : {DYNAMIC_SQLITE_PATH}")
    print("=" * 70)

    needs_judge = args.judge or args.experiment in ("additional_info", "all")
    print("")
    print("[Preflight] Checking model availability...")
    _preflight_models(need_judge=needs_judge)

    print("\n[Init] Loading BioBERT embedding model...")
    _get_biobert_model()
    print("[Init] Done.")

    k_results: Optional[Dict] = None
    budget_results: Optional[Dict] = None
    addinfo_results: Optional[Dict] = None

    if args.experiment in ("k_sensitivity", "all"):
        print("\n[1/3] Running K-Sensitivity experiment...")
        k_results = run_k_sensitivity(
            query_type=args.query_type,
            max_queries=args.max_queries,
            out_dir=out_dir,
            run_judge=args.judge,
        )
        _save_json(k_results, out_dir / "k_sensitivity" / "all_results.json")
        print("[1/3] K-Sensitivity complete.")

    if args.experiment in ("equal_budget", "all"):
        print("\n[2/3] Running Equal-Budget experiment...")
        budget_results = run_equal_budget(
            query_type=args.query_type,
            max_queries=args.max_queries,
            out_dir=out_dir,
            run_judge=args.judge,
        )
        _save_json(budget_results, out_dir / "equal_budget" / "all_results.json")
        print("[2/3] Equal-Budget complete.")

    if args.experiment in ("additional_info", "all"):
        print("\n[3/3] Running Additional Information Validation (SAIR/UAIR)...")
        addinfo_results = run_additional_info(
            query_type=args.query_type,
            max_queries=args.max_queries,
            k_for_validation=args.k_for_validation,
            out_dir=out_dir,
            pipelines=(["static", "dynamic"] if args.pipeline == "both" else [args.pipeline]),
        )
        _save_json(addinfo_results, out_dir / "additional_info" / "all_results.json")
        print("[3/3] Additional Info Validation complete.")

    print("\n[Report] Generating SERA_Evaluation_Report.md ...")
    report_path = generate_report(
        out_dir=out_dir,
        timestamp=timestamp,
        k_results=k_results,
        budget_results=budget_results,
        addinfo_results=addinfo_results,
        args=args,
    )

    print("\n" + "=" * 70)
    print("  Evaluation complete!")
    print(f"  Output directory : {out_dir}")
    print(f"  Final report     : {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
