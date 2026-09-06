"""
validate_complex_queries_v2.py
------------------------------
Retrieval-only check (no LLM) that the new complex questions satisfy
design condition #1: the DYNAMIC pipeline surfaces a relevant Super Node,
and that Super Node ranks at or near the top (ahead of raw chunks).

Mirrors the dynamic-retrieval block of scripts/benchmark_db.py.
"""
import os
import sys
import sqlite3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import app.config as config
from app.retrieval.retriever import Retriever
from app.db.chroma_client import ChromaClient
from app.db import super_node_store

# Point config at the DYNAMIC stores (same as benchmark_db.py dynamic block)
config.CHROMA_PERSIST_PATH = os.path.join(PROJECT_ROOT, "data", "chroma")
config.SQLITE_DB_PATH = os.path.join(PROJECT_ROOT, "data", "sqlite", "rag.db")
ChromaClient._client = None
ChromaClient._collection = None
ChromaClient._super_nodes_collection = None
super_node_store._collection = None

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 30

conn = sqlite3.connect(config.SQLITE_DB_PATH)
qs = [r[0] for r in conn.execute(
    "SELECT question FROM benchmark_qa_complex LIMIT ?", (LIMIT,)
)]
conn.close()

sn_present = 0
sn_top1 = 0
rows = []
for i, q in enumerate(qs, 1):
    res = Retriever.search(q, top_k=20)
    types = [c.get("type") for c in res]
    sn_ranks = [j for j, t in enumerate(types) if t in ("super_node", "meta_node")]
    first_sn = sn_ranks[0] + 1 if sn_ranks else None
    n_sn = len(sn_ranks)
    top_type = types[0] if types else None
    if n_sn:
        sn_present += 1
    if top_type in ("super_node", "meta_node"):
        sn_top1 += 1
    # title of the first super node
    sn_title = ""
    if sn_ranks:
        doc = (res[sn_ranks[0]].get("text") or "").strip().split("\n")
        sn_title = next((l.strip("*# ").strip() for l in doc if l.strip()), "")[:70]
    rows.append((i, first_sn, n_sn, top_type, sn_title, q[:60]))

print(f"{'#':>3} {'1stSN':>5} {'nSN':>4} {'top':>10}  {'super-node matched':<70}  query")
print("-" * 160)
for i, first_sn, n_sn, top_type, sn_title, q in rows:
    print(f"{i:>3} {str(first_sn):>5} {n_sn:>4} {str(top_type):>10}  {sn_title:<70}  {q}")

print("-" * 160)
print(f"Sampled {len(qs)} complex queries")
print(f"  super-node present in top-20 : {sn_present}/{len(qs)} ({100*sn_present/len(qs):.0f}%)")
print(f"  super-node is rank #1        : {sn_top1}/{len(qs)} ({100*sn_top1/len(qs):.0f}%)")
