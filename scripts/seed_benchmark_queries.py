"""
Seed benchmark queries into SERA's query logger so clusters form
and get synthesized into Super Nodes for the benchmark topics.
Each query is fed `hit_threshold` times to cross the synthesis threshold.
"""
import sys, os
sys.path.insert(0, r'C:\Projects\SERA')
from dotenv import load_dotenv
load_dotenv(r'C:\Projects\SERA\.env')

import app.config as config
from app.retrieval.retriever import Retriever
from app.logging_.query_logger import log_query

from scripts.seed_complex_queries_v2 import QUESTIONS as BENCHMARK_QUERIES

HIT_THRESHOLD = 15  # Feed each query 15 times to guarantee cluster formation

print(f"Seeding {len(BENCHMARK_QUERIES)} benchmark queries x{HIT_THRESHOLD} hits each...")
print("=" * 60)

for qi, query in enumerate(BENCHMARK_QUERIES, 1):
    print(f"\n[{qi}/{len(BENCHMARK_QUERIES)}] Query: {query[:80]}...")
    chunks = Retriever.search(query, top_k=5)
    chunk_ids = [c["id"] for c in chunks]
    print(f"  Retrieved {len(chunk_ids)} chunks: {chunk_ids[:3]}")

    for hit in range(HIT_THRESHOLD):
        log_query(query, chunk_ids)
        if (hit + 1) % 5 == 0:
            print(f"  Hit {hit+1}/{HIT_THRESHOLD} logged...")

    print(f"  Done - {HIT_THRESHOLD} hits logged for this query")

print("\n" + "=" * 60)
print("ALL QUERIES SEEDED. Now triggering synthesis...")
from app.scheduler.jobs.pattern_finder import scan_and_trigger
res = scan_and_trigger()
print("Synthesis results:", res)
print("=" * 60)
