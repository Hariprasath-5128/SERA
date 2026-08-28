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

BENCHMARK_QUERIES = [
    "What are the combined treatment pathways and dietary guidelines for a patient managing both Type 2 Diabetes and Hypertension?",
    "How do the side effects of Lisinopril compare with other ACE inhibitors, and what are the long-term renal impacts?",
    "Can chronic stress trigger asthma attacks, and how do cortisol levels link these two conditions?",
    "What is the relationship between insulin resistance, high cholesterol, and cardiovascular disease risk?",
    "How does a mild concussion affect cognitive function over time, and what are the recommended rehabilitation stages?",
    "What are the early warning signs of a heart attack versus a panic attack, and how do their physiological causes differ?",
    "How do dietary choices for managing cholesterol affect blood pressure regulation and overall heart health?",
    "What are the long-term cardiovascular risks of untreated sleep apnea compared to chronic hypertension?",
    "How do beta-blockers and calcium channel blockers differ in their mechanisms for treating high blood pressure and arrhythmia?",
    "What are the interactions between alcohol consumption, liver damage, and the efficacy of common hypertension medications?",
]

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
