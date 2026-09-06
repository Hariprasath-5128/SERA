import os
import sys
import sqlite3
import numpy as np
from typing import List, Dict

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.db.chroma_client import ChromaClient

SIMILARITY_THRESHOLD = 0.90  

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

def run_deduplication():
    print(f"\n{'='*60}")
    print("SUPER NODE DEDUPLICATION & CONSOLIDATION")
    print(f"{'='*60}\n")

    sn_col = ChromaClient.get_super_nodes_collection()
    if not sn_col:
        print("No Super Nodes collection found.")
        return

    res = sn_col.get(include=["embeddings", "metadatas", "documents"])
    ids = res.get("ids", [])
    embeddings = res.get("embeddings", [])
    
    if not ids or len(ids) < 2:
        print("Not enough Super Nodes to deduplicate.")
        return

    print(f"Loaded {len(ids)} Super Nodes. Analyzing semantic overlap...")

    duplicates_found = []
    processed = set()

    for i in range(len(ids)):
        if ids[i] in processed:
            continue
            
        for j in range(i + 1, len(ids)):
            if ids[j] in processed:
                continue

            sim = cosine_similarity(np.array(embeddings[i]), np.array(embeddings[j]))
            if sim >= SIMILARITY_THRESHOLD:
                print(f" [OVERLAP DETECTED] Sim: {sim:.3f} | {ids[i]} <--> {ids[j]}")
                duplicates_found.append((ids[i], ids[j]))
                processed.add(ids[j])

    if not duplicates_found:
        print("\nNo highly overlapping Super Nodes found! The database is clean.")
        return

    print(f"\nFound {len(duplicates_found)} redundant Super Nodes to consolidate.")
    
    db_path = os.path.join(project_root, "data", "sqlite", "rag.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    deleted_count = 0
    for keep_id, drop_id in duplicates_found:
        print(f" -> Consolidating {drop_id} INTO {keep_id}...")
        
        c.execute("UPDATE query_clusters SET super_node_id = ? WHERE super_node_id = ?", (keep_id, drop_id))
        
        sn_col.delete(ids=[drop_id])
        deleted_count += 1

    conn.commit()
    conn.close()

    print(f"\n{'='*60}")
    print(f"SUCCESS: Permanently removed {deleted_count} redundant Super Nodes.")
    print(f"The LLM context window will no longer suffer from this overlap!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    run_deduplication()
