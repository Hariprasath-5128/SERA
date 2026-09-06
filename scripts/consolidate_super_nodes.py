import os
import sys
import sqlite3
import numpy as np
import uuid
import json
import requests
from datetime import datetime
from collections import defaultdict
import networkx as nx

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.db.chroma_client import ChromaClient
from app.ingestion.embedder import Embedder
from app.validation import validator

SIMILARITY_THRESHOLD = 0.90  
LLM_API_URL = "http://localhost:11434/api/generate"

def cosine_similarity(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))

def consolidate_texts(texts: list[str]) -> str:
    prompt = "You are a medical expert. Combine the following medical text snippets into a single, comprehensive, highly accurate summary.\n"
    prompt += "RULES:\n1. Preserve ALL unique factual information.\n2. Remove repeated/redundant facts.\n3. Combine equivalent statements.\n4. Avoid introducing new unsupported information.\n5. Output ONLY the final merged text.\n\nTEXTS:\n"
    for i, t in enumerate(texts):
        prompt += f"--- TEXT {i+1} ---\n{t}\n\n"
        
    try:
        res = requests.post(LLM_API_URL, json={
            "model": "llama3:8b",
            "prompt": prompt,
            "stream": False
        })
        return res.json().get("response", "").strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""

def run_consolidation():
    print(f"\n{'='*60}")
    print("SERA OVERLAP DETECTION & EVIDENCE-PRESERVING MERGE (WITH RE-VALIDATION)")
    print(f"{'='*60}\n")

    sn_col = ChromaClient.get_super_nodes_collection()
    if not sn_col:
        print("No Super Nodes collection found.")
        return

    res = sn_col.get(include=["embeddings", "metadatas", "documents"])
    ids = res.get("ids", [])
    embeddings = res.get("embeddings", [])
    docs = res.get("documents", [])
    
    if not ids or len(ids) < 2:
        print("Not enough Super Nodes to deduplicate.")
        return

    print(f"Loaded {len(ids)} Super Nodes. Building semantic overlap graph...")

    G = nx.Graph()
    for id_val in ids:
        G.add_node(id_val)
        
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            sim = cosine_similarity(np.array(embeddings[i]), np.array(embeddings[j]))
            if sim >= SIMILARITY_THRESHOLD:
                G.add_edge(ids[i], ids[j], weight=sim)

    overlap_groups = [list(comp) for comp in nx.connected_components(G) if len(comp) > 1]
    
    if not overlap_groups:
        print("\nNo overlapping Super Node groups found! The database is clean.")
        return

    print(f"\nFound {len(overlap_groups)} Overlap Groups to consolidate.\n")
    
    db_path = os.path.join(project_root, "data", "sqlite", "rag.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    model = Embedder.get_model()

    total_merged = 0
    rejected = 0
    for group_idx, group in enumerate(overlap_groups, 1):
        print(f"[{group_idx}/{len(overlap_groups)}] Consolidating Group (size {len(group)}): {', '.join(group)}")
        
        texts = []
        for node_id in group:
            idx = ids.index(node_id)
            texts.append(docs[idx])
            
        merged_text = consolidate_texts(texts)
        if not merged_text:
            continue
            
        # STEP 6: RE-VALIDATION (Flagged concept clarification)
        print("   -> Running LLM-as-Judge Re-Validation (Entailment Check)...")
        source_concat = "\n\n".join(texts)
        
        val_res = validator.validate(source_text=source_concat, summary=merged_text)
        
        if not val_res.passed:
            print(f"   -> [REJECTED] Merge dropped critical facts! Flagged missing: {len(val_res.missing_facts)} concepts.")
            if val_res.missing_facts:
                print(f"      Missing example: {val_res.missing_facts[0][:60]}...")
            rejected += 1
            continue
            
        print("   -> [PASSED] Validation passed. All concepts preserved.")
            
        emb = model.encode([merged_text], convert_to_numpy=True, show_progress_bar=False)[0].tolist()
        new_sn_id = f"sn_consolidated_{uuid.uuid4().hex[:8]}"
        
        sn_col.add(
            ids=[new_sn_id],
            embeddings=[emb],
            documents=[merged_text],
            metadatas=[{
                "type": "super_node", 
                "consolidated_from": json.dumps(group),
                "created_at": datetime.utcnow().isoformat()
            }]
        )
        
        for old_id in group:
            c.execute("UPDATE query_clusters SET super_node_id = ? WHERE super_node_id = ?", (new_sn_id, old_id))
            sn_col.delete(ids=[old_id])
            total_merged += 1
            
        conn.commit()
        print(f"   -> Success! Created consolidated node: {new_sn_id}\n")

    conn.close()

    print(f"{'='*60}")
    print(f"SUCCESS: Consolidations Passed: {len(overlap_groups) - rejected} | Rejected: {rejected}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    run_consolidation()
