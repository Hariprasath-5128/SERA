import os
import sys
import sqlite3
import logging
import json
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.abspath('.'))
from app.db import sqlite_client
from app.retrieval.retriever import Retriever
from app.synthesis import synthesizer
from datetime import datetime
from app.synthesis.synthesizer import SynthesisJob

def run_it():
    db_path = os.path.join("data", "sqlite", "rag.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT question FROM benchmark_qa")
    questions = [row[0] for row in c.fetchall()]
    needed = set()
    for q in questions:
        raw = Retriever.search(q, top_k=5)
        for cand in raw:
            chunk_id = cand['id']
            c.execute("SELECT id, chunk_ids FROM query_clusters")
            for row in c.fetchall():
                try:
                    c_ids = json.loads(row[1])
                    if chunk_id in c_ids:
                        needed.add(row[0])
                except:
                    pass
            
    count = 0
    for cluster_id in needed:
        c.execute("SELECT canonical_query, super_node_id, synthesized, chunk_ids FROM query_clusters WHERE id = ?", (cluster_id,))
        row = c.fetchone()
        if not row or row[2] == 1: continue
        
        try:
            chunk_ids = json.loads(row[3])
        except:
            chunk_ids = []
            
        job = SynthesisJob(
            cluster_id=cluster_id,
            canonical_query=row[0],
            chunk_ids=chunk_ids,
            hit_count=10,
            triggered_at=datetime.utcnow(),
            existing_sn_id=row[1],
        )
        try:
            logger.info(f"Synthesizing Cluster {cluster_id} with REAL LLM...")
            sqlite_client.set_synthesizing(cluster_id, True)
            super_node_id = synthesizer.run(job)
            sqlite_client.mark_cluster_synthesized(cluster_id, super_node_id)
            count += 1
            logger.info(f"Successfully synthesized {count}/{len(needed)}: {super_node_id}")
        except Exception as e:
            sqlite_client.set_synthesizing(cluster_id, False)
            logger.error(f"Failed {cluster_id}: {e}")
    conn.close()

if __name__ == '__main__':
    run_it()
