import os
import sys
import sqlite3
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath('.'))

from app.db import sqlite_client
from app.synthesis import synthesizer
from app.synthesis.synthesizer import SynthesisJob

def run_synthesis():
    db_path = os.path.join("data", "sqlite", "rag.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Get all unsynthesized clusters
    c.execute("SELECT id, canonical_query, super_node_id, chunk_ids, hit_count FROM query_clusters WHERE synthesized = 0 ORDER BY hit_count DESC")
    clusters = c.fetchall()
    
    logger.info(f"Found {len(clusters)} unsynthesized clusters. Beginning synthesis...")
    
    count = 0
    for cluster in clusters:
        cluster_id = cluster[0]
        canonical_query = cluster[1]
        super_node_id = cluster[2]
        
        try:
            chunk_ids = json.loads(cluster[3]) if cluster[3] else []
        except:
            chunk_ids = []
            
        hit_count = cluster[4]
            
        job = SynthesisJob(
            cluster_id=cluster_id,
            canonical_query=canonical_query,
            chunk_ids=chunk_ids,
            hit_count=hit_count,
            triggered_at=datetime.utcnow(),
            existing_sn_id=super_node_id,
        )
        
        try:
            logger.info(f"[{count+1}/{len(clusters)}] Synthesizing: {canonical_query[:50]}...")
            sqlite_client.set_synthesizing(cluster_id, True)
            new_sn_id = synthesizer.run(job)
            sqlite_client.mark_cluster_synthesized(cluster_id, new_sn_id)
            count += 1
            logger.info(f"   -> Success! Super Node ID: {new_sn_id}")
        except Exception as e:
            sqlite_client.set_synthesizing(cluster_id, False)
            logger.error(f"   -> Failed {cluster_id}: {e}")
            
    conn.close()
    logger.info(f"Finished synthesizing {count} nodes.")

if __name__ == '__main__':
    run_synthesis()
