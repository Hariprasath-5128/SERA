import sys
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath('.'))

from app.db import sqlite_client
from app.patterns import finder
from app.synthesis import synthesizer
from datetime import datetime
from app.synthesis.synthesizer import SynthesisJob

def synthesize_30():
    try:
        ready = finder.get_ready_clusters()
    except Exception as e:
        logger.error(f"Failed to fetch: {e}")
        return
        
    count = 0
    for cluster in ready:
        if count >= 30:
            break
            
        job = SynthesisJob(
            cluster_id=cluster.cluster_id,
            canonical_query=cluster.canonical_query,
            chunk_ids=cluster.chunk_ids,
            hit_count=cluster.hit_count,
            triggered_at=datetime.utcnow(),
            existing_sn_id=cluster.existing_sn_id,
        )
        try:
            if not cluster.chunk_ids:
                sqlite_client.mark_cluster_failed(cluster.cluster_id)
                continue
                
            sqlite_client.set_synthesizing(cluster.cluster_id, True)
            super_node_id = synthesizer.run(job)
            sqlite_client.mark_cluster_synthesized(cluster.cluster_id, super_node_id)
            count += 1
            logger.info(f"Synthesized {count}/30: {super_node_id}")
            
        except Exception as e:
            sqlite_client.set_synthesizing(cluster.cluster_id, False)
            logger.error(f"Failed {cluster.cluster_id}: {e}")

if __name__ == '__main__':
    logger.info('Starting 30 samples...')
    synthesize_30()
    logger.info('Done!')
