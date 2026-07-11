import logging
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict

from app.db import sqlite_client
from app.patterns import finder
from app.synthesis import synthesizer

logger = logging.getLogger(__name__)

@dataclass
class SynthesisJob:
    cluster_id: int
    canonical_query: str
    chunk_ids: List[str]
    hit_count: int
    triggered_at: datetime
    existing_sn_id: Optional[str] = None


def scan_and_trigger() -> Dict[str, int]:
    """
    The actual APScheduler job payload.
    Returns a summary dict: { "triggered": int, "skipped": int, "failed": int }
    """
    try:
        ready = finder.get_ready_clusters()
    except Exception as e:
        logger.error(f"Failed to fetch ready clusters: {e}", exc_info=True)
        return {"triggered": 0, "skipped": 0, "failed": 0}

    results = {"triggered": 0, "skipped": 0, "failed": 0}

    for cluster in ready:
        job = SynthesisJob(
            cluster_id=cluster.cluster_id,
            canonical_query=cluster.canonical_query,
            chunk_ids=cluster.chunk_ids,
            hit_count=cluster.hit_count,
            triggered_at=datetime.utcnow(),
            existing_sn_id=cluster.existing_sn_id,
        )
        try:
            # SU12: Lock BEFORE calling synthesizer
            sqlite_client.set_synthesizing(cluster.cluster_id, True)
            
            # Phase 4 hook
            super_node_id = synthesizer.run(job)
            
            # Mark synthesized in query_clusters
            sqlite_client.mark_cluster_synthesized(cluster.cluster_id, super_node_id)
            
            results["triggered"] += 1
            logger.info(f"Successfully synthesized cluster {cluster.cluster_id} -> {super_node_id}")
            
        except NotImplementedError as e:
            # Phase 4 stub raises this
            sqlite_client.set_synthesizing(cluster.cluster_id, False)
            logger.warning(f"Synthesis skipped for cluster {cluster.cluster_id}: {e}")
            results["skipped"] += 1
        except Exception as e:
            # Any synthesis error
            sqlite_client.set_synthesizing(cluster.cluster_id, False)
            logger.error(f"Synthesis failed cluster {cluster.cluster_id}: {e}", exc_info=True)
            results["failed"] += 1

    return results
