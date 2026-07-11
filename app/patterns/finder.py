import hashlib
import json
import logging
from dataclasses import dataclass
from typing import List, Optional
import sqlite3

import numpy as np

from app import config
from app.db import sqlite_client
from app.db.chroma_client import ChromaClient

logger = logging.getLogger(__name__)

@dataclass
class PatternResult:
    cluster_id: int
    canonical_query: str
    chunk_ids: List[str]
    hit_count: int
    existing_sn_id: Optional[str]

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def should_resynthesize(cluster: sqlite3.Row, last_history_entry: sqlite3.Row) -> bool:
    """
    Evaluates compound trigger conditions (Upgrade U2).
    Returns True if:
      (B) chunk_ids_changed
      (C) centroid_shift > CENTROID_SHIFT_THRESHOLD
      (D) source_documents_updated
    """
    # 1. chunk_ids_changed
    cluster_chunk_ids = json.loads(cluster["chunk_ids"])
    sorted_ids_json = json.dumps(sorted(cluster_chunk_ids)).encode('utf-8')
    current_chunk_ids_hash = sha256(sorted_ids_json)
    
    if current_chunk_ids_hash != last_history_entry["chunk_ids_hash"]:
        logger.debug(f"Cluster {cluster['id']}: chunk_ids changed.")
        return True

    # 2. centroid_shift
    current_centroid_bytes = cluster["query_embedding"]
    current_centroid_hash = sha256(current_centroid_bytes)
    
    if current_centroid_hash != last_history_entry["centroid_hash"]:
        current_centroid = np.frombuffer(current_centroid_bytes, dtype=np.float32)
        last_centroid = np.frombuffer(last_history_entry["centroid"], dtype=np.float32)
        
        shift = 1.0 - cosine_sim(current_centroid, last_centroid)
        if shift > config.CENTROID_SHIFT_THRESHOLD:
            logger.debug(f"Cluster {cluster['id']}: centroid shifted by {shift:.4f}.")
            return True

    # 3. source_documents_updated
    if cluster_chunk_ids:
        try:
            collection = ChromaClient.get_collection()
            results = collection.get(
                ids=cluster_chunk_ids,
                include=["metadatas"]
            )
            metadatas = results.get("metadatas", [])
            last_timestamp = last_history_entry["timestamp"]
            
            for meta in metadatas:
                if meta and "updated_at" in meta:
                    if meta["updated_at"] > last_timestamp:
                        logger.debug(f"Cluster {cluster['id']}: source document updated.")
                        return True
        except Exception as e:
            logger.error(f"Error checking ChromaDB for source updates: {e}")

    return False

def get_ready_clusters(
    min_hit_count: int = config.HIT_COUNT_THRESHOLD,
    resynth_delta: int = config.RESYNTH_DELTA,
) -> List[PatternResult]:
    """
    Queries SQLite for clusters that are ready for synthesis.
    """
    # (A) Delta Gate via fast SQL scan
    candidates = sqlite_client.get_ready_clusters(min_hit_count, resynth_delta)
    results = []

    for cluster in candidates:
        last_entry = sqlite_client.get_latest_history_entry(cluster["id"])
        
        # First synthesis — always proceed
        if last_entry is None:
            results.append(PatternResult(
                cluster_id=cluster["id"],
                canonical_query=cluster["canonical_query"],
                chunk_ids=json.loads(cluster["chunk_ids"]),
                hit_count=cluster["hit_count"],
                existing_sn_id=None
            ))
            continue
            
        # Re-synthesis — check compound trigger conditions
        if should_resynthesize(cluster, last_entry):
            results.append(PatternResult(
                cluster_id=cluster["id"],
                canonical_query=cluster["canonical_query"],
                chunk_ids=json.loads(cluster["chunk_ids"]),
                hit_count=cluster["hit_count"],
                existing_sn_id=cluster["super_node_id"]
            ))
        else:
            logger.debug(f"Cluster {cluster['id']}: delta met but no content change — skip.")

    return results
