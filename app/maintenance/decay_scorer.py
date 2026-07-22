import math
import logging
from datetime import datetime, timezone
from app import config
from app.db import sqlite_client
from app.db.chroma_client import ChromaClient
from app.db import super_node_store

logger = logging.getLogger(__name__)

def compute_exponential_decay(meta: dict, half_life_days: float = 7.0) -> float:
    """
    SU3 — Half-Life Exponential Decay.
    Judges relevancy based on recency of access, not total age.
    """
    last_accessed_str = meta.get("last_accessed")
    if not last_accessed_str:
        # Fallback to created_at if last_accessed doesn't exist
        last_accessed_str = meta.get("created_at")

    if not last_accessed_str:
        # Fallback to current time if neither exists
        return 1.0

    last_accessed = datetime.fromisoformat(last_accessed_str)
    
    # Use timezone-aware subtraction to avoid TypeError with offset-aware datetimes
    now = datetime.now(timezone.utc) if last_accessed.tzinfo is not None else datetime.utcnow()
    
    # Calculate days elapsed (allowing decimal places for precision)
    days_since_last_access = max(0.0, (now - last_accessed).total_seconds() / (24.0 * 3600.0))

    # Exponential half-life decay factor
    decay_factor = math.exp(-days_since_last_access / half_life_days)

    # Cap SQLite true access count at 10 to prevent runaway scores
    access_count = meta.get("access_count_sqlite", 0)
    base_utility = min(access_count, 10)

    return float(base_utility * decay_factor)


def compute_and_apply_decay() -> None:
    """
    SU10 integration: pulls true access_count from SQLite before computing decay.
    Syncs the authoritative SQLite hit_count into ChromaDB metadata as access_count.
    Deletes the node from ChromaDB if the score falls below config.DECAY_PRUNE_THRESHOLD (0.05).
    """
    super_coll = ChromaClient.get_super_nodes_collection()
    if super_coll is None:
        logger.warning("decay_scorer: super_nodes collection not available yet.")
        return

    try:
        all_sn = super_coll.get(
            where={"type": {"$eq": "super_node"}},
            include=["metadatas"]
        )
    except Exception as e:
        logger.error("decay_scorer: failed to get super-nodes: %s", e)
        return

    ids = all_sn.get("ids", [])
    metadatas = all_sn.get("metadatas", [])

    logger.info("decay_scorer: processing decay for %d super-nodes...", len(ids))

    for sn_id, meta in zip(ids, metadatas):
        if not meta:
            continue

        # SU10: Pull true access count from SQLite (atomic source of truth)
        try:
            sqlite_row = sqlite_client.get_cluster_by_super_node_id(sn_id)
            true_access_count = sqlite_row["hit_count"] if sqlite_row else int(meta.get("hit_count", 0))
        except Exception as e:
            logger.warning("decay_scorer: failed to fetch hit_count from SQLite for sn_id=%s: %s", sn_id, e)
            true_access_count = int(meta.get("hit_count", 0))

        # Inject SQLite count for decay computation
        meta_with_sqlite = {**meta, "access_count_sqlite": true_access_count}
        
        half_life = getattr(config, "HALF_LIFE_DAYS", 7.0)
        score = compute_exponential_decay(meta_with_sqlite, half_life_days=half_life)

        # Sync count and score back to ChromaDB metadata
        updated_meta = {
            **meta,
            "decay_score": score,
            "access_count": true_access_count,
            "hit_count": true_access_count
        }
        
        try:
            super_node_store.update_metadata(sn_id, updated_meta)
        except Exception as e:
            logger.warning("decay_scorer: failed to update metadata in ChromaDB for sn_id=%s: %s", sn_id, e)
            continue

        prune_threshold = getattr(config, "DECAY_PRUNE_THRESHOLD", 0.05)
        if score < prune_threshold:
            try:
                super_coll.delete(ids=[sn_id])
                # Reset cluster in SQLite so it can be re-synthesized in the future
                sqlite_client.reset_cluster_after_pruning(sn_id)
                # Log pruning event in SQLite
                sqlite_client.log_maintenance_event(
                    event_type="prune",
                    super_node_id=sn_id,
                    reason=f"decay_score={score:.4f} < threshold={prune_threshold:.2f}",
                    decay_score=score
                )
                
                # Calculate days since last access for the log message
                last_accessed_str = meta.get("last_accessed") or meta.get("created_at")
                days = 0
                if last_accessed_str:
                    last_accessed = datetime.fromisoformat(last_accessed_str)
                    now = datetime.now(timezone.utc) if last_accessed.tzinfo is not None else datetime.utcnow()
                    days = int((now - last_accessed).days)
                
                logger.info("decay_scorer: node_pruned sn_id=%s | score=%.4f | days_since_access=%d", sn_id, score, days)
            except Exception as e:
                logger.error("decay_scorer: failed to prune super-node sn_id=%s: %s", sn_id, e)
