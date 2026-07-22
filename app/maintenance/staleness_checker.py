import json
import logging
from app.db import sqlite_client
from app.db.chroma_client import ChromaClient
from app.db import super_node_store

logger = logging.getLogger(__name__)

def run_soft_staleness_check() -> None:
    """
    SU8 — Soft-Staleness Check.
    - Fetch all super-nodes from ChromaDB.
    - Verify how many original IDs still exist in the raw collection.
    - If match_ratio < 1.0, mark is_stale=True, and queue re-synthesis in SQLite.
    """
    super_coll = ChromaClient.get_super_nodes_collection()
    if super_coll is None:
        logger.warning("staleness_checker: super_nodes collection not available yet.")
        return

    raw_coll = ChromaClient.get_collection()
    if raw_coll is None:
        logger.warning("staleness_checker: raw_chunks collection not available.")
        return

    try:
        all_sn = super_coll.get(
            where={"type": {"$eq": "super_node"}},
            include=["metadatas"]
        )
    except Exception as e:
        logger.error("staleness_checker: failed to get super-nodes: %s", e)
        return

    ids = all_sn.get("ids", [])
    metadatas = all_sn.get("metadatas", [])

    logger.info("staleness_checker: checking %d super-nodes...", len(ids))

    for sn_id, meta in zip(ids, metadatas):
        if not meta or "source_chunks" not in meta:
            continue

        try:
            source_chunk_ids = json.loads(meta["source_chunks"])
        except Exception as e:
            logger.warning("staleness_checker: failed to parse source_chunks for sn_id=%s: %s", sn_id, e)
            continue

        if not source_chunk_ids:
            continue

        try:
            existing = raw_coll.get(
                ids=source_chunk_ids,
                include=["ids"]
            )
            existing_ids = existing.get("ids", [])
            match_ratio = len(existing_ids) / len(source_chunk_ids)
        except Exception as e:
            logger.warning("staleness_checker: staleness_check_error for sn_id=%s: %s", sn_id, e)
            continue

        if match_ratio < 1.0:
            # Soft flag: mark stale
            updated_meta = {**meta, "is_stale": True}
            super_node_store.update_metadata(sn_id, updated_meta)

            # Queue background re-synthesis in SQLite
            cluster_id = meta.get("cluster_id")
            if cluster_id is not None:
                try:
                    sqlite_client.flag_cluster_for_re_synthesis(int(cluster_id))
                    sqlite_client.log_maintenance_event(
                        event_type="soft_stale",
                        super_node_id=sn_id,
                        reason=f"source_chunk_drift: match_ratio={match_ratio:.4f}"
                    )
                    logger.info(
                        "staleness_checker: super_node_marked_stale sn_id=%s | match_ratio=%.4f | cluster_id=%s",
                        sn_id, match_ratio, cluster_id
                    )
                except Exception as e:
                    logger.error(
                        "staleness_checker: failed to flag cluster/log event for sn_id=%s: %s",
                        sn_id, e
                    )
