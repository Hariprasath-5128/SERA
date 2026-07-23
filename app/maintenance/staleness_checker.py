import json
import logging
from app.db import sqlite_client
from app.db.chroma_client import ChromaClient
from app.db import super_node_store

logger = logging.getLogger(__name__)


def _propagate_staleness_up(sn_id: str, visited: set) -> None:
    """
    SU13 — Upward propagation: walk parent_meta_id chain and mark each
    ancestor stale until we reach a node with no parent or already visited.
    """
    if sn_id in visited:
        return
    visited.add(sn_id)
    try:
        meta = super_node_store.get_metadata(sn_id)
        if not meta:
            return
        parent_id = meta.get("parent_meta_id", "")
        if not parent_id:
            return
        parent_meta = super_node_store.get_metadata(parent_id)
        if not parent_meta:
            return
        super_node_store.update_metadata(parent_id, {
            **parent_meta,
            "is_stale": True,
            "stale_reason": "child_source_drift",
        })
        logger.info(
            "staleness_checker: SU13 upward propagation — parent %s marked stale (child=%s)",
            parent_id, sn_id,
        )
        _propagate_staleness_up(parent_id, visited)
    except Exception as e:
        logger.warning("staleness_checker: upward propagation failed for sn_id=%s: %s", sn_id, e)


def _propagate_staleness_down(meta_id: str, visited: set) -> None:
    """
    SU13 — Downward propagation: mark all children of a meta-node stale
    so that when the parent is re-synthesised, children are also refreshed.
    """
    if meta_id in visited:
        return
    visited.add(meta_id)
    try:
        meta = super_node_store.get_metadata(meta_id)
        if not meta:
            return
        children = json.loads(meta.get("children", "[]"))
        for child_id in children:
            child_meta = super_node_store.get_metadata(child_id)
            if child_meta:
                super_node_store.update_metadata(child_id, {
                    **child_meta,
                    "is_stale": True,
                    "stale_reason": "parent_stale",
                })
                logger.info(
                    "staleness_checker: SU13 downward propagation — child %s marked stale (parent=%s)",
                    child_id, meta_id,
                )
                _propagate_staleness_down(child_id, visited)
    except Exception as e:
        logger.warning(
            "staleness_checker: downward propagation failed for meta_id=%s: %s", meta_id, e
        )


def run_soft_staleness_check() -> None:
    """
    SU8 — Soft-Staleness Check with SU13 bidirectional propagation.
    - Fetch all super-nodes from ChromaDB.
    - Verify how many original IDs still exist in the raw collection.
    - If match_ratio < 1.0:
        • Mark the node is_stale=True (SU8).
        • Queue re-synthesis in SQLite.
        • Propagate staleness UP to parent meta-nodes (SU13).
        • Propagate staleness DOWN to children of any stale meta-node (SU13).
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

    ids       = all_sn.get("ids", [])
    metadatas = all_sn.get("metadatas", [])

    logger.info("staleness_checker: checking %d super-nodes for source drift...", len(ids))

    visited_propagation: set = set()

    for sn_id, meta in zip(ids, metadatas):
        if not meta or "source_chunks" not in meta:
            continue

        try:
            source_chunk_ids = json.loads(meta["source_chunks"])
        except Exception as e:
            logger.warning(
                "staleness_checker: failed to parse source_chunks for sn_id=%s: %s", sn_id, e
            )
            continue

        if not source_chunk_ids:
            continue

        try:
            existing     = raw_coll.get(ids=source_chunk_ids, include=[])
            existing_ids = existing.get("ids", [])
            match_ratio  = len(existing_ids) / len(source_chunk_ids)
        except Exception as e:
            logger.warning(
                "staleness_checker: staleness_check_error for sn_id=%s: %s", sn_id, e
            )
            continue

        if match_ratio < 1.0:
            # SU8: Soft flag — mark stale but keep serving
            updated_meta = {**meta, "is_stale": True, "stale_reason": "source_chunk_drift"}
            super_node_store.update_metadata(sn_id, updated_meta)

            # Queue re-synthesis in SQLite
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
                        "staleness_checker: marked stale sn_id=%s | match_ratio=%.4f | cluster_id=%s",
                        sn_id, match_ratio, cluster_id
                    )
                except Exception as e:
                    logger.error(
                        "staleness_checker: failed to flag cluster for sn_id=%s: %s", sn_id, e
                    )

            # SU13: Propagate upward to parent meta-nodes
            _propagate_staleness_up(sn_id, visited_propagation)

            # SU13: If this node is a meta-node, propagate downward to children
            if meta.get("type") == "meta_node":
                _propagate_staleness_down(sn_id, visited_propagation)
