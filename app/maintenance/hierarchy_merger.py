import json
import logging
from datetime import datetime, timezone
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app import config
from app.db import sqlite_client
from app.db.chroma_client import ChromaClient
from app.db import super_node_store
from app.synthesis.synthesizer import SynthesisJob, run as run_synthesis

logger = logging.getLogger(__name__)

def merge_similar_nodes() -> None:
    """
    SU7 + SU11 — Hierarchy Merger with Ground-Truth Re-Synthesis.
    - Fetch super-nodes that are not already in a hierarchy.
    - Find pairs with cosine similarity > config.HIERARCHY_MERGE_THRESHOLD (0.88).
    - Union their source raw chunk IDs and run the synthesis pipeline (SU11).
    - Mark child nodes in_hierarchy = True and store parent_meta_id (SU7).
    """
    super_coll = ChromaClient.get_super_nodes_collection()
    if super_coll is None:
        logger.warning("hierarchy_merger: super_nodes collection not available yet.")
        return

    try:
        # SU7: Fetch super-nodes that are not already merged
        all_sn = super_coll.get(
            where={
                "$and": [
                    {"type": {"$eq": "super_node"}},
                    {"in_hierarchy": {"$eq": False}}
                ]
            },
            include=["metadatas", "embeddings"]
        )
    except Exception as e:
        logger.error("hierarchy_merger: failed to get eligible super-nodes: %s", e)
        return

    ids = all_sn.get("ids", [])
    metadatas = all_sn.get("metadatas", [])
    embeddings = all_sn.get("embeddings", [])

    if len(ids) < 2:
        logger.info("hierarchy_merger: not enough eligible super-nodes to merge (%d found).", len(ids))
        return

    # Filter and construct list of active nodes
    nodes = []
    for sn_id, meta, emb in zip(ids, metadatas, embeddings):
        if meta and meta.get("type") == "super_node" and not meta.get("in_hierarchy", False):
            nodes.append({
                "id": sn_id,
                "metadata": meta,
                "embedding": emb
            })

    # Limit to top 500 sorted by hit_count to avoid O(N^2) pairwise comparison bloat
    nodes.sort(key=lambda x: int(x["metadata"].get("hit_count", 0)), reverse=True)
    nodes = nodes[:500]

    if len(nodes) < 2:
        return

    logger.info("hierarchy_merger: computing pairwise similarity for %d super-nodes...", len(nodes))

    embs = np.array([n["embedding"] for n in nodes])
    sim_matrix = cosine_similarity(embs)
    np.fill_diagonal(sim_matrix, 0.0)

    merge_threshold = getattr(config, "HIERARCHY_MERGE_THRESHOLD", 0.88)
    pairs = np.argwhere(sim_matrix > merge_threshold)
    pairs = [(a, b) for a, b in pairs if a < b]

    logger.info("hierarchy_merger: found %d pair(s) exceeding similarity threshold %s.", len(pairs), merge_threshold)

    merged_ids = set()

    for idx_a, idx_b in pairs:
        id_a = nodes[idx_a]["id"]
        id_b = nodes[idx_b]["id"]

        # Prevent double-merging in a single run (ensure children are tree-structured)
        if id_a in merged_ids or id_b in merged_ids:
            continue

        meta_a = nodes[idx_a]["metadata"]
        meta_b = nodes[idx_b]["metadata"]

        # SU11: Enforce MAX_LINEAGE_DEPTH cap
        depth_a = int(meta_a.get("lineage_depth", 0))
        depth_b = int(meta_b.get("lineage_depth", 0))
        merged_depth = max(depth_a, depth_b) + 1

        max_depth = getattr(config, "MAX_LINEAGE_DEPTH", 2)
        if merged_depth > max_depth:
            logger.warning(
                "hierarchy_merger: merger_depth_cap_reached sn_a=%s | sn_b=%s | merged_depth=%d | cap=%d",
                id_a, id_b, merged_depth, max_depth
            )
            # Route to manual review instead of auto-merging
            try:
                sqlite_client.flag_for_manual_review(
                    id_a, id_b,
                    reason=f"lineage_depth_cap: would reach depth {merged_depth} > cap {max_depth}"
                )
                sqlite_client.log_maintenance_event(
                    event_type="depth_cap_refused",
                    super_node_id=id_a,
                    reason=f"Hierarchy merge depth limit reached ({merged_depth} > {max_depth}) for pair {id_a} and {id_b}"
                )
            except Exception as e:
                logger.error("hierarchy_merger: failed to write manual review flags: %s", e)
            continue

        # SU11: Union source_chunks (always raw doc_ IDs)
        try:
            chunks_a = json.loads(meta_a.get("source_chunks", "[]"))
            chunks_b = json.loads(meta_b.get("source_chunks", "[]"))
        except Exception as e:
            logger.warning("hierarchy_merger: failed to parse children source chunks: %s", e)
            continue

        unioned_raw_ids = list(set(chunks_a + chunks_b))

        # Defensive check: Assert only raw doc_ chunk IDs exist in the union
        non_raw = [cid for cid in unioned_raw_ids if not cid.startswith("doc_")]
        if non_raw:
            logger.error("hierarchy_merger: Super-node IDs detected in source_chunks union: %s — SU9 violated!", non_raw)
            # Filter them out to be safe
            unioned_raw_ids = [cid for cid in unioned_raw_ids if cid.startswith("doc_")]

        if not unioned_raw_ids:
            logger.warning("hierarchy_merger: union of source chunks is empty. Skipping pair.")
            continue

        # SU11: Run full Phase 4 synthesis on raw unioned IDs
        merge_query = f"Synthesis of: {meta_a.get('source_query', 'N/A')} AND {meta_b.get('source_query', 'N/A')}"
        hit_sum = int(meta_a.get("hit_count", 0)) + int(meta_b.get("hit_count", 0))
        
        job = SynthesisJob(
            cluster_id=None,  # meta-nodes have no cluster association in SQLite
            canonical_query=merge_query,
            chunk_ids=unioned_raw_ids,
            hit_count=hit_sum,
            triggered_at=datetime.now(timezone.utc),
            existing_sn_id=None
        )

        logger.info("hierarchy_merger: synthesizing meta-node for children %s and %s...", id_a, id_b)

        try:
            meta_id = run_synthesis(job)
        except Exception as e:
            logger.error("hierarchy_merger: merger_synthesis_failed sn_a=%s | sn_b=%s | error=%s", id_a, id_b, e)
            continue

        # Fetch synthesized meta-node and update its type and children links
        try:
            merged_meta = super_node_store.get_metadata(meta_id)
            updated_meta = {
                **merged_meta,
                "type": "meta_node",
                "children": json.dumps([id_a, id_b]),
                "lineage_depth": merged_depth,
                "source_chunks": json.dumps(unioned_raw_ids),
                "in_hierarchy": False
            }
            super_node_store.update_metadata(meta_id, updated_meta)

            # SU7: Lock child nodes out of future merger passes and record parent meta node link
            for child_id, child_meta in [(id_a, meta_a), (id_b, meta_b)]:
                updated_child = {
                    **child_meta,
                    "in_hierarchy": True,
                    "parent_meta_id": meta_id
                }
                super_node_store.update_metadata(child_id, updated_child)

            # Record successfully processed merge
            merged_ids.update([id_a, id_b])
            sqlite_client.log_maintenance_event(
                event_type="merge",
                super_node_id=meta_id,
                reason=f"meta_node_created from children {[id_a, id_b]} at depth {merged_depth}"
            )
            logger.info(
                "hierarchy_merger: meta_node_created meta_id=%s | children=%s | lineage_depth=%d | raw_chunks=%d",
                meta_id, [id_a, id_b], merged_depth, len(unioned_raw_ids)
            )

        except Exception as e:
            logger.error("hierarchy_merger: failed to finalise meta-node metadata for meta_id=%s: %s", meta_id, e)
            continue
