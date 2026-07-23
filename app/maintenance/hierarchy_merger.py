import json
import logging
from datetime import datetime, timezone
import numpy as np

from app import config
from app.db import sqlite_client
from app.db.chroma_client import ChromaClient
from app.db import super_node_store
from app.synthesis.synthesizer import SynthesisJob, run as run_synthesis
from app.synthesis.prompts import META_NODE_SYSTEM_PROMPT, META_NODE_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ANN helpers — used when node count exceeds ANN_THRESHOLD to avoid O(N²)
# ---------------------------------------------------------------------------

ANN_THRESHOLD = 10_000   # Switch from pairwise to ANN above this count


def _find_pairs_pairwise(nodes: list, threshold: float) -> list[tuple[int, int]]:
    """Full O(N²) cosine similarity — safe up to ANN_THRESHOLD nodes."""
    from sklearn.metrics.pairwise import cosine_similarity
    embs = np.array([n["embedding"] for n in nodes])
    sim_matrix = cosine_similarity(embs)
    np.fill_diagonal(sim_matrix, 0.0)
    raw_pairs = np.argwhere(sim_matrix > threshold)
    return [(int(a), int(b)) for a, b in raw_pairs if a < b]


def _find_pairs_ann(nodes: list, threshold: float) -> list[tuple[int, int]]:
    """
    Approximate Nearest Neighbour search — O(N log N) using ball_tree.
    Used when len(nodes) > ANN_THRESHOLD to keep the merger tractable.

    For each node we query its K=50 nearest neighbours and keep only those
    whose cosine similarity (= 1 - distance in ball_tree cosine metric)
    exceeds the threshold.
    """
    from sklearn.neighbors import NearestNeighbors

    embs = np.array([n["embedding"] for n in nodes], dtype=np.float32)
    k = min(51, len(nodes))   # +1 because the query itself is returned at rank-0

    nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", metric="cosine")
    nn.fit(embs)
    distances, indices = nn.kneighbors(embs)

    # cosine distance = 1 − cosine_similarity  →  similarity = 1 − distance
    pairs = set()
    for i, (dists, nbrs) in enumerate(zip(distances, indices)):
        for dist, j in zip(dists, nbrs):
            if i == j:
                continue
            sim = 1.0 - float(dist)
            if sim > threshold:
                pair = (min(i, j), max(i, j))
                pairs.add(pair)
    return list(pairs)


# ---------------------------------------------------------------------------
# Main merger function
# ---------------------------------------------------------------------------

def merge_similar_nodes() -> None:
    """
    SU7 + SU11 — Hierarchy Merger with Ground-Truth Re-Synthesis.

    Hierarchical structure produced:
      Disease (meta_node, depth=1)
        ├── Symptoms super-node   (in_hierarchy=True, depth=0)
        ├── Treatments super-node (in_hierarchy=True, depth=0)
        └── Causes super-node    (in_hierarchy=True, depth=0)

    Changes from baseline:
    - Uses META_NODE_PROMPT_TEMPLATE so merged content is formatted as
      a structured disease encyclopaedia (Overview / Symptoms / Causes /
      Effects / Treatments / Related Information).
    - ANN fallback for large collections (> ANN_THRESHOLD nodes).
    - MAX_LINEAGE_DEPTH defaults to 3 (was 2).
    - HIERARCHY_MERGE_THRESHOLD defaults to 0.75 (was 0.88).
    """
    super_coll = ChromaClient.get_super_nodes_collection()
    if super_coll is None:
        logger.warning("hierarchy_merger: super_nodes collection not available yet.")
        return

    try:
        # SU7: Fetch super-nodes not yet in a hierarchy
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

    ids        = all_sn.get("ids", [])
    metadatas  = all_sn.get("metadatas", [])
    embeddings = all_sn.get("embeddings", [])

    if len(ids) < 2:
        logger.info(
            "hierarchy_merger: not enough eligible super-nodes to merge (%d found).",
            len(ids)
        )
        return

    # Build node list (Python-side filter for ChromaDB boolean compat)
    nodes = []
    for sn_id, meta, emb in zip(ids, metadatas, embeddings):
        if meta and meta.get("type") == "super_node" and not meta.get("in_hierarchy", False):
            nodes.append({"id": sn_id, "metadata": meta, "embedding": emb})

    # Sort by hit_count descending and cap for efficiency
    nodes.sort(key=lambda x: int(x["metadata"].get("hit_count", 0)), reverse=True)
    nodes = nodes[:500]       # hard cap: always limit candidate pool

    if len(nodes) < 2:
        return

    merge_threshold = getattr(config, "HIERARCHY_MERGE_THRESHOLD", 0.75)
    max_depth       = getattr(config, "MAX_LINEAGE_DEPTH", 3)

    # Select pairwise strategy based on collection size
    if len(nodes) > ANN_THRESHOLD:
        logger.info(
            "hierarchy_merger: %d nodes exceeds ANN_THRESHOLD=%d — using ANN search.",
            len(nodes), ANN_THRESHOLD
        )
        pairs = _find_pairs_ann(nodes, merge_threshold)
    else:
        logger.info(
            "hierarchy_merger: computing pairwise similarity for %d super-nodes...",
            len(nodes)
        )
        pairs = _find_pairs_pairwise(nodes, merge_threshold)

    logger.info(
        "hierarchy_merger: found %d pair(s) exceeding similarity threshold %.2f.",
        len(pairs), merge_threshold
    )

    merged_ids: set[str] = set()

    for idx_a, idx_b in pairs:
        id_a = nodes[idx_a]["id"]
        id_b = nodes[idx_b]["id"]

        # Prevent double-merging (tree-structure safety)
        if id_a in merged_ids or id_b in merged_ids:
            continue

        meta_a = nodes[idx_a]["metadata"]
        meta_b = nodes[idx_b]["metadata"]

        # SU11: Enforce MAX_LINEAGE_DEPTH
        depth_a      = int(meta_a.get("lineage_depth", 0))
        depth_b      = int(meta_b.get("lineage_depth", 0))
        merged_depth = max(depth_a, depth_b) + 1

        if merged_depth > max_depth:
            logger.warning(
                "hierarchy_merger: depth_cap_reached sn_a=%s | sn_b=%s | depth=%d | cap=%d",
                id_a, id_b, merged_depth, max_depth
            )
            try:
                sqlite_client.flag_for_manual_review(
                    id_a, id_b,
                    reason=f"lineage_depth_cap: would reach depth {merged_depth} > cap {max_depth}"
                )
                sqlite_client.log_maintenance_event(
                    event_type="depth_cap_refused",
                    super_node_id=id_a,
                    reason=(
                        f"Hierarchy merge depth limit reached "
                        f"({merged_depth} > {max_depth}) for pair {id_a} and {id_b}"
                    )
                )
            except Exception as e:
                logger.error("hierarchy_merger: failed to write depth_cap flags: %s", e)
            continue

        # SU11: Union raw source chunk IDs
        try:
            chunks_a = json.loads(meta_a.get("source_chunks", "[]"))
            chunks_b = json.loads(meta_b.get("source_chunks", "[]"))
        except Exception as e:
            logger.warning("hierarchy_merger: failed to parse source_chunks: %s", e)
            continue

        unioned_raw_ids = list(set(chunks_a + chunks_b))

        # SU9 guard — strip any accidentally included super-node IDs
        non_raw = [cid for cid in unioned_raw_ids if cid.startswith("sn_")]
        if non_raw:
            logger.error(
                "hierarchy_merger: super-node IDs in union %s — stripping (SU9 guard).",
                non_raw
            )
            unioned_raw_ids = [cid for cid in unioned_raw_ids if not cid.startswith("sn_")]

        if not unioned_raw_ids:
            logger.warning("hierarchy_merger: empty chunk union for pair. Skipping.")
            continue

        # Structured merge query for the meta-node
        query_a = meta_a.get("source_query", "Unknown topic A")
        query_b = meta_b.get("source_query", "Unknown topic B")
        
        # Ask LLM to generate a short common name for the merged node
        from app.generation.llm_client import llm_call
        title_prompt = (
            f"You are a medical categorization system.\n"
            f"Extract a very short, common overarching topic name (maximum 2 to 4 words) that encompasses both of these topics:\n"
            f"Topic A: {query_a}\n"
            f"Topic B: {query_b}\n\n"
            f"Output ONLY the raw string name, no quotes, no extra text. Examples: 'Lisinopril', 'Water Intake', 'Type 2 Diabetes'."
        )
        try:
            merge_query = llm_call(
                system_prompt="You extract short common names for medical topics.",
                user_prompt=title_prompt,
                temperature=0.1,
                max_tokens=20
            ).strip().strip("'\"")
        except Exception as e:
            logger.warning("hierarchy_merger: failed to generate title with LLM, falling back to concat: %s", e)
            merge_query = f"Medical encyclopaedia entry combining: {query_a} | {query_b}"
            
        hit_sum     = int(meta_a.get("hit_count", 0)) + int(meta_b.get("hit_count", 0))

        # SU11: Run full synthesis using META_NODE_PROMPT_TEMPLATE (structured headings)
        job = SynthesisJob(
            cluster_id=None,
            canonical_query=merge_query,
            chunk_ids=unioned_raw_ids,
            hit_count=hit_sum,
            triggered_at=datetime.now(timezone.utc),
            existing_sn_id=None,
            # Inject structured disease prompt
            prompt_template=META_NODE_PROMPT_TEMPLATE,
            prompt_system_override=META_NODE_SYSTEM_PROMPT,
            child_queries=[query_a, query_b],
        )

        logger.info(
            "hierarchy_merger: synthesizing structured meta-node for %s + %s ...",
            id_a, id_b
        )

        try:
            meta_id = run_synthesis(job)
        except Exception as e:
            logger.error(
                "hierarchy_merger: synthesis failed sn_a=%s | sn_b=%s | error=%s",
                id_a, id_b, e
            )
            continue

        # Update meta-node metadata to declare hierarchy role
        try:
            merged_meta = super_node_store.get_metadata(meta_id)
            super_node_store.update_metadata(meta_id, {
                **merged_meta,
                "type":          "meta_node",
                "children":      json.dumps([id_a, id_b]),
                "lineage_depth": merged_depth,
                "source_chunks": json.dumps(unioned_raw_ids),
                "in_hierarchy":  False,   # meta-node itself is available for higher merges
            })

            # SU7: Lock children to prevent re-merging
            for child_id, child_meta in [(id_a, meta_a), (id_b, meta_b)]:
                super_node_store.update_metadata(child_id, {
                    **child_meta,
                    "in_hierarchy":  True,
                    "parent_meta_id": meta_id,
                })

            merged_ids.update([id_a, id_b])
            sqlite_client.log_maintenance_event(
                event_type="merge",
                super_node_id=meta_id,
                reason=(
                    f"meta_node_created from children [{id_a}, {id_b}] "
                    f"at depth {merged_depth} | threshold={merge_threshold}"
                )
            )
            logger.info(
                "hierarchy_merger: meta_node_created meta_id=%s | children=[%s, %s] "
                "| depth=%d | chunks=%d",
                meta_id, id_a, id_b, merged_depth, len(unioned_raw_ids)
            )

        except Exception as e:
            logger.error(
                "hierarchy_merger: failed to finalise meta-node %s: %s", meta_id, e
            )
