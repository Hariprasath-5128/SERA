"""
app/logging_/query_logger.py
-----------------------------
Core query-clustering engine — Phase 2.

Entry point
-----------
    log_query(raw_query, retrieved_chunk_ids) -> cluster_id: int

What happens on every call
--------------------------
1. Normalise + embed the incoming query.
2. Load all existing cluster centroids from SQLite.
3. Compute batch cosine similarity of the new embedding vs all centroids.
4. If any cluster exceeds COSINE_THRESHOLD (0.92):
       a. SU6 — update the centroid BEFORE incrementing hit_count (so `n`
          in the running average formula is the pre-increment count).
       b. Merge chunk_ids (set-union of existing + incoming raw IDs).
       c. Increment hit_count and refresh last_hit.
       d. Write query_log row (matched_existing=True).
5. If no cluster matches:
       a. INSERT new cluster row.
       b. Write query_log row (matched_existing=False).
6. Return the cluster_id.

SU6 — Running Weighted Average centroid update
----------------------------------------------
    new_centroid = (old_centroid * n + new_emb) / (n + 1)

The centroid update happens BEFORE hit_count is incremented, so `n`
correctly equals the number of previous hits that shaped the centroid.

SU9 — Ground-Truth Anchoring
-----------------------------
chunk_ids passed into this module must already be filtered to doc_* IDs.
That filtering is enforced by QueryInterceptorMiddleware before this
function is called. No further filtering is applied here.

Scalability note
----------------
fetch_all_clusters() loads all centroid BLOBs into RAM for the cosine
scan. This is acceptable at < 10 000 clusters.  The function boundary is
designed so the SQLite backend can be swapped for FAISS without changing
any call site.
"""

import json
import logging
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app import config
from app.db import sqlite_client
from app.logging_ import normalizer

logger = logging.getLogger(__name__)

# Embedding dimensionality — read lazily on first call to avoid loading the
# model at import time.
_EMB_DIM: Optional[int] = None


def _get_dim() -> int:
    """Return the embedding dimensionality, loading the model if needed."""
    global _EMB_DIM
    if _EMB_DIM is None:
        # Trigger lazy load; encode a dummy string just to get shape.
        vec = normalizer.embed("probe")
        _EMB_DIM = vec.shape[0]
    return _EMB_DIM


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_query(
    raw_query: str,
    retrieved_chunk_ids: list[str],
) -> int:
    """
    Main entry point called by QueryInterceptorMiddleware after every
    successful POST /query response.

    Parameters
    ----------
    raw_query : str
        The original, unmodified query string from the request body.
    retrieved_chunk_ids : list[str]
        Raw doc_* chunk IDs from the retriever response.
        super-node (sn_*) IDs have already been stripped by middleware (SU9).

    Returns
    -------
    int
        The cluster_id this query was assigned to (new or existing).
    """
    # Step 1 — Normalise + embed
    new_emb = normalizer.embed(raw_query)
    canonical = normalizer.normalize(raw_query)

    # Step 2 — Load all existing cluster centroids
    clusters = sqlite_client.fetch_all_clusters()

    matched_cluster_id: Optional[int] = None

    if clusters:
        # Step 3 — Batch cosine similarity
        dim = _get_dim()
        centroids = []
        valid_cluster_ids = []

        for row in clusters:
            blob = row["query_embedding"]
            if not blob or len(blob) != dim * 4:
                # Corrupt or empty BLOB — skip this cluster gracefully
                logger.warning(
                    "query_logger: skipping cluster %d — corrupt embedding BLOB "
                    "(expected %d bytes, got %d)",
                    row["id"],
                    dim * 4,
                    len(blob) if blob else 0,
                )
                continue
            centroid = np.frombuffer(blob, dtype=np.float32)
            centroids.append(centroid)
            valid_cluster_ids.append(row["id"])

        if centroids:
            centroid_matrix = np.stack(centroids)          # (n_clusters, dim)
            sims = cosine_similarity(
                new_emb.reshape(1, -1), centroid_matrix
            )[0]                                           # (n_clusters,)

            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            if best_sim >= config.COSINE_THRESHOLD:
                matched_cluster_id = valid_cluster_ids[best_idx]

    if matched_cluster_id is not None:
        # ── Existing cluster: merge + update ──────────────────────────────
        cluster_row = sqlite_client.get_cluster(matched_cluster_id)
        existing_chunk_ids: list[str] = json.loads(cluster_row["chunk_ids"] or "[]")
        n = cluster_row["hit_count"]                       # count BEFORE this hit

        # SU6 — Running centroid update BEFORE incrementing hit_count
        old_centroid = np.frombuffer(
            cluster_row["query_embedding"], dtype=np.float32
        )
        new_centroid = (old_centroid * n + new_emb) / (n + 1)
        sqlite_client.update_cluster_embedding(matched_cluster_id, new_centroid)

        # Merge chunk_ids (set-union, preserves insertion order via dict)
        merged_chunk_ids = list(
            dict.fromkeys(existing_chunk_ids + retrieved_chunk_ids)
        )

        # Increment hit_count + refresh last_hit
        sqlite_client.update_cluster(matched_cluster_id, merged_chunk_ids)

        cluster_id = matched_cluster_id
        matched = True

        logger.debug(
            "query_logger: merged into cluster %d (sim=%.4f, hit_count=%d→%d)",
            cluster_id,
            best_sim,
            n,
            n + 1,
        )

    else:
        # ── New cluster ───────────────────────────────────────────────────
        cluster_id = sqlite_client.insert_cluster(
            canonical_query=canonical,
            embedding=new_emb,
            chunk_ids=retrieved_chunk_ids,
        )
        matched = False

        logger.debug(
            "query_logger: new cluster %d created for query='%s'",
            cluster_id,
            canonical[:60],
        )

    # Step 4 — Audit log
    sqlite_client.insert_query_log(
        raw_query=raw_query,
        cluster_id=cluster_id,
        retrieved_chunks=retrieved_chunk_ids,
        matched_existing=matched,
    )

    return cluster_id
