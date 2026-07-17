"""
app/retrieval/router.py
------------------------
Phase 5 -- Thin collection-query wrappers for the retrieval pipeline.

Responsibilities
----------------
* ``query_super_nodes``: query the super_nodes ChromaDB collection and
  return results in a normalised dict format.  Handles the graceful
  fallback when the collection is missing or empty (blueprint pitfall #3:
  SU5 masking may reduce effective super-node top_k below top_k -- the
  raw_chunk fill from this module compensates automatically).

* ``query_raw_chunks``: query the raw_chunks collection in the same
  normalised format so the retriever can merge both lists without
  special-casing.

Both functions return results with key "id" (the ChromaDB document ID).
This ensures the retriever returns a uniform "id" key, fixing the
pre-existing mismatch where retriever.py returned "chunk_id" while
query.py read res.get("id", "unknown").
"""

import logging
from typing import List, Dict, Any

from app.db.chroma_client import ChromaClient

logger = logging.getLogger(__name__)


def query_super_nodes(q_emb: list, top_k: int) -> List[Dict[str, Any]]:
    """
    Query the super_nodes ChromaDB collection.

    Graceful fallback (blueprint pitfall #3 mitigation):
    - If the collection does not exist yet (Phase 3/4 not run), returns [].
    - If the collection exists but is empty, returns [].
    - If fewer items exist than top_k, n_results is clamped to the count.

    Each result dict contains:
        id       (str)   ChromaDB document ID
        doc      (str)   document text
        meta     (dict)  ChromaDB metadata
        distance (float) cosine distance (pre-boost)
    """
    col = ChromaClient.get_super_nodes_collection()
    if col is None:
        logger.debug("router: super_nodes collection unavailable -- fallback to raw_chunks only")
        return []

    count = col.count()
    if count == 0:
        logger.debug("router: super_nodes collection is empty -- fallback to raw_chunks only")
        return []

    # Clamp n_results so ChromaDB does not raise when count < top_k
    n_results = min(top_k, count)

    raw = col.query(
        query_embeddings=[q_emb],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    results: List[Dict[str, Any]] = []
    for id_, doc, meta, dist in zip(
        raw["ids"][0],
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        results.append({"id": id_, "doc": doc, "meta": meta, "distance": dist})

    logger.debug("router: super_nodes returned %d candidates", len(results))
    return results


def query_raw_chunks(q_emb: list, top_k: int) -> List[Dict[str, Any]]:
    """
    Query the raw_chunks ChromaDB collection.

    Always returns results (collection is guaranteed to exist after Phase 1).

    Each result dict contains:
        id       (str)   ChromaDB document ID
        doc      (str)   document text
        meta     (dict)  ChromaDB metadata
        distance (float) cosine distance
    """
    col = ChromaClient.get_collection()
    raw = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    results: List[Dict[str, Any]] = []
    for id_, doc, meta, dist in zip(
        raw["ids"][0],
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        results.append({"id": id_, "doc": doc, "meta": meta, "distance": dist})

    logger.debug("router: raw_chunks returned %d candidates", len(results))
    return results
