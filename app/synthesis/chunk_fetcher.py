"""
app/synthesis/chunk_fetcher.py
--------------------------------
Fetches raw chunk texts + embeddings from the ChromaDB `raw_chunks`
collection and applies SU1 Semantic Deduplication before synthesis.

Why deduplication is necessary
--------------------------------
Phase 2 (query_logger) merges chunk_ids across every query hit into the
cluster using a set-union. Over time a cluster can accumulate dozens of
chunk_ids, many of which are near-identical paragraphs scraped from
different mirrors of the same NIH/MedQuAD page.

Feeding all of them to the LLM:
  (a) wastes the token budget
  (b) causes the model to repeat the same facts in the summary
  (c) in extreme cases, collapses the context window entirely

SU1 — Semantic Deduplication
-------------------------------
Algorithm:
  1. Fetch documents + embeddings for chunk_ids from raw_chunks.
  2. If len(docs) <= 5:  skip clustering; concatenate + truncate.
  3. AgglomerativeClustering(
         n_clusters       = None,
         distance_threshold = 0.15,   # cosine distance; ~cos_sim >= 0.85
         metric           = 'cosine',
         linkage          = 'average'
     )
  4. Pick one representative per cluster (the first one encountered).
  5. Join with "\n\n---\n\n" and hard-truncate to max_tokens * 4 chars.

scikit-learn >= 1.0 is required for metric='cosine' in AgglomerativeClustering.
"""

import logging
from typing import List, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from app.db.chroma_client import ChromaClient

logger = logging.getLogger(__name__)

# Separator used between deduplicated source passages in the context block.
_PASSAGE_SEP = "\n\n---\n\n"

# Minimum number of chunks before deduplication clustering is applied.
_DEDUP_MIN_CHUNKS = 5

# Cosine distance threshold for AgglomerativeClustering.
# distance = 1 - cosine_similarity; 0.15 ≈ cos_sim >= 0.85
_DISTANCE_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_from_chroma(chunk_ids: List[str]) -> Tuple[List[str], List[np.ndarray]]:
    """
    Fetches documents and embeddings for the given chunk_ids from ChromaDB
    raw_chunks collection.

    Returns
    -------
    docs : List[str]
        The raw text of each chunk, in the same order as the returned ids.
    embeddings : List[np.ndarray]
        Corresponding embedding vectors as numpy arrays.

    Notes
    -----
    - Silently skips any chunk_id not found in ChromaDB (deleted after ingestion).
    - If fewer than 1 doc is returned, returns empty lists — caller handles this.
    """
    if not chunk_ids:
        return [], []

    collection = ChromaClient.get_collection()

    try:
        result = collection.get(
            ids=chunk_ids,
            include=["documents", "embeddings"],
        )
    except Exception as exc:
        logger.error("chunk_fetcher: ChromaDB get failed: %s", exc)
        return [], []

    raw_ids   = result.get("ids", [])
    raw_docs  = result.get("documents", [])
    raw_embs  = result.get("embeddings", [])

    # Filter out any None entries (ChromaDB returns None for missing IDs)
    docs: List[str] = []
    embeddings: List[np.ndarray] = []

    for doc, emb in zip(raw_docs, raw_embs):
        if doc is not None and emb is not None:
            docs.append(doc)
            embeddings.append(np.array(emb, dtype=np.float32))

    logger.debug(
        "chunk_fetcher: fetched %d / %d requested chunks from ChromaDB",
        len(docs),
        len(chunk_ids),
    )
    return docs, embeddings


def _deduplicate(
    docs: List[str],
    embeddings: List[np.ndarray],
) -> Tuple[List[str], List[np.ndarray]]:
    """
    SU1 — Cluster docs by semantic similarity and pick one representative
    per cluster. Chunks with cosine distance < 0.15 (cos_sim > 0.85) are
    treated as duplicates.

    Parameters
    ----------
    docs : List[str]
    embeddings : List[np.ndarray]

    Returns
    -------
    Deduplicated (docs, embeddings) in cluster order.
    """
    n = len(docs)

    if n <= _DEDUP_MIN_CHUNKS:
        logger.debug(
            "chunk_fetcher: %d chunks <= threshold (%d); skipping dedup",
            n, _DEDUP_MIN_CHUNKS,
        )
        return docs, embeddings

    matrix = np.stack(embeddings)  # shape: (n, dim)

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=_DISTANCE_THRESHOLD,
        metric="cosine",
        linkage="average",
    )
    labels = clusterer.fit_predict(matrix)

    # Pick first-encountered doc per unique cluster label
    seen_clusters: set = set()
    dedup_docs:   List[str]        = []
    dedup_embs:   List[np.ndarray] = []

    for label, doc, emb in zip(labels, docs, embeddings):
        if label not in seen_clusters:
            seen_clusters.add(label)
            dedup_docs.append(doc)
            dedup_embs.append(emb)

    logger.info(
        "chunk_fetcher: SU1 dedup — %d chunks → %d representatives",
        n,
        len(dedup_docs),
    )
    return dedup_docs, dedup_embs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_and_deduplicate(
    chunk_ids: List[str],
    max_tokens: int = 4000,
) -> Tuple[str, List[np.ndarray]]:
    """
    Fetch raw chunks from ChromaDB, apply SU1 semantic deduplication,
    and return a single source-text block ready for the synthesis prompt.

    Parameters
    ----------
    chunk_ids : List[str]
        Raw doc_* chunk IDs from the SynthesisJob. sn_* IDs are never
        present here — already filtered by SU9 in query_interceptor.
    max_tokens : int
        Approximate token budget for the synthesizer. The returned text is
        hard-truncated to max_tokens * 4 characters as a safety net
        (1 token ≈ 4 chars for English medical text).

    Returns
    -------
    source_text : str
        Deduplicated passages joined by "\\n\\n---\\n\\n".
        Empty string if ChromaDB returns no results.
    source_embeddings : List[np.ndarray]
        Embedding vectors of the selected representative chunks.
        Used by synthesizer.check_fidelity_bound() (SU14).

    Notes
    -----
    - Skips deduplication when ≤ 5 chunks are present.
    - SU14 edge case: if fewer than 2 embeddings are returned the caller
      (synthesizer.py) skips the fidelity bound check and logs a warning.
    """
    docs, embeddings = _fetch_from_chroma(chunk_ids)

    if not docs:
        logger.warning(
            "chunk_fetcher: no documents returned from ChromaDB for %d chunk_ids",
            len(chunk_ids),
        )
        return "", []

    docs, embeddings = _deduplicate(docs, embeddings)

    source_text = _PASSAGE_SEP.join(docs)

    # Hard character-level truncation as a final safety net
    char_budget = max_tokens * 4
    if len(source_text) > char_budget:
        source_text = source_text[:char_budget]
        logger.debug(
            "chunk_fetcher: source_text truncated to %d chars (budget=%d tokens)",
            char_budget, max_tokens,
        )

    logger.info(
        "chunk_fetcher: final source_text — %d chars, %d passages",
        len(source_text),
        len(docs),
    )
    return source_text, embeddings
