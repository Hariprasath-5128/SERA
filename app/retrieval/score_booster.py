"""
app/retrieval/score_booster.py
-------------------------------
SU4 - epsilon-Greedy Score Boost for Preferential Retrieval.

Encapsulates the single random draw that determines whether this query
runs in "exploitation" mode (super-nodes receive the 0.85 distance
multiplier, making them appear closer than raw chunks) or "exploration"
mode (no multiplier applied -- super-nodes and raw chunks compete purely
on semantic similarity).

Design rationale
----------------
Keeping this logic in its own module makes it trivially mockable in
tests: patch `app.retrieval.score_booster.random.random` to return a
deterministic value and the retriever's branching is fully controlled
without touching any I/O layer.

Distance metric note (mitigation for blueprint pitfall #1)
----------------------------------------------------------
Both ChromaDB collections (`raw_chunks` and `super_nodes`) are created
with ``hnsw:space=cosine``, so distances are on the same 0-2 cosine
scale.  The 0.85 multiplier therefore has consistent semantics across
both collections.  This module does NOT validate the distance space --
that guarantee lives in ``chroma_client.get_super_nodes_collection()``.
"""

import logging
import random

from app import config

logger = logging.getLogger(__name__)


def compute_boost(
    epsilon: float = config.EXPLORATION_EPSILON,
    multiplier: float = config.SUPER_NODE_SCORE_MULTIPLIER,
) -> tuple:
    """
    SU4 -- Draw the epsilon-greedy strategy for the current query.

    Algorithm::

        r = random.random()          # uniform in [0, 1)
        if r > epsilon  -> Exploitation: apply ``multiplier`` to super-node
                           distances, making them appear semantically closer.
        if r <= epsilon -> Exploration:  apply 1.0 (no change) so super-nodes
                           and raw chunks are ranked on raw cosine similarity.

    The mode label ("exploit" / "explore") is logged at INFO level so
    benchmark pipelines can stratify metrics by retrieval strategy
    (mitigation for blueprint pitfall #2: exploration skew in benchmarks).

    Args:
        epsilon:    Fraction of queries that use exploration mode.
                    Default: ``config.EXPLORATION_EPSILON`` (0.10).
        multiplier: Distance multiplier applied during exploitation.
                    Default: ``config.SUPER_NODE_SCORE_MULTIPLIER`` (0.85).

    Returns:
        Tuple of (active_multiplier: float, mode_label: str).
        ``active_multiplier`` is what the caller should multiply against
        each super-node cosine distance before merging with raw chunks.
    """
    r = random.random()
    if r > epsilon:
        logger.info(
            "retrieval_strategy mode=exploit epsilon=%.2f multiplier=%.2f r=%.4f",
            epsilon, multiplier, r,
        )
        return multiplier, "exploit"
    else:
        logger.info(
            "retrieval_strategy mode=explore epsilon=%.2f multiplier=1.0 r=%.4f",
            epsilon, r,
        )
        return 1.0, "explore"
