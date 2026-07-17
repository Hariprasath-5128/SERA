"""
app/retrieval/retriever.py
---------------------------
Phase 5 -- Preferential Retrieval.

Upgrades from Phase 1 single-collection lookup to a dual-collection
search with three embedded safety upgrades:

    SU4  epsilon-Greedy score boost  -- prevents super-node filter bubble
    SU5  Hierarchical masking        -- prevents parent/child context flooding
    SU10 Atomic SQLite access log    -- prevents concurrent counter loss

Key design decisions
--------------------
* Both ChromaDB collections use ``hnsw:space=cosine`` so distances are
  on the same 0-2 scale and the 0.85 multiplier has correct semantics
  (mitigation for blueprint pitfall #1 -- score space mismatch).

* Return schema uses key "id" (not "chunk_id") to match what query.py
  already reads via res.get("id", "unknown").  This fixes the
  pre-existing mismatch in the Phase 1 retriever while keeping the route
  and middleware unchanged.

* Graceful fallback: if super_nodes collection is absent or empty,
  only raw_chunk results are returned -- identical to Phase 1 behaviour.

* SU10 logging failure is caught and swallowed so an SQLite hiccup
  never propagates to the user's query response.
"""

import json
import logging
from typing import List, Dict, Any

from app.ingestion.embedder import Embedder
from app.retrieval.score_booster import compute_boost
from app.retrieval.router import query_super_nodes, query_raw_chunks
from app.db import sqlite_client
from app.config import DEFAULT_TOP_K, EXPLORATION_EPSILON

logger = logging.getLogger(__name__)


class Retriever:

    @classmethod
    def search(
        cls,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        epsilon: float = EXPLORATION_EPSILON,
    ) -> List[Dict[str, Any]]:
        """
        Phase 5 unified retrieval combining SU4, SU5, and SU10.

        Args:
            query:   Raw user query string.
            top_k:   Number of results to return.
            epsilon: Exploration rate for SU4 epsilon-greedy strategy.
                     Defaults to config.EXPLORATION_EPSILON (0.10).

        Returns:
            List of result dicts, sorted ascending by boosted cosine
            distance (lower = more similar).  Each dict has:
                id        str   ChromaDB document ID
                text      str   document text
                metadata  dict  ChromaDB metadata
                distance  float cosine distance (post-SU4 boost for super-nodes)
                type      str   "raw_chunk" | "super_node" | "meta_node"

        Note on "id" vs "chunk_id"
        ---------------------------
        The Phase 1 retriever returned "chunk_id".  Phase 5 returns "id"
        to match query.py's existing res.get("id", "unknown") call,
        correcting a pre-existing key mismatch without any route changes.
        The middleware (query_interceptor) reads from the JSON response
        body (source_chunks[].chunk_id via the SourceChunk Pydantic model)
        so it is unaffected by this internal key change.
        """
        if not query.strip():
            return []

        # ── 1. Embed query ────────────────────────────────────────────────────
        model = Embedder.get_model()
        q_emb: list = model.encode(
            [query], convert_to_numpy=True, show_progress_bar=False
        )[0].tolist()

        # ── 2. Dual-collection query ──────────────────────────────────────────
        sn_candidates = query_super_nodes(q_emb, top_k)
        raw_candidates = query_raw_chunks(q_emb, top_k)

        # ── 3. SU4 -- epsilon-Greedy strategy ─────────────────────────────────
        # mode label ("exploit"/"explore") is logged inside compute_boost so
        # benchmark pipelines can stratify metrics by retrieval strategy.
        multiplier, mode = compute_boost(epsilon=epsilon)

        # ── 4. SU5 -- Hierarchical masking ────────────────────────────────────
        # First pass: collect child IDs of any meta_node in the super-node
        # results.  These children will be excluded below to prevent the
        # context window from flooding with redundant parent+child content.
        masked_ids: set = set()
        for c in sn_candidates:
            if c["meta"].get("type") == "meta_node":
                try:
                    children = json.loads(c["meta"].get("children", "[]"))
                    masked_ids.update(children)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "retriever: could not parse children JSON for meta_node %s -- skipping mask",
                        c["id"],
                    )
        masked_count = 0

        # ── 5. Build merged candidate list ───────────────────────────────────
        candidates: List[Dict[str, Any]] = []

        # Super-node candidates: apply SU5 mask and SU4 distance boost.
        for c in sn_candidates:
            if c["id"] in masked_ids:
                # SU5: child node shadowed by its retrieved parent meta_node.
                # The vacated slot is automatically filled by raw_chunk
                # candidates when both pools are merged (blueprint pitfall #3).
                masked_count += 1
                logger.debug("retriever: SU5 masked child super-node %s", c["id"])
                continue

            node_type = c["meta"].get("type", "super_node")
            candidates.append({
                "id":       c["id"],
                "text":     c["doc"],
                "metadata": c["meta"],
                "distance": c["distance"] * multiplier,  # SU4 boost
                "type":     node_type,
            })

        # Raw-chunk candidates: no boost, always compete on raw cosine score.
        for c in raw_candidates:
            candidates.append({
                "id":       c["id"],
                "text":     c["doc"],
                "metadata": c["meta"],
                "distance": c["distance"],
                "type":     "raw_chunk",
            })

        if masked_count:
            logger.info(
                "retriever: SU5 masked %d child super-node(s) -- slots filled by raw_chunks",
                masked_count,
            )

        # ── 6. Re-rank and slice top_k ────────────────────────────────────────
        # Lower cosine distance = more similar = better rank.
        candidates.sort(key=lambda x: x["distance"])
        top = candidates[:top_k]

        # ── 7. SU10 -- Atomic SQLite access logging ───────────────────────────
        # Only log super-nodes that survived into the FINAL top-k, not all
        # candidates.  This keeps the hit_count signal accurate: a node
        # only gets credit if it actually reached the user.
        super_node_ids_in_top = [
            c["id"]
            for c in top
            if c["type"] in ("super_node", "meta_node") and c.get("id")
        ]
        if super_node_ids_in_top:
            try:
                sqlite_client.log_super_node_access(super_node_ids_in_top)
            except Exception:
                # SU10 failure must never crash the query path.
                logger.exception(
                    "retriever: SU10 log_super_node_access failed (non-fatal, "
                    "query response is unaffected)"
                )

        logger.info(
            "retriever: query=%r top_k=%d mode=%s masked=%d sn_hits=%d",
            query, top_k, mode, masked_count, len(super_node_ids_in_top),
        )
        return top


# ─────────────────────────────────────────────────────────────────────────────
# Interactive CLI test  (python -m app.retrieval.retriever)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 51)
    print("     SERA Phase-5 Retriever Interactive Test     ")
    print("=" * 51)
    print("Sample question: 'What are the symptoms of Mabry syndrome?'\n")

    while True:
        user_query = input("Enter your medical question (or 'q' to quit): ").strip()
        if user_query.lower() in {"q", "quit", "exit"}:
            break
        if not user_query:
            continue

        print(f"\n[Retrieving top chunks for: '{user_query}']...")
        results = Retriever.search(user_query, top_k=3)

        if not results:
            print("No results found.")
            continue

        for i, res in enumerate(results, 1):
            print(f"\n--- Result {i} (Distance: {res['distance']:.4f} | Type: {res['type']}) ---")
            print(f"ID:      {res['id']}")
            print(f"Section: {res['metadata'].get('section', 'N/A')}")
            print(f"Source:  {res['metadata'].get('source_url', 'N/A')}")
            print(f"Text:    {res['text'][:250]}...\n")
