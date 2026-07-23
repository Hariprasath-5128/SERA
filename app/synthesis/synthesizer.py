"""
app/synthesis/synthesizer.py
------------------------------
Phase 4 — Synthesis Orchestrator.

This module is the central brain of Phase 4. It coordinates the full
pipeline for converting a cluster of raw medical chunks into a validated
Super-Node stored in ChromaDB.

Upgrade Targets Implemented Here
----------------------------------
  SU1  — Semantic Deduplication     (via chunk_fetcher.fetch_and_deduplicate)
  SU2  — LLM-as-Judge Validation    (via validator.validate)
  SU12 — Re-Synthesis ID Reuse      (existing_sn_id preserved on update)
  SU13 — Upward Staleness Propagation (parent marked stale on child re-synthesis)
  SU14 — Synthesis Fidelity Bound   (drift_margin & sim_to_source checks)

Flow
-----
  1. Fetch + deduplicate raw chunks (SU1)
  2. LLM synthesis with retry loop (up to MAX_SYNTHESIS_RETRIES)
  3. LLM-as-judge validation (SU2) on each attempt
  4. Optional hallucination check via entity_extractor
  5. SU14 fidelity bound check (drift_margin / sim_to_source)
  6. Embed summary → 384-dim vector
  7. Upsert to super_nodes ChromaDB collection (via super_node_store)
  8. Write full audit trail to super_node_history SQLite table
  9. SU13 parent staleness propagation if re-synthesis
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app import config
from app.db import sqlite_client
from app.db import super_node_store
from app.generation.llm_client import llm_call
from app.ingestion.embedder import Embedder
from app.synthesis import chunk_fetcher
from app.synthesis.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    RESYNTH_PROMPT_TEMPLATE,
    META_NODE_SYSTEM_PROMPT,
    META_NODE_PROMPT_TEMPLATE,
)
from app.validation import entity_extractor, validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures (defined here; imported by Phase 3 pattern_finder etc.)
# ---------------------------------------------------------------------------

@dataclass
class SynthesisJob:
    """
    Inputs to the synthesizer, emitted by Phase 3 pattern_finder.

    chunk_ids      : raw doc_* IDs only — guaranteed by SU9 middleware.
    existing_sn_id : if set, Phase 4 will UPSERT this ID (SU12 re-synthesis).
                     If None, a fresh sn_* ID is minted.
    """
    cluster_id:      Optional[int]
    canonical_query: str
    chunk_ids:       list[str]
    hit_count:       int
    triggered_at:    datetime
    existing_sn_id:  Optional[str] = None
    # Optional override for meta-node merges (supplies structured disease prompt)
    prompt_template:        Optional[str] = None
    prompt_system_override: Optional[str] = None
    child_queries:          Optional[list[str]] = None


@dataclass
class ValidationResult:
    passed:            bool
    coverage_score:    float
    missing_facts:     list[str]
    source_fact_count: int
    summary_fact_count: int


class SynthesisError(Exception):
    """Raised when synthesis fails unrecoverably (fidelity, no chunks, etc.)."""
    pass


class ValidationError(SynthesisError):
    """Raised when LLM-as-judge rejects after all retries are exhausted."""
    pass


# ---------------------------------------------------------------------------
# SU14 — Synthesis Fidelity Bound
# ---------------------------------------------------------------------------

def check_fidelity_bound(
    summary_emb:      np.ndarray,
    source_chunk_embs: list[np.ndarray],
    source_query_emb:  np.ndarray,
) -> dict:
    """
    SU14 — Detect semantic drift in the generated summary.

    A valid summary should sit *between* its source chunks and the query in
    embedding space — a cleaner expression of the source, not an answer
    over-tuned to the query.

    Checks:
      suspected_overfit    : summary is MORE similar to the query than any raw
                             chunk ever was (drift_margin > MAX_DRIFT_MARGIN).
                             Soft flag — node is still served.
      suspected_disconnect : summary has drifted AWAY from its own source
                             (sim_to_source < MIN_SOURCE_ANCHOR).
                             Hard failure — synthesis is rejected.

    Edge case: if fewer than 2 source embeddings are provided, this check
    is skipped (returns a neutral result) and a warning is logged.
    """
    if len(source_chunk_embs) < 2:
        logger.warning(
            "synthesizer: SU14 skipped — fewer than 2 source embeddings "
            "(got %d). Returning neutral fidelity result.",
            len(source_chunk_embs),
        )
        return {
            "sim_summary_to_source":  1.0,
            "sim_summary_to_query":   0.0,
            "sim_raw_to_query_max":   0.0,
            "drift_margin":           0.0,
            "suspected_overfit":      False,
            "suspected_disconnect":   False,
        }

    source_centroid = np.mean(source_chunk_embs, axis=0)

    sim_summary_to_source = float(
        cosine_similarity([summary_emb], [source_centroid])[0][0]
    )
    sim_summary_to_query = float(
        cosine_similarity([summary_emb], [source_query_emb])[0][0]
    )
    sim_raw_to_query_max = float(max(
        cosine_similarity([e], [source_query_emb])[0][0]
        for e in source_chunk_embs
    ))

    drift_margin = sim_summary_to_query - sim_raw_to_query_max

    return {
        "sim_summary_to_source":  sim_summary_to_source,
        "sim_summary_to_query":   sim_summary_to_query,
        "sim_raw_to_query_max":   sim_raw_to_query_max,
        "drift_margin":           float(drift_margin),
        "suspected_overfit":      drift_margin > config.MAX_DRIFT_MARGIN,
        "suspected_disconnect":   sim_summary_to_source < config.MIN_SOURCE_ANCHOR,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _emb_sha256(emb: np.ndarray) -> str:
    return hashlib.sha256(emb.astype(np.float32).tobytes()).hexdigest()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encode_text(text: str) -> np.ndarray:
    """Embed a single raw string using the shared BAAI/bge-m3 model."""
    model = Embedder.get_model()
    return model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def run(job: SynthesisJob) -> str:
    """
    Execute the full Phase 4 synthesis pipeline for one cluster.

    Parameters
    ----------
    job : SynthesisJob
        Emitted by Phase 3 pattern_finder.

    Returns
    -------
    str
        The super_node_id (e.g. "sn_42_1720789200") of the upserted node.

    Raises
    ------
    SynthesisError
        If no chunks found, or SU14 hard-disconnects the summary.
    ValidationError
        If LLM-as-judge fails after MAX_SYNTHESIS_RETRIES.
    """
    max_retries = config.MAX_SYNTHESIS_RETRIES
    logger.info(
        "synthesizer: START cluster_id=%s | chunks=%d | query='%s'",
        job.cluster_id, len(job.chunk_ids), job.canonical_query[:60],
    )

    # ── Step 1: Fetch + Deduplicate (SU1) ─────────────────────────────────
    source_text, source_embeddings = chunk_fetcher.fetch_and_deduplicate(
        job.chunk_ids, max_tokens=4000
    )
    if not source_text:
        raise SynthesisError(
            f"synthesizer: No chunks found for cluster_id={job.cluster_id} "
            f"chunk_ids={job.chunk_ids}"
        )

    # ── Step 2 & 3: Synthesis + LLM-as-judge retry loop ──────────────────
    summary:       Optional[str] = None
    result:        Optional[validator.ValidationResult] = None
    missing_facts: list[str] = []

    for attempt in range(max_retries):
        if attempt == 0:
            # Use structured meta-node prompt if provided, else default
            if job.prompt_template:
                child_q_str = "\n".join(
                    f"- {q}" for q in (job.child_queries or [job.canonical_query])
                )
                user_prompt = job.prompt_template.format(
                    child_queries=child_q_str,
                    source_chunks_text=source_text,
                    canonical_query=job.canonical_query,
                )
            else:
                user_prompt = USER_PROMPT_TEMPLATE.format(
                    canonical_query=job.canonical_query,
                    source_chunks_text=source_text,
                )
        else:
            user_prompt = RESYNTH_PROMPT_TEMPLATE.format(
                missing_facts=missing_facts,
                source_chunks_text=source_text,
            )

        active_system = job.prompt_system_override or SYSTEM_PROMPT
        summary = llm_call(active_system, user_prompt, temperature=0.1, max_tokens=2500)

        # SU2 validation
        result = validator.validate(source_text, summary)

        if result.passed:
            logger.info(
                "synthesizer: validation PASSED | attempt=%d | coverage=%.3f",
                attempt + 1, result.coverage_score,
            )
            break

        missing_facts = result.missing_facts
        logger.warning(
            "synthesizer: validation FAILED | attempt=%d/%d | coverage=%.3f | missing=%d facts",
            attempt + 1, max_retries, result.coverage_score, len(missing_facts),
        )

    if not result or not result.passed:
        if job.cluster_id is not None:
            sqlite_client.set_synthesizing(job.cluster_id, False)
            # Mark cluster as failed so scheduler stops retrying
            with sqlite_client.get_connection() as conn:
                conn.execute(
                    "UPDATE query_clusters SET synthesis_failed=1 WHERE id=?",
                    (job.cluster_id,)
                )
        raise ValidationError(
            f"synthesizer: Max retries ({max_retries}) exceeded for "
            f"cluster_id={job.cluster_id}. "
            f"Final coverage: {result.coverage_score:.3f}"
        )

    # ── Step 4: Optional hallucination check (advisory) ──────────────────
    hallucination = entity_extractor.check_hallucination(source_text, summary)
    if hallucination["added_entities"]:
        logger.warning(
            "synthesizer: %d potential hallucination(s) detected for "
            "cluster_id=%s: %s",
            len(hallucination["added_entities"]),
            job.cluster_id,
            hallucination["added_entities"],
        )

    # ── Step 5: Embed summary ─────────────────────────────────────────────
    summary_emb = _encode_text(summary)

    # ── Step 6: SU14 — Fidelity Bound Check ──────────────────────────────
    query_emb = _encode_text(job.canonical_query)
    fidelity  = check_fidelity_bound(
        summary_emb=summary_emb,
        source_chunk_embs=source_embeddings,
        source_query_emb=query_emb,
    )
    logger.info(
        "synthesizer: SU14 fidelity | cluster_id=%s | "
        "sim_to_source=%.3f | drift_margin=%.3f | overfit=%s | disconnect=%s",
        job.cluster_id,
        fidelity["sim_summary_to_source"],
        fidelity["drift_margin"],
        fidelity["suspected_overfit"],
        fidelity["suspected_disconnect"],
    )

    if fidelity["suspected_disconnect"]:
        raise SynthesisError(
            f"synthesizer: SU14 hard reject — cluster_id={job.cluster_id} "
            f"sim_to_source={fidelity['sim_summary_to_source']:.3f} "
            f"< MIN_SOURCE_ANCHOR={config.MIN_SOURCE_ANCHOR}. "
            f"Summary disconnected from ground truth."
        )

    fidelity_flagged = False
    if fidelity["suspected_overfit"]:
        logger.warning(
            "synthesizer: SU14 soft flag — cluster_id=%s | drift_margin=%.3f",
            job.cluster_id, fidelity["drift_margin"],
        )
        fidelity_flagged = True

    # ── Step 7: SU12 — ID assignment ─────────────────────────────────────
    now_iso = _utcnow_iso()

    if job.existing_sn_id:
        # Re-synthesis: preserve the same sn_id and increment revision
        sn_id = job.existing_sn_id
        existing_meta = super_node_store.get_metadata(sn_id)
        revision      = existing_meta.get("revision", 1) + 1
        parent_meta_id = existing_meta.get("parent_meta_id", "")
        lineage_depth  = existing_meta.get("lineage_depth", 0)
        created_at     = existing_meta.get("created_at", now_iso)
        logger.info(
            "synthesizer: SU12 re-synthesis — sn_id=%s | revision=%d",
            sn_id, revision,
        )
    else:
        # First synthesis: mint a new sn_id
        sn_id          = f"sn_{job.cluster_id}_{int(datetime.now(timezone.utc).timestamp())}"
        revision        = 1
        parent_meta_id  = ""
        lineage_depth   = 0
        created_at      = now_iso

    # ── Step 8: Upsert to super_nodes ChromaDB ───────────────────────────
    super_node_store.upsert(
        sn_id=sn_id,
        summary=summary,
        embedding=summary_emb,
        metadata={
            "type":               "super_node",
            "source_query":       job.canonical_query,
            "source_chunks":      json.dumps(job.chunk_ids),
            "hit_count":          job.hit_count,
            "created_at":         created_at,
            "last_accessed":      now_iso,
            "access_count":       0,
            "decay_score":        1.0,
            "cluster_id":         job.cluster_id if job.cluster_id is not None else -1,
            "fact_coverage":      result.coverage_score,
            "in_hierarchy":       False,
            "is_stale":           False,
            "parent_meta_id":     parent_meta_id,
            "lineage_depth":      lineage_depth,
            "revision":           revision,
            "fidelity_flagged":   fidelity_flagged,
            "drift_margin":       fidelity["drift_margin"],
            "sim_summary_to_source": fidelity["sim_summary_to_source"],
            "sim_summary_to_query":  fidelity["sim_summary_to_query"],
        },
    )
    logger.info(
        "synthesizer: upserted super_node | sn_id=%s | revision=%d",
        sn_id, revision,
    )

    # ── Step 9: SQLite audit trail ────────────────────────────────────────
    if job.cluster_id is not None:
        source_centroid = (
            np.mean(source_embeddings, axis=0)
            if source_embeddings
            else np.zeros(summary_emb.shape)
        )
        sqlite_client.insert_super_node_history(
            cluster_id=job.cluster_id,
            super_node_id=sn_id,
            revision=revision,
            summary=summary,
            embedding_hash=_emb_sha256(summary_emb),
            coverage_score=result.coverage_score,
            drift_margin=fidelity["drift_margin"],
            chunk_ids_hash=_sha256(json.dumps(sorted(job.chunk_ids))),
            centroid_hash=_emb_sha256(source_centroid),
            centroid=source_centroid.astype(np.float32).tobytes(),
        )

        # Mark cluster as synthesized in SQLite
        sqlite_client.mark_cluster_synthesized(job.cluster_id, sn_id)
        logger.info(
            "synthesizer: cluster_id=%s marked synthesized | sn_id=%s",
            job.cluster_id, sn_id,
        )
    else:
        logger.info(
            "synthesizer: synthesized meta-node/cluster-less node | sn_id=%s",
            sn_id,
        )

    # ── Step 10: SU13 — Upward Staleness Propagation ─────────────────────
    if revision > 1 and parent_meta_id:
        try:
            parent_meta = super_node_store.get_metadata(parent_meta_id)
            super_node_store.update_metadata(
                parent_meta_id,
                {**parent_meta, "is_stale": True, "stale_reason": "child_revised"},
            )
            parent_cluster_id = parent_meta.get("cluster_id")
            if parent_cluster_id:
                sqlite_client.flag_cluster_for_re_synthesis(int(parent_cluster_id))
            logger.info(
                "synthesizer: SU13 — parent_id=%s marked stale (child sn_id=%s, revision=%d)",
                parent_meta_id, sn_id, revision,
            )
        except Exception as exc:
            logger.warning(
                "synthesizer: SU13 upward propagation failed | parent_id=%s | error=%s",
                parent_meta_id, exc,
            )

    return sn_id
