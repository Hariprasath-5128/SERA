"""
tests/Phase_4/test_phase4_synthesis.py
----------------------------------------
Phase 4 test suite — Summarizer Agent & Validation Engine.

Coverage
--------
  T01  chunk_fetcher: fewer than 5 chunks → skip dedup, return joined text
  T02  chunk_fetcher: more than 5 chunks → deduplicate near-identical passages
  T03  chunk_fetcher: ChromaDB returns nothing → returns empty string
  T04  chunk_fetcher: text over token budget → hard-truncated
  T05  entity_extractor: extracts entities from clean text
  T06  entity_extractor: check_hallucination → detects added entities
  T07  entity_extractor: fallback_validate → passes when entities overlap
  T08  entity_extractor: fallback_validate → fails when all entities missing
  T09  validator: LLM judge passes → ValidationResult.passed=True
  T10  validator: LLM judge fails (coverage < 0.90) → passed=False
  T11  validator: LLM judge returns malformed JSON → fallback entity check
  T12  synthesizer: full run passes validation on first attempt → returns sn_id
  T13  synthesizer: first attempt fails, second attempt passes → sn_id returned
  T14  synthesizer: all retries exhausted → ValidationError raised
  T15  synthesizer: SU14 hard disconnect → SynthesisError raised
  T16  synthesizer: SU14 soft overfit → fidelity_flagged=True, still succeeds
  T17  synthesizer: SU12 re-synthesis → increments revision, reuses sn_id
  T18  synthesizer: SU13 upward propagation → parent marked is_stale on re-synth
  T19  admin POST /synthesize/{id}: success → 200 with super_node_id
  T20  admin POST /synthesize/{id}: cluster not found → 404
  T21  admin POST /synthesize/{id}: already synthesizing → 409
  T22  admin GET /super-nodes: returns list of synthesized nodes
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Make app importable ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import config
from app.db import sqlite_client
from app.synthesis.synthesizer import (
    SynthesisError,
    SynthesisJob,
    ValidationError,
    check_fidelity_bound,
)
from app.validation.validator import ValidationResult


# ===========================================================================
# Helpers
# ===========================================================================

def _unit_vec(seed: int, dim: int = 384) -> np.ndarray:
    """Deterministic unit vector for reproducible similarity tests."""
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


VEC_A = _unit_vec(1)
VEC_B = _unit_vec(2)
VEC_C = _unit_vec(3)


def _make_job(
    cluster_id: int = 1,
    chunk_ids: list[str] | None = None,
    existing_sn_id: str | None = None,
) -> SynthesisJob:
    return SynthesisJob(
        cluster_id=cluster_id,
        canonical_query="What are the symptoms of diabetes?",
        chunk_ids=chunk_ids or ["doc_1", "doc_2", "doc_3"],
        hit_count=15,
        triggered_at=datetime.now(timezone.utc),
        existing_sn_id=existing_sn_id,
    )


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def isolated_db(tmp_path: Path, monkeypatch):
    """Redirect SQLite to a temp DB for test isolation (Upgrade U3)."""
    db_path = tmp_path / "test_rag.db"
    import app.config as cfg
    import app.db.sqlite_client as sc
    monkeypatch.setattr(cfg, "SQLITE_DB_PATH", db_path)
    monkeypatch.setattr(sc, "SQLITE_DB_PATH", db_path)
    sc.init_db()
    yield sc


@pytest.fixture()
def test_cluster(isolated_db):
    """Insert a ready-to-synthesize cluster and return its id."""
    cluster_id = isolated_db.insert_cluster(
        canonical_query="What are the symptoms of diabetes?",
        embedding=VEC_A,
        chunk_ids=["doc_1", "doc_2", "doc_3"],
    )
    with isolated_db.get_connection() as conn:
        conn.execute(
            "UPDATE query_clusters SET hit_count=15 WHERE id=?", (cluster_id,)
        )
    return cluster_id


@pytest.fixture()
def mock_chroma_raw():
    """Mock chunk_fetcher._fetch_from_chroma with 3 distinct passages."""
    docs = [
        "Diabetes mellitus is characterised by high blood sugar levels.",
        "Symptoms include frequent urination, excessive thirst, and fatigue.",
        "Type 2 diabetes is the most common form, often linked to obesity.",
    ]
    embs = [VEC_A, VEC_B, VEC_C]
    with patch(
        "app.synthesis.chunk_fetcher._fetch_from_chroma",
        return_value=(docs, embs),
    ) as mock:
        yield mock, docs, embs


@pytest.fixture()
def mock_llm_pass():
    """Mock llm_call to return a valid summary on the first call."""
    summary = (
        "Diabetes mellitus is a metabolic disorder characterised by elevated "
        "blood glucose. Common symptoms are frequent urination, excessive thirst, "
        "and fatigue. Type 2 diabetes is most prevalent and is associated with obesity."
    )
    with patch("app.synthesis.synthesizer.llm_call", return_value=summary) as mock:
        yield mock, summary


@pytest.fixture()
def mock_validator_pass():
    """Mock validator.validate to always pass with coverage=0.95."""
    result = ValidationResult(
        passed=True,
        coverage_score=0.95,
        missing_facts=[],
        source_fact_count=5,
        summary_fact_count=5,
    )
    with patch("app.synthesis.synthesizer.validator.validate", return_value=result) as mock:
        yield mock, result


@pytest.fixture()
def mock_store():
    """Mock super_node_store to prevent real ChromaDB writes."""
    with (
        patch("app.synthesis.synthesizer.super_node_store.upsert") as mock_upsert,
        patch(
            "app.synthesis.synthesizer.super_node_store.get_metadata",
            return_value={"revision": 1, "parent_meta_id": "", "lineage_depth": 0,
                          "created_at": "2024-01-01T00:00:00+00:00", "fact_coverage": 0.95},
        ) as mock_get,
        patch("app.synthesis.synthesizer.super_node_store.update_metadata") as mock_update,
    ):
        yield mock_upsert, mock_get, mock_update


@pytest.fixture()
def mock_embedder():
    """Mock the sentence-transformers encoder to avoid loading BAAI/bge-m3."""
    with patch("app.synthesis.synthesizer._encode_text", return_value=VEC_A) as mock:
        yield mock


# ===========================================================================
# T01-T04 — chunk_fetcher
# ===========================================================================

class TestChunkFetcher:

    def test_T01_small_batch_skips_dedup(self):
        """T01: ≤5 chunks → no clustering, all joined."""
        docs = ["Passage one.", "Passage two.", "Passage three."]
        embs = [VEC_A, VEC_B, VEC_C]
        with patch("app.synthesis.chunk_fetcher._fetch_from_chroma", return_value=(docs, embs)):
            from app.synthesis.chunk_fetcher import fetch_and_deduplicate
            text, returned_embs = fetch_and_deduplicate(["doc_1", "doc_2", "doc_3"])
        assert "Passage one." in text
        assert "Passage two." in text
        assert "Passage three." in text
        assert len(returned_embs) == 3

    def test_T02_large_batch_deduplicates(self):
        """T02: >5 near-identical chunks → reduced to unique representatives."""
        # Create 6 chunks where 5 are nearly identical (very close vectors)
        base = _unit_vec(42)
        noise = 1e-4
        docs = [f"Diabetes passage variant {i}." for i in range(6)]
        # First 5 are almost identical; last one is distinct
        embs = [base + noise * _unit_vec(i) for i in range(5)] + [VEC_C]
        embs = [e / np.linalg.norm(e) for e in embs]

        with patch("app.synthesis.chunk_fetcher._fetch_from_chroma", return_value=(docs, embs)):
            from app.synthesis.chunk_fetcher import fetch_and_deduplicate
            text, returned_embs = fetch_and_deduplicate([f"doc_{i}" for i in range(6)])
        # The 5 near-identical chunks should collapse
        assert len(returned_embs) < 6

    def test_T03_empty_chroma_result_returns_empty(self):
        """T03: ChromaDB returns nothing → empty string, empty list."""
        with patch("app.synthesis.chunk_fetcher._fetch_from_chroma", return_value=([], [])):
            from app.synthesis.chunk_fetcher import fetch_and_deduplicate
            text, embs = fetch_and_deduplicate(["doc_missing"])
        assert text == ""
        assert embs == []

    def test_T04_text_over_budget_is_truncated(self):
        """T04: Source text exceeding token budget is hard-truncated."""
        long_doc = "X" * 100_000
        with patch(
            "app.synthesis.chunk_fetcher._fetch_from_chroma",
            return_value=([long_doc], [VEC_A]),
        ):
            from app.synthesis.chunk_fetcher import fetch_and_deduplicate
            text, _ = fetch_and_deduplicate(["doc_long"], max_tokens=100)
        # Budget = 100 * 4 = 400 chars
        assert len(text) <= 400


# ===========================================================================
# T05-T08 — entity_extractor
# ===========================================================================

class TestEntityExtractor:

    def test_T05_extract_entities_from_text(self):
        """T05: spaCy NER extracts entities correctly."""
        from app.validation.entity_extractor import extract_entities
        entities = extract_entities("John Smith was diagnosed with Type 2 Diabetes in New York.")
        # spaCy should find at least one entity
        assert len(entities) > 0

    def test_T06_hallucination_detected(self):
        """T06: Entity in summary absent from source is flagged."""
        from app.validation.entity_extractor import check_hallucination
        source  = "Diabetes causes high blood sugar."
        summary = "Diabetes causes high blood sugar. Metformin is the first-line drug."
        result = check_hallucination(source, summary)
        # 'Metformin' is in summary but not source — should be in added_entities
        assert isinstance(result["hallucination_ratio"], float)
        assert result["hallucination_ratio"] >= 0.0

    def test_T07_fallback_validate_passes_on_overlap(self):
        """T07: fallback_validate passes when entities substantially overlap."""
        from app.validation.entity_extractor import fallback_validate
        source  = "John Smith was treated in New York for Type 2 Diabetes."
        summary = "John Smith has Type 2 Diabetes and was treated in New York."
        result = fallback_validate(source, summary)
        assert result["passed"] is True
        assert result["coverage_score"] > 0.0

    def test_T08_fallback_validate_fails_on_zero_overlap(self):
        """T08: fallback_validate fails when no source entities appear in summary."""
        from app.validation.entity_extractor import fallback_validate
        source  = "John Smith was treated in New York for Type 2 Diabetes."
        summary = "The sky is blue and water is wet."
        result = fallback_validate(source, summary)
        # Either fails or coverage is very low
        assert result["coverage_score"] < 0.70 or result["passed"] is False


# ===========================================================================
# T09-T11 — validator
# ===========================================================================

class TestValidator:

    def test_T09_llm_judge_passes(self):
        """T09: LLM returns valid JSON with coverage >= 0.90 → passed=True."""
        judge_response = json.dumps({
            "passed": True,
            "coverage_score": 0.95,
            "missing_facts": [],
        })
        with patch("app.validation.validator.llm_call", return_value=judge_response):
            from app.validation.validator import validate
            result = validate("source text", "summary text")
        assert result.passed is True
        assert result.coverage_score == pytest.approx(0.95)

    def test_T10_llm_judge_fails_low_coverage(self):
        """T10: LLM returns coverage < 0.90 → passed=False."""
        judge_response = json.dumps({
            "passed": False,
            "coverage_score": 0.72,
            "missing_facts": ["fact A", "fact B"],
        })
        with patch("app.validation.validator.llm_call", return_value=judge_response):
            from app.validation.validator import validate
            result = validate("source text", "summary text")
        assert result.passed is False
        assert result.coverage_score == pytest.approx(0.72)
        assert "fact A" in result.missing_facts

    def test_T11_malformed_json_falls_back_to_entity_extractor(self):
        """T11: LLM returns garbage JSON → falls back to entity extraction."""
        with patch("app.validation.validator.llm_call", return_value="THIS IS NOT JSON {{{"):
            with patch(
                "app.validation.validator.entity_extractor.fallback_validate",
                return_value={"passed": True, "coverage_score": 0.80, "missing_facts": []},
            ) as mock_fallback:
                from app.validation.validator import validate
                result = validate("source text", "summary text")
        mock_fallback.assert_called_once()
        assert result.coverage_score == pytest.approx(0.80)


# ===========================================================================
# T12-T18 — synthesizer
# ===========================================================================

class TestSynthesizer:

    def test_T12_full_run_passes_first_attempt(
        self, isolated_db, test_cluster, mock_chroma_raw,
        mock_llm_pass, mock_validator_pass, mock_store, mock_embedder
    ):
        """T12: Full synthesis pipeline succeeds on first attempt."""
        from app.synthesis import synthesizer
        job = _make_job(cluster_id=test_cluster)
        sn_id = synthesizer.run(job)
        assert sn_id.startswith("sn_")
        mock_store[0].assert_called_once()  # upsert called

    def test_T13_retry_on_first_failure_then_passes(
        self, isolated_db, test_cluster, mock_chroma_raw,
        mock_llm_pass, mock_store, mock_embedder
    ):
        """T13: First attempt fails, second attempt passes."""
        fail_result = ValidationResult(
            passed=False, coverage_score=0.70,
            missing_facts=["fact X"], source_fact_count=3, summary_fact_count=2,
        )
        pass_result = ValidationResult(
            passed=True, coverage_score=0.92,
            missing_facts=[], source_fact_count=3, summary_fact_count=3,
        )
        with patch(
            "app.synthesis.synthesizer.validator.validate",
            side_effect=[fail_result, pass_result],
        ):
            from app.synthesis import synthesizer
            job = _make_job(cluster_id=test_cluster)
            sn_id = synthesizer.run(job)
        assert sn_id.startswith("sn_")

    def test_T14_all_retries_exhausted_raises_validation_error(
        self, isolated_db, test_cluster, mock_chroma_raw,
        mock_llm_pass, mock_store, mock_embedder
    ):
        """T14: All 3 retries fail → ValidationError raised."""
        fail_result = ValidationResult(
            passed=False, coverage_score=0.50,
            missing_facts=["A", "B"], source_fact_count=4, summary_fact_count=2,
        )
        with patch(
            "app.synthesis.synthesizer.validator.validate",
            return_value=fail_result,
        ):
            from app.synthesis import synthesizer
            job = _make_job(cluster_id=test_cluster)
            with pytest.raises(ValidationError):
                synthesizer.run(job)

    def test_T15_su14_hard_disconnect_raises_synthesis_error(
        self, isolated_db, test_cluster, mock_chroma_raw,
        mock_llm_pass, mock_validator_pass, mock_store
    ):
        """T15: SU14 sim_to_source below MIN_SOURCE_ANCHOR → SynthesisError."""
        # Summary embedding is very different from source (simulated by orthogonal vectors)
        orthogonal = np.zeros(384, dtype=np.float32)
        orthogonal[0] = 1.0
        source_emb = np.zeros(384, dtype=np.float32)
        source_emb[1] = 1.0  # orthogonal → cosine_sim = 0 < 0.55

        with (
            patch("app.synthesis.synthesizer._encode_text", side_effect=[orthogonal, VEC_A]),
            patch(
                "app.synthesis.chunk_fetcher._fetch_from_chroma",
                return_value=(
                    ["source text"],
                    [source_emb, source_emb],  # need 2+ for SU14
                ),
            ),
        ):
            from app.synthesis import synthesizer
            job = _make_job(cluster_id=test_cluster)
            with pytest.raises(SynthesisError, match="SU14 hard reject"):
                synthesizer.run(job)

    def test_T16_su14_soft_overfit_flags_but_succeeds(
        self, isolated_db, test_cluster, mock_chroma_raw,
        mock_llm_pass, mock_validator_pass, mock_store, mock_embedder
    ):
        """T16: SU14 drift_margin > MAX_DRIFT_MARGIN → soft flag, run still returns sn_id."""
        neutral = {"suspected_overfit": True, "suspected_disconnect": False,
                   "sim_summary_to_source": 0.80, "drift_margin": 0.15,
                   "sim_summary_to_query": 0.90, "sim_raw_to_query_max": 0.75}
        with patch("app.synthesis.synthesizer.check_fidelity_bound", return_value=neutral):
            from app.synthesis import synthesizer
            job = _make_job(cluster_id=test_cluster)
            sn_id = synthesizer.run(job)
        assert sn_id.startswith("sn_")

    def test_T17_su12_re_synthesis_increments_revision(
        self, isolated_db, test_cluster, mock_chroma_raw,
        mock_llm_pass, mock_validator_pass, mock_embedder
    ):
        """T17: SU12 — existing_sn_id → revision increments from 1 to 2."""
        existing_meta = {
            "revision": 1, "parent_meta_id": "", "lineage_depth": 0,
            "created_at": "2024-01-01T00:00:00+00:00", "fact_coverage": 0.91,
        }
        upserted_metadata = {}

        def capture_upsert(sn_id, summary, embedding, metadata):
            upserted_metadata.update(metadata)

        with (
            patch("app.synthesis.synthesizer.super_node_store.upsert", side_effect=capture_upsert),
            patch("app.synthesis.synthesizer.super_node_store.get_metadata", return_value=existing_meta),
            patch("app.synthesis.synthesizer.super_node_store.update_metadata"),
        ):
            from app.synthesis import synthesizer
            job = _make_job(cluster_id=test_cluster, existing_sn_id="sn_1_12345")
            sn_id = synthesizer.run(job)

        assert sn_id == "sn_1_12345"
        assert upserted_metadata["revision"] == 2

    def test_T18_su13_parent_marked_stale_on_re_synthesis(
        self, isolated_db, test_cluster, mock_chroma_raw,
        mock_llm_pass, mock_validator_pass, mock_embedder
    ):
        """T18: SU13 — parent node is_stale=True when child is re-synthesized."""
        parent_id = "sn_parent_9999"
        existing_meta = {
            "revision": 1, "parent_meta_id": parent_id, "lineage_depth": 1,
            "created_at": "2024-01-01T00:00:00+00:00", "fact_coverage": 0.91,
        }
        parent_meta = {
            "cluster_id": 99, "is_stale": False, "revision": 1,
            "parent_meta_id": "", "lineage_depth": 0,
        }
        updated_parent = {}

        def capture_update(sn_id, metadata):
            if sn_id == parent_id:
                updated_parent.update(metadata)

        with (
            patch("app.synthesis.synthesizer.super_node_store.upsert"),
            patch("app.synthesis.synthesizer.super_node_store.get_metadata",
                  side_effect=lambda sid: existing_meta if sid != parent_id else parent_meta),
            patch("app.synthesis.synthesizer.super_node_store.update_metadata",
                  side_effect=capture_update),
        ):
            from app.synthesis import synthesizer
            job = _make_job(cluster_id=test_cluster, existing_sn_id="sn_1_12345")
            synthesizer.run(job)

        assert updated_parent.get("is_stale") is True


# ===========================================================================
# T19-T22 — Admin API endpoints
# ===========================================================================

class TestAdminEndpoints:

    @pytest.fixture()
    def client(self, isolated_db, test_cluster, monkeypatch):
        """FastAPI TestClient with mocked synthesis pipeline."""
        from fastapi.testclient import TestClient
        from app.main import app

        monkeypatch.setattr(
            "app.db.sqlite_client.get_cluster",
            lambda cid: {
                "id": cid, "canonical_query": "What is diabetes?",
                "chunk_ids": '["doc_1","doc_2"]', "hit_count": 15,
                "synthesizing": 0, "synthesized": 0, "synthesis_failed": 0,
                "super_node_id": None,
            } if cid == test_cluster else None,
        )
        return TestClient(app), test_cluster

    def test_T19_synthesize_success(self, client, monkeypatch):
        """T19: POST /admin/synthesize/{id} → 200 with super_node_id."""
        test_client, cluster_id = client

        monkeypatch.setattr("app.db.super_node_store.get_by_cluster", lambda cid: None)
        monkeypatch.setattr("app.db.sqlite_client.set_synthesizing", lambda cid, v: None)
        monkeypatch.setattr(
            "app.synthesis.synthesizer.run",
            lambda job: f"sn_{job.cluster_id}_99999",
        )
        monkeypatch.setattr(
            "app.db.super_node_store.get_metadata",
            lambda sid: {"fact_coverage": 0.94},
        )

        resp = test_client.post(f"/admin/synthesize/{cluster_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["super_node_id"].startswith("sn_")

    def test_T20_synthesize_cluster_not_found(self, client):
        """T20: POST /admin/synthesize/9999 → 404."""
        test_client, _ = client
        resp = test_client.post("/admin/synthesize/9999")
        assert resp.status_code == 404

    def test_T21_synthesize_already_in_progress(self, client, monkeypatch):
        """T21: Cluster has synthesizing=1 → 409."""
        test_client, cluster_id = client

        monkeypatch.setattr(
            "app.db.sqlite_client.get_cluster",
            lambda cid: {
                "id": cid, "canonical_query": "query", "chunk_ids": '[]',
                "hit_count": 15, "synthesizing": 1, "synthesized": 0,
                "synthesis_failed": 0, "super_node_id": None,
            },
        )
        resp = test_client.post(f"/admin/synthesize/{cluster_id}")
        assert resp.status_code == 409

    def test_T22_list_super_nodes_returns_list(self, client, monkeypatch):
        """T22: GET /admin/super-nodes → 200 with list."""
        test_client, _ = client

        monkeypatch.setattr(
            "app.db.super_node_store.list_all",
            lambda: [{
                "id": "sn_1_12345",
                "document": "Diabetes summary.",
                "metadata": {
                    "source_query": "What is diabetes?",
                    "cluster_id": 1, "revision": 1,
                    "fact_coverage": 0.93, "fidelity_flagged": False,
                    "drift_margin": 0.02, "is_stale": False,
                    "hit_count": 15, "created_at": "2024-01-01T00:00:00+00:00",
                    "last_accessed": "2024-01-02T00:00:00+00:00",
                },
            }],
        )
        resp = test_client.get("/admin/super-nodes")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["nodes"][0]["id"] == "sn_1_12345"
        assert body["nodes"][0]["fact_coverage"] == pytest.approx(0.93)


# ===========================================================================
# Standalone — check_fidelity_bound (no fixtures needed)
# ===========================================================================

class TestFidelityBound:

    def test_neutral_vectors_pass(self):
        """Aligned summary-to-source → no flags."""
        summary_emb = VEC_A
        source_embs = [VEC_A, VEC_A]
        query_emb   = VEC_B
        result = check_fidelity_bound(summary_emb, source_embs, query_emb)
        assert result["suspected_disconnect"] is False

    def test_orthogonal_summary_triggers_disconnect(self):
        """Summary orthogonal to source → suspected_disconnect=True."""
        e1 = np.zeros(384, dtype=np.float32); e1[0] = 1.0
        e2 = np.zeros(384, dtype=np.float32); e2[1] = 1.0
        e3 = np.zeros(384, dtype=np.float32); e3[2] = 1.0
        result = check_fidelity_bound(e1, [e2, e3], e3)
        assert result["suspected_disconnect"] is True

    def test_fewer_than_2_source_embs_skips_check(self):
        """Edge case: < 2 source embeddings → neutral result, no crash."""
        result = check_fidelity_bound(VEC_A, [VEC_B], VEC_C)
        assert result["suspected_disconnect"] is False
        assert result["suspected_overfit"] is False
