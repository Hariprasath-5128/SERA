"""
tests/test_phase2_logging.py
-----------------------------
Phase 2 — Query Logging & Hashing

Tests cover:
  1. New query creates a cluster
  2. Semantically similar query merges into existing cluster (hit_count +=1)
  3. Dissimilar query creates a second, separate cluster
  4. SU6 — centroid BLOB shifts after second match
  5. SU9 — sn_* IDs are filtered out by the middleware; only doc_* IDs logged

All tests use:
  - An in-memory SQLite database (tmp_path fixture) so nothing touches disk.
  - Mocked embeddings (controlled float32 vectors) so tests don't require
    the GPU/model to be available.
  - The FastAPI TestClient for middleware integration tests.
"""

import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vec(seed: float, dim: int = 8) -> np.ndarray:
    """Reproducible unit-normalised float32 vector."""
    rng = np.random.default_rng(int(seed * 1_000_000))
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# A pair of vectors that are very close (cosine sim > 0.92)
VEC_A  = _make_vec(1.0)
VEC_A2 = (VEC_A + _make_vec(1.1) * 0.01)  # tiny perturbation → still very similar
VEC_A2 = (VEC_A2 / np.linalg.norm(VEC_A2)).astype(np.float32)

# A vector pointing in a very different direction (cosine sim << 0.92)
VEC_B = _make_vec(99.0)
# Force VEC_B to be near-orthogonal to VEC_A
VEC_B = VEC_B - VEC_A * float(np.dot(VEC_B, VEC_A))
VEC_B = (VEC_B / np.linalg.norm(VEC_B)).astype(np.float32)

DIM = VEC_A.shape[0]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Return a fresh temporary SQLite path and pre-create all tables."""
    p = tmp_path / "test_rag.db"
    return p


@pytest.fixture(autouse=True)
def patch_sqlite_path(db_path: Path, monkeypatch):
    """
    Redirect every sqlite_client import to use the temp DB.

    We monkeypatch the module-level SQLITE_DB_PATH in both config and
    sqlite_client so that all connections go to the temp file.
    """
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))

    import app.config as cfg
    import app.db.sqlite_client as sc
    import app.logging_.query_logger as ql

    monkeypatch.setattr(cfg, "SQLITE_DB_PATH", db_path)
    monkeypatch.setattr(sc, "SQLITE_DB_PATH", db_path)

    # Reset cached dimension so mock embeddings (dim=8) don't trigger the corruption check
    ql._EMB_DIM = None

    # Also patch the config values that sqlite_client imports at call time
    monkeypatch.setattr(cfg, "COSINE_THRESHOLD", 0.92)

    # Initialise tables in the temp DB
    sc.init_db()


@pytest.fixture()
def mock_embed_A(monkeypatch):
    """Patch normalizer.embed to always return VEC_A."""
    import app.logging_.normalizer as norm
    monkeypatch.setattr(norm, "embed", lambda text: VEC_A.copy())


@pytest.fixture()
def mock_embed_A2(monkeypatch):
    """Patch normalizer.embed to always return VEC_A2 (near-identical to A)."""
    import app.logging_.normalizer as norm
    monkeypatch.setattr(norm, "embed", lambda text: VEC_A2.copy())


@pytest.fixture()
def mock_embed_B(monkeypatch):
    """Patch normalizer.embed to always return VEC_B (dissimilar to A)."""
    import app.logging_.normalizer as norm
    monkeypatch.setattr(norm, "embed", lambda text: VEC_B.copy())


# ---------------------------------------------------------------------------
# Test 1 — new query creates exactly one cluster
# ---------------------------------------------------------------------------

def test_new_query_creates_cluster(mock_embed_A):
    from app.logging_ import query_logger
    from app.db import sqlite_client

    cluster_id = query_logger.log_query(
        raw_query="What is Niemann-Pick disease?",
        retrieved_chunk_ids=["doc_c1", "doc_c2"],
    )

    assert isinstance(cluster_id, int)
    assert cluster_id > 0

    rows = sqlite_client.fetch_all_clusters()
    assert len(rows) == 1
    assert rows[0]["canonical_query"] == "what is niemann-pick disease?"
    assert rows[0]["hit_count"] == 1

    ids = json.loads(rows[0]["chunk_ids"])
    assert set(ids) == {"doc_c1", "doc_c2"}


# ---------------------------------------------------------------------------
# Test 2 — similar query merges into existing cluster, hit_count becomes 2
# ---------------------------------------------------------------------------

def test_similar_query_increments_hit(monkeypatch):
    """
    Send query A, then query A2 (near-identical vector).
    The second call should match the first cluster and increment hit_count to 2.
    """
    import app.logging_.normalizer as norm
    from app.logging_ import query_logger
    from app.db import sqlite_client

    # First query — cluster created
    monkeypatch.setattr(norm, "embed", lambda text: VEC_A.copy())
    cid1 = query_logger.log_query(
        raw_query="What is Niemann-Pick disease?",
        retrieved_chunk_ids=["doc_c1"],
    )

    # Second query — similar vector, should merge
    monkeypatch.setattr(norm, "embed", lambda text: VEC_A2.copy())
    cid2 = query_logger.log_query(
        raw_query="Tell me about Niemann Pick disease",
        retrieved_chunk_ids=["doc_c2"],
    )

    assert cid1 == cid2, "Both queries must resolve to the same cluster"

    rows = sqlite_client.fetch_all_clusters()
    assert len(rows) == 1
    assert rows[0]["hit_count"] == 2

    merged_ids = json.loads(rows[0]["chunk_ids"])
    assert set(merged_ids) == {"doc_c1", "doc_c2"}, \
        "chunk_ids should be the union of both queries"


# ---------------------------------------------------------------------------
# Test 3 — dissimilar query creates a second cluster
# ---------------------------------------------------------------------------

def test_dissimilar_query_creates_new_cluster(monkeypatch):
    import app.logging_.normalizer as norm
    from app.logging_ import query_logger
    from app.db import sqlite_client

    # First query
    monkeypatch.setattr(norm, "embed", lambda text: VEC_A.copy())
    query_logger.log_query(
        raw_query="What is Niemann-Pick disease?",
        retrieved_chunk_ids=["doc_c1"],
    )

    # Second, completely unrelated query
    monkeypatch.setattr(norm, "embed", lambda text: VEC_B.copy())
    query_logger.log_query(
        raw_query="How does insulin regulation work?",
        retrieved_chunk_ids=["doc_c3"],
    )

    rows = sqlite_client.fetch_all_clusters()
    assert len(rows) == 2, "Dissimilar queries must produce two separate clusters"


# ---------------------------------------------------------------------------
# Test 4 — SU6: centroid BLOB shifts after second match
# ---------------------------------------------------------------------------

def test_centroid_updates_on_match(monkeypatch):
    """
    After a second query merges into an existing cluster, the stored
    centroid BLOB must differ from the original (running weighted average).
    """
    import app.logging_.normalizer as norm
    from app.logging_ import query_logger
    from app.db import sqlite_client

    # Create cluster with VEC_A
    monkeypatch.setattr(norm, "embed", lambda text: VEC_A.copy())
    cid = query_logger.log_query(
        raw_query="What is Niemann-Pick disease?",
        retrieved_chunk_ids=["doc_c1"],
    )

    row_before = sqlite_client.get_cluster(cid)
    blob_before = bytes(row_before["query_embedding"])

    # Merge with VEC_A2
    monkeypatch.setattr(norm, "embed", lambda text: VEC_A2.copy())
    query_logger.log_query(
        raw_query="Tell me about Niemann Pick disease",
        retrieved_chunk_ids=["doc_c2"],
    )

    row_after = sqlite_client.get_cluster(cid)
    blob_after = bytes(row_after["query_embedding"])

    assert blob_before != blob_after, \
        "SU6: centroid BLOB must change after a second matching query"

    # Verify the new centroid is the correct running average
    centroid_before = np.frombuffer(blob_before, dtype=np.float32)
    expected = (centroid_before * 1 + VEC_A2) / 2   # n=1 before second hit
    actual   = np.frombuffer(blob_after,  dtype=np.float32)
    np.testing.assert_allclose(actual, expected, atol=1e-5,
        err_msg="SU6: centroid must equal (old*n + new)/(n+1)")


# ---------------------------------------------------------------------------
# Test 5 — SU9: middleware filters sn_* IDs, only doc_* IDs are logged
# ---------------------------------------------------------------------------

def test_interceptor_filters_super_node_ids(monkeypatch, tmp_path):
    """
    When the /query response contains mixed chunk_ids (doc_* and sn_*),
    only the doc_* IDs must reach query_logger.log_query().
    """
    import app.logging_.normalizer as norm
    monkeypatch.setattr(norm, "embed", lambda text: VEC_A.copy())

    # Capture what log_query actually receives
    logged_calls: list[dict] = []

    import app.logging_.query_logger as ql
    original_log = ql.log_query

    def capture_log(raw_query, retrieved_chunk_ids):
        logged_calls.append({
            "raw_query": raw_query,
            "chunk_ids": retrieved_chunk_ids,
        })
        return original_log(raw_query, retrieved_chunk_ids)

    monkeypatch.setattr(ql, "log_query", capture_log)

    # Build a minimal fake FastAPI app that returns mixed IDs
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.middleware.query_interceptor import QueryInterceptorMiddleware

    fake_app = FastAPI()
    fake_app.add_middleware(QueryInterceptorMiddleware)

    @fake_app.post("/query/")
    def fake_query(req: dict):
        return {
            "answer": "Test answer",
            "source_chunks": [
                {"chunk_id": "doc_c1",  "text": "raw chunk 1", "metadata": {}, "distance": 0.1},
                {"chunk_id": "sn_001",  "text": "super node",  "metadata": {}, "distance": 0.2},
                {"chunk_id": "doc_c2",  "text": "raw chunk 2", "metadata": {}, "distance": 0.3},
                {"chunk_id": "sn_002",  "text": "super node 2","metadata": {}, "distance": 0.4},
            ],
            "latency_ms": 42.0,
        }

    client = TestClient(fake_app, raise_server_exceptions=True)
    resp = client.post("/query/", json={"query": "What is Niemann-Pick disease?", "top_k": 3})

    assert resp.status_code == 200
    assert len(logged_calls) == 1, "log_query must be called exactly once"

    received_ids = logged_calls[0]["chunk_ids"]
    assert "sn_001" not in received_ids, "SU9: sn_* IDs must be filtered out"
    assert "sn_002" not in received_ids, "SU9: sn_* IDs must be filtered out"
    assert "doc_c1" in received_ids, "doc_* IDs must be kept"
    assert "doc_c2" in received_ids, "doc_* IDs must be kept"
    assert len(received_ids) == 2, "Exactly 2 raw IDs should reach the logger"
