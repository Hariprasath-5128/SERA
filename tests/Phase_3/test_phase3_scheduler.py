"""
tests/Phase_3/test_phase3_scheduler.py

Phase 3 tests for the Synthesis Trigger and pattern finder logic.
Uses the transactional isolation layer (Upgrade U3) to prevent touching the real database.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
import pytest
from unittest.mock import patch, MagicMock

import numpy as np

from app import config
from app.db import sqlite_client
from app.patterns import finder
from app.scheduler.jobs.pattern_finder import scan_and_trigger
from app.synthesis import synthesizer


# --- Helpers ---
def _make_vec(seed: float, dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(int(seed * 1000))
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)

VEC_A = _make_vec(1.0)
VEC_B = _make_vec(2.0)


# --- Fixtures ---
@pytest.fixture()
def isolated_db(tmp_path: Path, monkeypatch):
    """
    U3 Transactional Isolation: Redirect sqlite_client to a fresh DB for each test.
    """
    db_path = tmp_path / "test_rag.db"
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    import app.config as cfg
    import app.db.sqlite_client as sc
    monkeypatch.setattr(cfg, "SQLITE_DB_PATH", db_path)
    monkeypatch.setattr(sc, "SQLITE_DB_PATH", db_path)

    # Initialize tables
    sc.init_db()
    yield sc


@pytest.fixture()
def mock_synthesizer(monkeypatch):
    """
    Mocks the Phase 4 synthesizer stub to return a fake super_node_id.
    """
    mock_run = MagicMock(return_value="sn_test_12345")
    monkeypatch.setattr(synthesizer, "run", mock_run)
    return mock_run


@pytest.fixture()
def mock_chroma(monkeypatch):
    """
    Mocks ChromaClient.get_collection().get() to return static metadata.
    """
    class MockCollection:
        def get(self, ids, include):
            return {"metadatas": [{"updated_at": "2020-01-01T00:00:00Z"}]}
            
    mock_get_col = MagicMock(return_value=MockCollection())
    monkeypatch.setattr("app.db.chroma_client.ChromaClient.get_collection", mock_get_col)
    return mock_get_col


def setup_cluster(
    sqlite_client_mod,
    hit_count: int,
    last_syn_hit_count: int = 0,
    chunk_ids: list = ["doc_1", "doc_2"],
    emb: np.ndarray = VEC_A,
    synthesized: int = 0
):
    """Helper to insert a cluster for testing."""
    cluster_id = sqlite_client_mod.insert_cluster(
        canonical_query="test query",
        embedding=emb,
        chunk_ids=chunk_ids
    )
    # Manually adjust hit_count and synthesized flags
    with sqlite_client_mod.get_connection() as conn:
        conn.execute(
            """
            UPDATE query_clusters 
            SET hit_count = ?, last_synthesized_hit_count = ?, synthesized = ?
            WHERE id = ?
            """,
            (hit_count, last_syn_hit_count, synthesized, cluster_id)
        )
    return cluster_id


# --- Tests ---

def test_pattern_finder_triggers_above_threshold(isolated_db, mock_synthesizer, mock_chroma):
    setup_cluster(isolated_db, hit_count=10, last_syn_hit_count=0)
    
    res = scan_and_trigger()
    print(f"\n[test_pattern_finder_triggers_above_threshold] Result: {res}")
    assert res["triggered"] == 1
    assert mock_synthesizer.call_count == 1


def test_pattern_finder_skips_below_threshold(isolated_db, mock_synthesizer, mock_chroma):
    setup_cluster(isolated_db, hit_count=5, last_syn_hit_count=0)
    
    res = scan_and_trigger()
    print(f"\n[test_pattern_finder_skips_below_threshold] Result: {res}")
    assert res["triggered"] == 0
    assert mock_synthesizer.call_count == 0


def test_pattern_finder_retriggers_after_delta_with_change(isolated_db, mock_synthesizer, mock_chroma):
    # Setup history so it detects a change
    cluster_id = setup_cluster(isolated_db, hit_count=20, last_syn_hit_count=10, synthesized=1)
    
    # Insert history entry with DIFFERENT chunk_ids hash to trigger condition (B)
    isolated_db.insert_super_node_history(
        cluster_id=cluster_id,
        super_node_id="sn_old",
        revision=1,
        summary="Old summary",
        embedding_hash="hash",
        coverage_score=0.9,
        drift_margin=0.0,
        chunk_ids_hash="old_hash_to_force_change",
        centroid_hash="cent_hash",
        centroid=VEC_A.tobytes()
    )
    
    res = scan_and_trigger()
    print(f"\n[test_pattern_finder_retriggers_after_delta_with_change] Result: {res}")
    assert res["triggered"] == 1
    assert mock_synthesizer.call_count == 1


def test_pattern_finder_skips_below_delta(isolated_db, mock_synthesizer, mock_chroma):
    cluster_id = setup_cluster(isolated_db, hit_count=15, last_syn_hit_count=10, synthesized=1)
    
    res = scan_and_trigger()
    print(f"\n[test_pattern_finder_skips_below_delta] Result: {res}")
    assert res["triggered"] == 0


def test_no_double_synthesis(isolated_db, mock_synthesizer, mock_chroma):
    setup_cluster(isolated_db, hit_count=10, last_syn_hit_count=0)
    
    # We call scan twice, but since the first call finishes successfully it marks it synthesized.
    res1 = scan_and_trigger()
    res2 = scan_and_trigger()
    
    print(f"\n[test_no_double_synthesis] Result 1: {res1}")
    print(f"[test_no_double_synthesis] Result 2: {res2}")
    
    assert res1["triggered"] == 1
    assert res2["triggered"] == 0
    assert mock_synthesizer.call_count == 1


def test_scan_and_trigger_releases_lock_on_failure(isolated_db, monkeypatch, mock_chroma):
    cluster_id = setup_cluster(isolated_db, hit_count=10, last_syn_hit_count=0)
    
    mock_run_fail = MagicMock(side_effect=Exception("Synthesis failed intentionally"))
    monkeypatch.setattr(synthesizer, "run", mock_run_fail)
    
    res = scan_and_trigger()
    print(f"\n[test_scan_and_trigger_releases_lock_on_failure] Result: {res}")
    assert res["failed"] == 1
    
    # Lock should be released
    row = isolated_db.get_cluster(cluster_id)
    assert row["synthesizing"] == 0
