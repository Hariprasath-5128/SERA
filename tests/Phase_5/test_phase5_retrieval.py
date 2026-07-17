"""
tests/Phase_5/test_phase5_retrieval.py
----------------------------------------
Phase 5 test suite -- Preferential Retrieval.

Coverage
--------
  T01  score_booster: r > epsilon -> multiplier = 0.85, mode = "exploit"
  T02  score_booster: r <= epsilon -> multiplier = 1.0, mode = "explore"
  T03  router.query_super_nodes: collection unavailable -> returns []
  T04  router.query_super_nodes: collection empty -> returns []
  T05  router.query_super_nodes: count < top_k -> clamps n_results
  T06  retriever.search: blank query -> returns []
  T07  retriever.search: no super_nodes -> returns raw_chunks only, type=raw_chunk
  T08  SU5 masking: meta_node retrieved -> its children IDs excluded from candidates
  T09  SU5 masking: no meta_node -> nothing masked, all super-node candidates present
  T10  SU4 exploit: super_node dist 0.40 * 0.85 = 0.34 beats raw_chunk dist 0.35
  T11  SU4 explore: super_node dist 0.40 * 1.0 = 0.40 loses to raw_chunk dist 0.35
  T12  SU10: super_node in top_k -> log_super_node_access called with correct ID
  T13  SU10: no super_node in top_k -> log_super_node_access NOT called
  T14  SU10: log_super_node_access failure -> query still returns results (non-fatal)
  T15  log_super_node_access: empty list -> no DB call (no-op guard)
  T16  log_super_node_access: atomic UPDATE called with correct placeholders
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

# Make app importable from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import config
from app.retrieval.score_booster import compute_boost
from app.retrieval.router import query_super_nodes, query_raw_chunks
from app.retrieval.retriever import Retriever
from app.db.sqlite_client import log_super_node_access


# ===========================================================================
# Fixtures / helpers
# ===========================================================================

def _make_sn_candidate(id_: str, dist: float, node_type: str = "super_node",
                        children: list = None) -> dict:
    """Build a super-node candidate dict as returned by router.query_super_nodes."""
    meta = {"type": node_type}
    if children is not None:
        meta["children"] = json.dumps(children)
    return {"id": id_, "doc": f"doc text for {id_}", "meta": meta, "distance": dist}


def _make_raw_candidate(id_: str, dist: float) -> dict:
    """Build a raw-chunk candidate dict as returned by router.query_raw_chunks."""
    return {"id": id_, "doc": f"raw doc {id_}", "meta": {"section": "INFO"}, "distance": dist}


def _mock_embedder(monkeypatch):
    """Patch Embedder so no actual model is loaded during tests."""
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    monkeypatch.setattr("app.retrieval.retriever.Embedder.get_model", lambda: fake_model)
    return fake_model


# ===========================================================================
# T01-T02 -- score_booster
# ===========================================================================

class TestScoreBooster:

    def test_T01_exploit_mode(self):
        """r > epsilon -> multiplier = SUPER_NODE_SCORE_MULTIPLIER, mode = exploit."""
        with patch("app.retrieval.score_booster.random.random", return_value=0.95):
            mult, mode = compute_boost(epsilon=0.10, multiplier=0.85)
        assert mult == pytest.approx(0.85)
        assert mode == "exploit"

    def test_T02_explore_mode(self):
        """r <= epsilon -> multiplier = 1.0, mode = explore."""
        with patch("app.retrieval.score_booster.random.random", return_value=0.05):
            mult, mode = compute_boost(epsilon=0.10, multiplier=0.85)
        assert mult == pytest.approx(1.0)
        assert mode == "explore"

    def test_T01b_boundary_exactly_epsilon_is_explore(self):
        """r == epsilon (not strictly greater) -> explore mode."""
        with patch("app.retrieval.score_booster.random.random", return_value=0.10):
            mult, mode = compute_boost(epsilon=0.10, multiplier=0.85)
        assert mode == "explore"
        assert mult == pytest.approx(1.0)


# ===========================================================================
# T03-T05 -- router.query_super_nodes
# ===========================================================================

class TestRouter:

    def test_T03_super_nodes_collection_unavailable(self):
        """get_super_nodes_collection() returns None -> [] (graceful fallback)."""
        with patch("app.retrieval.router.ChromaClient.get_super_nodes_collection",
                   return_value=None):
            result = query_super_nodes([0.1, 0.2], top_k=3)
        assert result == []

    def test_T04_super_nodes_collection_empty(self):
        """Collection exists but count == 0 -> [] (graceful fallback)."""
        fake_col = MagicMock()
        fake_col.count.return_value = 0
        with patch("app.retrieval.router.ChromaClient.get_super_nodes_collection",
                   return_value=fake_col):
            result = query_super_nodes([0.1, 0.2], top_k=3)
        assert result == []
        fake_col.query.assert_not_called()

    def test_T05_super_nodes_clamps_n_results(self):
        """count=2 < top_k=5 -> query is called with n_results=2, not 5."""
        fake_col = MagicMock()
        fake_col.count.return_value = 2
        fake_col.query.return_value = {
            "ids": [["sn_a", "sn_b"]],
            "documents": [["doc a", "doc b"]],
            "metadatas": [[{"type": "super_node"}, {"type": "super_node"}]],
            "distances": [[0.2, 0.4]],
        }
        with patch("app.retrieval.router.ChromaClient.get_super_nodes_collection",
                   return_value=fake_col):
            result = query_super_nodes([0.1], top_k=5)
        # n_results must be clamped to 2
        called_kwargs = fake_col.query.call_args.kwargs
        assert called_kwargs["n_results"] == 2
        assert len(result) == 2


# ===========================================================================
# T06-T07 -- retriever.search basic cases
# ===========================================================================

class TestRetrieverBasic:

    def test_T06_blank_query_returns_empty(self):
        """Empty / whitespace query -> []."""
        assert Retriever.search("   ") == []
        assert Retriever.search("") == []

    def test_T07_no_super_nodes_returns_raw_chunks_only(self, monkeypatch):
        """When super_nodes unavailable, results contain only raw_chunk type."""
        _mock_embedder(monkeypatch)

        raw = [_make_raw_candidate("doc_1", 0.3), _make_raw_candidate("doc_2", 0.5)]

        with patch("app.retrieval.retriever.query_super_nodes", return_value=[]),              patch("app.retrieval.retriever.query_raw_chunks", return_value=raw),              patch("app.retrieval.retriever.compute_boost", return_value=(0.85, "exploit")),              patch("app.retrieval.retriever.sqlite_client.log_super_node_access") as mock_log:

            results = Retriever.search("test query", top_k=5)

        assert len(results) == 2
        assert all(r["type"] == "raw_chunk" for r in results)
        mock_log.assert_not_called()


# ===========================================================================
# T08-T09 -- SU5 Hierarchical Masking
# ===========================================================================

class TestSU5HierarchicalMasking:

    def test_T08_meta_node_children_are_masked(self, monkeypatch):
        """
        meta_node with children=[sn_child] retrieved -> sn_child excluded.
        Slot filled by raw_chunk candidate.
        """
        _mock_embedder(monkeypatch)

        sn_meta = _make_sn_candidate("mn_001", dist=0.1, node_type="meta_node",
                                      children=["sn_child"])
        sn_child = _make_sn_candidate("sn_child", dist=0.15, node_type="super_node")
        raw = [_make_raw_candidate("doc_fill", 0.5)]

        with patch("app.retrieval.retriever.query_super_nodes",
                   return_value=[sn_meta, sn_child]),              patch("app.retrieval.retriever.query_raw_chunks", return_value=raw),              patch("app.retrieval.retriever.compute_boost",
                   return_value=(1.0, "explore")),              patch("app.retrieval.retriever.sqlite_client.log_super_node_access"):

            results = Retriever.search("mask test", top_k=5)

        result_ids = [r["id"] for r in results]
        assert "sn_child" not in result_ids, "SU5 should have excluded the child node"
        assert "mn_001" in result_ids, "Parent meta_node should still be present"
        assert "doc_fill" in result_ids, "Raw chunk should fill the vacated slot"

    def test_T09_no_meta_node_nothing_masked(self, monkeypatch):
        """No meta_node in results -> masked_ids is empty -> all candidates present."""
        _mock_embedder(monkeypatch)

        sn1 = _make_sn_candidate("sn_001", dist=0.2, node_type="super_node")
        sn2 = _make_sn_candidate("sn_002", dist=0.3, node_type="super_node")
        raw = [_make_raw_candidate("doc_1", 0.4)]

        with patch("app.retrieval.retriever.query_super_nodes",
                   return_value=[sn1, sn2]),              patch("app.retrieval.retriever.query_raw_chunks", return_value=raw),              patch("app.retrieval.retriever.compute_boost",
                   return_value=(0.85, "exploit")),              patch("app.retrieval.retriever.sqlite_client.log_super_node_access"):

            results = Retriever.search("no mask test", top_k=5)

        result_ids = [r["id"] for r in results]
        assert "sn_001" in result_ids
        assert "sn_002" in result_ids
        assert "doc_1" in result_ids


# ===========================================================================
# T10-T11 -- SU4 epsilon-Greedy Score Boost
# ===========================================================================

class TestSU4EpsilonGreedy:

    def test_T10_exploit_super_node_beats_raw_chunk(self, monkeypatch):
        """
        Exploit mode: sn dist 0.40 * 0.85 = 0.34 < raw dist 0.35
        -> super_node ranked first.
        """
        _mock_embedder(monkeypatch)

        sn = _make_sn_candidate("sn_top", dist=0.40, node_type="super_node")
        raw = [_make_raw_candidate("doc_below", dist=0.35)]

        with patch("app.retrieval.retriever.query_super_nodes", return_value=[sn]),              patch("app.retrieval.retriever.query_raw_chunks", return_value=raw),              patch("app.retrieval.retriever.compute_boost",
                   return_value=(0.85, "exploit")),              patch("app.retrieval.retriever.sqlite_client.log_super_node_access"):

            results = Retriever.search("boost test", top_k=2)

        assert results[0]["id"] == "sn_top", (
            f"Super-node should rank first in exploit mode; got {results[0]['id']}"
        )
        assert results[0]["distance"] == pytest.approx(0.40 * 0.85)

    def test_T11_explore_raw_chunk_beats_super_node(self, monkeypatch):
        """
        Explore mode: sn dist 0.40 * 1.0 = 0.40 > raw dist 0.35
        -> raw_chunk ranked first.
        """
        _mock_embedder(monkeypatch)

        sn = _make_sn_candidate("sn_second", dist=0.40, node_type="super_node")
        raw = [_make_raw_candidate("doc_first", dist=0.35)]

        with patch("app.retrieval.retriever.query_super_nodes", return_value=[sn]),              patch("app.retrieval.retriever.query_raw_chunks", return_value=raw),              patch("app.retrieval.retriever.compute_boost",
                   return_value=(1.0, "explore")),              patch("app.retrieval.retriever.sqlite_client.log_super_node_access"):

            results = Retriever.search("explore test", top_k=2)

        assert results[0]["id"] == "doc_first", (
            f"Raw chunk should rank first in explore mode; got {results[0]['id']}"
        )
        assert results[0]["distance"] == pytest.approx(0.35)


# ===========================================================================
# T12-T14 -- SU10 Atomic SQLite Access Logging
# ===========================================================================

class TestSU10AtomicLogging:

    def test_T12_super_node_in_top_k_triggers_log(self, monkeypatch):
        """super_node survives into top-k -> log_super_node_access called with its ID."""
        _mock_embedder(monkeypatch)

        sn = _make_sn_candidate("sn_logged", dist=0.1, node_type="super_node")
        raw = [_make_raw_candidate("doc_1", dist=0.9)]

        with patch("app.retrieval.retriever.query_super_nodes", return_value=[sn]),              patch("app.retrieval.retriever.query_raw_chunks", return_value=raw),              patch("app.retrieval.retriever.compute_boost",
                   return_value=(0.85, "exploit")),              patch("app.retrieval.retriever.sqlite_client.log_super_node_access") as mock_log:

            Retriever.search("log test", top_k=2)

        mock_log.assert_called_once_with(["sn_logged"])

    def test_T13_no_super_node_in_top_k_no_log(self, monkeypatch):
        """Only raw_chunks in top-k -> log_super_node_access NOT called."""
        _mock_embedder(monkeypatch)

        raw = [_make_raw_candidate("doc_1", dist=0.2)]

        with patch("app.retrieval.retriever.query_super_nodes", return_value=[]),              patch("app.retrieval.retriever.query_raw_chunks", return_value=raw),              patch("app.retrieval.retriever.compute_boost",
                   return_value=(0.85, "exploit")),              patch("app.retrieval.retriever.sqlite_client.log_super_node_access") as mock_log:

            results = Retriever.search("no log test", top_k=2)

        mock_log.assert_not_called()

    def test_T14_su10_failure_is_non_fatal(self, monkeypatch):
        """
        If log_super_node_access raises an exception, the query still
        returns results -- SU10 failure must never surface to the caller.
        """
        _mock_embedder(monkeypatch)

        sn = _make_sn_candidate("sn_crash", dist=0.1, node_type="super_node")
        raw = [_make_raw_candidate("doc_safe", dist=0.5)]

        with patch("app.retrieval.retriever.query_super_nodes", return_value=[sn]),              patch("app.retrieval.retriever.query_raw_chunks", return_value=raw),              patch("app.retrieval.retriever.compute_boost",
                   return_value=(0.85, "exploit")),              patch("app.retrieval.retriever.sqlite_client.log_super_node_access",
                   side_effect=RuntimeError("DB crash")):

            # Must not raise
            results = Retriever.search("crash test", top_k=2)

        assert len(results) == 2, "Results should still be returned despite SU10 failure"


# ===========================================================================
# T15-T16 -- log_super_node_access unit tests
# ===========================================================================

class TestLogSuperNodeAccess:

    def test_T15_empty_list_is_noop(self):
        """Empty list -> function returns immediately without touching DB."""
        with patch("app.db.sqlite_client.get_connection") as mock_conn:
            log_super_node_access([])
        mock_conn.assert_not_called()

    def test_T16_correct_sql_placeholders(self):
        """3 super_node_ids -> UPDATE with 3 placeholders and correct args."""
        ids = ["sn_a", "sn_b", "sn_c"]
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_cursor

        with patch("app.db.sqlite_client.get_connection", return_value=mock_conn):
            log_super_node_access(ids)

        call_args = mock_conn.execute.call_args
        sql: str = call_args[0][0]
        params = call_args[0][1]

        assert "UPDATE query_clusters" in sql
        assert "hit_count = hit_count + 1" in sql
        assert sql.count("?") == 3, f"Expected 3 placeholders, got: {sql}"
        assert params == ["sn_a", "sn_b", "sn_c"]
