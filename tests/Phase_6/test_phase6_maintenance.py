import json
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

# Make app importable from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import config
from app.db import sqlite_client, super_node_store
from app.db.chroma_client import ChromaClient
from app.maintenance.staleness_checker import run_soft_staleness_check
from app.maintenance.decay_scorer import compute_exponential_decay, compute_and_apply_decay
from app.maintenance.hierarchy_merger import merge_similar_nodes


# ===========================================================================
# SU8 tests
# ===========================================================================

def test_staleness_check_marks_stale_not_deletes():
    """Verify that missing raw chunks mark the super-node is_stale=True, but do not delete it."""
    mock_super_coll = MagicMock()
    mock_raw_coll = MagicMock()

    # Setup mock super-nodes collection to return 1 super-node
    mock_super_coll.get.return_value = {
        "ids": ["sn_123"],
        "metadatas": [{
            "type": "super_node",
            "source_chunks": json.dumps(["doc_c1", "doc_c2"]),
            "cluster_id": 42
        }]
    }

    # Setup mock raw collection to return only 1 of the 2 chunks (match_ratio = 0.5)
    mock_raw_coll.get.return_value = {
        "ids": ["doc_c2"]
    }

    with patch("app.db.chroma_client.ChromaClient.get_super_nodes_collection", return_value=mock_super_coll), \
         patch("app.db.chroma_client.ChromaClient.get_collection", return_value=mock_raw_coll), \
         patch("app.db.super_node_store.update_metadata") as mock_update_meta, \
         patch("app.db.sqlite_client.flag_cluster_for_re_synthesis") as mock_flag, \
         patch("app.db.sqlite_client.log_maintenance_event") as mock_log_event:
        
        run_soft_staleness_check()

        # Check update_metadata was called to set is_stale=True
        mock_update_meta.assert_called_once()
        args, kwargs = mock_update_meta.call_args
        assert args[0] == "sn_123"
        assert args[1]["is_stale"] is True

        # Ensure delete was not called
        mock_super_coll.delete.assert_not_called()


def test_staleness_check_queues_re_synthesis():
    """Verify that staleness check queues re-synthesis in SQLite."""
    mock_super_coll = MagicMock()
    mock_raw_coll = MagicMock()

    mock_super_coll.get.return_value = {
        "ids": ["sn_123"],
        "metadatas": [{
            "type": "super_node",
            "source_chunks": json.dumps(["doc_c1"]),
            "cluster_id": 42
        }]
    }

    # Setup mock raw chunks to return empty (match_ratio = 0.0)
    mock_raw_coll.get.return_value = {
        "ids": []
    }

    with patch("app.db.chroma_client.ChromaClient.get_super_nodes_collection", return_value=mock_super_coll), \
         patch("app.db.chroma_client.ChromaClient.get_collection", return_value=mock_raw_coll), \
         patch("app.db.super_node_store.update_metadata"), \
         patch("app.db.sqlite_client.flag_cluster_for_re_synthesis") as mock_flag, \
         patch("app.db.sqlite_client.log_maintenance_event") as mock_log_event:

        run_soft_staleness_check()

        # Verify cluster flagged in SQLite
        mock_flag.assert_called_once_with(42)
        mock_log_event.assert_called_once_with(
            event_type="soft_stale",
            super_node_id="sn_123",
            reason="source_chunk_drift: match_ratio=0.0000"
        )


# ===========================================================================
# SU3 & SU10 tests (Decay)
# ===========================================================================

def test_decay_score_recent_access_survives():
    """Verify recently accessed nodes survive decay (decay_score > threshold)."""
    meta = {
        "last_accessed": datetime.now(timezone.utc).isoformat(),
        "access_count_sqlite": 5
    }
    score = compute_exponential_decay(meta, half_life_days=7)
    assert score > 0.05  # PRUNE_THRESHOLD / DECAY_PRUNE_THRESHOLD


def test_decay_score_old_node_pruned():
    """Verify untouched old nodes decay below threshold."""
    old_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    meta = {
        "last_accessed": old_date,
        "access_count_sqlite": 1
    }
    score = compute_exponential_decay(meta, half_life_days=7)
    assert score < 0.05  # Falls below DECAY_PRUNE_THRESHOLD (0.05)


def test_linear_decay_terminal_trap_absent():
    """Verify that a node created long ago but accessed recently survives (no linear terminal trap)."""
    meta = {
        "last_accessed": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        "access_count_sqlite": 8
    }
    score = compute_exponential_decay(meta, half_life_days=7)
    assert score > 1.0  # High utility due to recent access overrides long age


def test_decay_scorer_uses_sqlite_not_chroma_counter():
    """Verify that decay scorer queries SQLite for true hit counts (SU10) and updates ChromaDB metadata."""
    mock_super_coll = MagicMock()
    mock_super_coll.get.return_value = {
        "ids": ["sn_123"],
        "metadatas": [{
            "type": "super_node",
            "last_accessed": datetime.now(timezone.utc).isoformat(),
            "hit_count": 1,  # Stale ChromaDB value
            "access_count": 1
        }]
    }

    # SQLite returns 50 (authoritative value)
    mock_sqlite_row = {"hit_count": 50}

    with patch("app.db.chroma_client.ChromaClient.get_super_nodes_collection", return_value=mock_super_coll), \
         patch("app.db.sqlite_client.get_cluster_by_super_node_id", return_value=mock_sqlite_row), \
         patch("app.db.super_node_store.update_metadata") as mock_update_meta:

        compute_and_apply_decay()

        # Check metadata update includes synced SQLite hit count (50)
        mock_update_meta.assert_called_once()
        args, kwargs = mock_update_meta.call_args
        assert args[0] == "sn_123"
        assert args[1]["access_count"] == 50
        assert args[1]["hit_count"] == 50
        assert args[1]["decay_score"] > 0.05


# ===========================================================================
# SU7 & SU11 tests (Hierarchy Merger)
# ===========================================================================

def test_merger_skips_nodes_already_in_hierarchy():
    """Verify that already-merged children (in_hierarchy=True) are skipped."""
    mock_super_coll = MagicMock()
    
    # Returns 1 node in hierarchy, 1 not (only 1 eligible -> can't merge)
    mock_super_coll.get.return_value = {
        "ids": ["sn_A", "sn_B"],
        "metadatas": [
            {"type": "super_node", "in_hierarchy": True, "hit_count": 5},
            {"type": "super_node", "in_hierarchy": False, "hit_count": 5}
        ],
        "embeddings": [[0.1, 0.2], [0.15, 0.22]]
    }

    with patch("app.db.chroma_client.ChromaClient.get_super_nodes_collection", return_value=mock_super_coll), \
         patch("app.synthesis.synthesizer.run") as mock_synth_run:
        
        merge_similar_nodes()

        # Synthesis should never run since there aren't two eligible nodes
        mock_synth_run.assert_not_called()


def test_merger_locks_children_after_merge():
    """Verify child nodes are locked (in_hierarchy=True, parent_meta_id set) after a successful merge."""
    mock_super_coll = MagicMock()
    
    # Two identical super-nodes (sim = 1.0)
    mock_super_coll.get.return_value = {
        "ids": ["sn_A", "sn_B"],
        "metadatas": [
            {"type": "super_node", "in_hierarchy": False, "hit_count": 5, "source_chunks": '["doc_1"]', "source_query": "Q1"},
            {"type": "super_node", "in_hierarchy": False, "hit_count": 5, "source_chunks": '["doc_2"]', "source_query": "Q2"}
        ],
        "embeddings": [[0.1, 0.2], [0.1, 0.2]]
    }

    with patch("app.db.chroma_client.ChromaClient.get_super_nodes_collection", return_value=mock_super_coll), \
         patch("app.maintenance.hierarchy_merger.run_synthesis", return_value="mn_123") as mock_synth_run, \
         patch("app.db.super_node_store.get_metadata", return_value={"type": "super_node"}), \
         patch("app.db.super_node_store.update_metadata") as mock_update_meta, \
         patch("app.db.sqlite_client.log_maintenance_event"):
        
        merge_similar_nodes()

        # Synthesis should run once
        mock_synth_run.assert_called_once()

        # Verify child updates to lock them in hierarchy
        calls = mock_update_meta.call_args_list
        # Call 1: updates newly synthesized node to meta_node type
        # Call 2 & 3: updates child nodes sn_A and sn_B
        child_calls = [c for c in calls if c[0][0] in ("sn_A", "sn_B")]
        assert len(child_calls) == 2
        for c in child_calls:
            assert c[0][1]["in_hierarchy"] is True
            assert c[0][1]["parent_meta_id"] == "mn_123"


def test_merger_uses_raw_chunk_ids_not_documents():
    """Verify that merger runs synthesis using union of raw chunk IDs, not prior summaries (SU11)."""
    mock_super_coll = MagicMock()
    
    mock_super_coll.get.return_value = {
        "ids": ["sn_A", "sn_B"],
        "metadatas": [
            {"type": "super_node", "in_hierarchy": False, "hit_count": 5, "source_chunks": '["doc_1", "doc_2"]', "source_query": "Q1"},
            {"type": "super_node", "in_hierarchy": False, "hit_count": 5, "source_chunks": '["doc_2", "doc_3"]', "source_query": "Q2"}
        ],
        "embeddings": [[0.1, 0.2], [0.1, 0.2]]
    }

    with patch("app.db.chroma_client.ChromaClient.get_super_nodes_collection", return_value=mock_super_coll), \
         patch("app.maintenance.hierarchy_merger.run_synthesis", return_value="mn_123") as mock_synth_run, \
         patch("app.db.super_node_store.get_metadata", return_value={"type": "super_node"}), \
         patch("app.db.super_node_store.update_metadata"), \
         patch("app.db.sqlite_client.log_maintenance_event"):
        
        merge_similar_nodes()

        mock_synth_run.assert_called_once()
        job = mock_synth_run.call_args[0][0]
        # Verify the chunks unioned: doc_1, doc_2, doc_3
        assert set(job.chunk_ids) == {"doc_1", "doc_2", "doc_3"}
        assert job.cluster_id is None  # Meta-nodes are cluster-less
        assert all(cid.startswith("doc_") for cid in job.chunk_ids)


def test_merger_enforces_depth_cap():
    """Verify that merges exceeding MAX_LINEAGE_DEPTH are blocked and queued for manual review (SU11)."""
    mock_super_coll = MagicMock()
    
    # Both super-nodes are already at lineage_depth=2 (MAX_LINEAGE_DEPTH = 2)
    mock_super_coll.get.return_value = {
        "ids": ["sn_A", "sn_B"],
        "metadatas": [
            {"type": "super_node", "in_hierarchy": False, "hit_count": 5, "lineage_depth": 2, "source_chunks": '["doc_1"]', "source_query": "Q1"},
            {"type": "super_node", "in_hierarchy": False, "hit_count": 5, "lineage_depth": 2, "source_chunks": '["doc_2"]', "source_query": "Q2"}
        ],
        "embeddings": [[0.1, 0.2], [0.1, 0.2]]
    }

    with patch("app.db.chroma_client.ChromaClient.get_super_nodes_collection", return_value=mock_super_coll), \
         patch("app.synthesis.synthesizer.run") as mock_synth_run, \
         patch("app.db.sqlite_client.flag_for_manual_review") as mock_flag_review, \
         patch("app.db.sqlite_client.log_maintenance_event") as mock_log_event:
        
        merge_similar_nodes()

        # Synthesis should NOT run
        mock_synth_run.assert_not_called()

        # Verify routed to manual review queue
        mock_flag_review.assert_called_once_with(
            "sn_A", "sn_B",
            reason="lineage_depth_cap: would reach depth 3 > cap 2"
        )
        mock_log_event.assert_called_once_with(
            event_type="depth_cap_refused",
            super_node_id="sn_A",
            reason="Hierarchy merge depth limit reached (3 > 2) for pair sn_A and sn_B"
        )
