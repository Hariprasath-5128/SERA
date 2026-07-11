-- =============================================================================
-- SERA — SQLite Schema  (Migration 001: Initial)
-- =============================================================================
-- Applies to: data/sqlite/rag.db
-- Run order:  1 (initial, no dependencies)
-- Safe:       All statements use IF NOT EXISTS / CREATE INDEX IF NOT EXISTS
-- =============================================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- -----------------------------------------------------------------------------
-- Phase 1 — Benchmark Q&A pairs
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS benchmark_qa (
    id           TEXT PRIMARY KEY,
    document_id  TEXT,
    question     TEXT,
    answer       TEXT,
    source_url   TEXT
);

-- -----------------------------------------------------------------------------
-- Phase 2 — Query Clusters
--
-- Each row represents a semantic cluster of similar user queries.
--
-- query_embedding              BLOB   float32 little-endian centroid (SU6 running average)
-- chunk_ids                    TEXT   JSON list of raw doc_ chunk IDs — sn_* IDs never stored (SU9)
-- synthesized                  INT    Legacy flag; kept for migration compatibility
-- synthesizing                 INT    Concurrency lock — set to 1 before calling Phase 4 (SU12)
-- pending_re_synthesis         INT    Set by staleness_checker (SU8) or synthesizer (SU13)
-- last_synthesized_hit_count   INT    Snapshot of hit_count at last synthesis — enables SU12 delta gate
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS query_clusters (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_query             TEXT    NOT NULL,
    query_embedding             BLOB    NOT NULL,
    hit_count                   INTEGER NOT NULL DEFAULT 1,
    chunk_ids                   TEXT    NOT NULL DEFAULT '[]',
    first_seen                  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_hit                    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    synthesized                 INTEGER NOT NULL DEFAULT 0,
    synthesizing                INTEGER NOT NULL DEFAULT 0,
    synthesis_failed            INTEGER NOT NULL DEFAULT 0,
    super_node_id               TEXT,
    retry_count                 INTEGER NOT NULL DEFAULT 0,
    pending_re_synthesis        INTEGER NOT NULL DEFAULT 0,
    last_synthesized_hit_count  INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_qc_hit_count
    ON query_clusters(hit_count DESC, synthesized, synthesizing);

-- SU8 / SU13: re-synthesis queue scan
CREATE INDEX IF NOT EXISTS idx_qc_re_synthesis
    ON query_clusters(pending_re_synthesis, synthesized);

-- SU12: delta-gate scan — (hit_count - last_synthesized_hit_count) >= RESYNTH_DELTA
CREATE INDEX IF NOT EXISTS idx_qc_resynth_delta
    ON query_clusters(hit_count, last_synthesized_hit_count, synthesizing);

-- -----------------------------------------------------------------------------
-- Phase 2 — Query Log  (per-query audit trail)
--
-- retrieved_chunks  TEXT  JSON list of raw doc_ IDs only —
--                         super-node IDs already stripped by SU9 middleware
-- matched_existing  INT   1 if this query merged into an existing cluster; 0 if new
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS query_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_query        TEXT    NOT NULL,
    cluster_id       INTEGER REFERENCES query_clusters(id),
    retrieved_chunks TEXT    NOT NULL DEFAULT '[]',
    matched_existing INTEGER NOT NULL DEFAULT 0,
    timestamp        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ql_cluster
    ON query_log(cluster_id, timestamp DESC);

-- -----------------------------------------------------------------------------
-- Phase 6 — Maintenance Log  (pruning, merging, staleness events)
--
-- event_type values:
--   'prune'                  decay scorer pruned a dead node
--   'soft_stale'             staleness checker flagged a node
--   're_synthesis_queued'    cluster queued for re-synthesis
--   'depth_cap_refused'      SU11: merger refused because MAX_LINEAGE_DEPTH exceeded
--   'manual_review_flagged'  SU11: pair written to manual_review_queue instead
-- reason examples:
--   'source_chunk_drift'     SU8: upstream chunk changed
--   'child_revised'          SU13: child super-node was re-synthesized
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS maintenance_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type    TEXT    NOT NULL,
    super_node_id TEXT,
    reason        TEXT,
    decay_score   REAL,
    timestamp     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------------------------------
-- Phase 6 — Manual Review Queue  (SU11 depth-cap violations)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS manual_review_queue (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    sn_id_a    TEXT    NOT NULL,
    sn_id_b    TEXT    NOT NULL,
    reason     TEXT    NOT NULL,
    flagged_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved   INTEGER NOT NULL DEFAULT 0
);
