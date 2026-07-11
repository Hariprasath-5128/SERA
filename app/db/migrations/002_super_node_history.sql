CREATE TABLE IF NOT EXISTS super_node_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id      INTEGER NOT NULL REFERENCES query_clusters(id),
    super_node_id   TEXT    NOT NULL,          -- e.g. "sn_42_1720000000"
    revision        INTEGER NOT NULL,          -- 1, 2, 3 ... incremented each re-synthesis
    summary         TEXT    NOT NULL,          -- full synthesis text at this revision
    embedding_hash  TEXT    NOT NULL,          -- SHA-256 of the embedding bytes (for diff checks)
    coverage_score  REAL    NOT NULL DEFAULT 0.0,  -- from LLM-as-judge validator (Phase 4)
    drift_margin    REAL    NOT NULL DEFAULT 0.0,  -- SU14: sim_summary_to_query - sim_raw_to_query_max
    chunk_ids_hash  TEXT    NOT NULL,          -- SHA-256 of sorted JSON chunk_ids (change detection)
    centroid_hash   TEXT    NOT NULL,          -- SHA-256 of centroid embedding bytes (change detection)
    centroid        BLOB    NOT NULL,          -- Centroid embedding bytes (for cosine sim check)
    timestamp       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_snh_cluster_revision
    ON super_node_history(cluster_id, revision DESC);

CREATE INDEX IF NOT EXISTS idx_snh_super_node
    ON super_node_history(super_node_id);
