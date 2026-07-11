import sqlite3
import json
import logging
from contextlib import contextmanager
from typing import Optional

import numpy as np

from app.config import SQLITE_DB_PATH, BASE_DIR

logger = logging.getLogger(__name__)


@contextmanager
def get_connection():
    """
    Context-manager that yields a WAL-mode SQLite connection and always
    commits + closes it, even on exception.
    """
    conn = sqlite3.connect(str(SQLITE_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        # Enable WAL mode for concurrent read + write access
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

def init_db():
    """
    Create all tables if they don't exist yet.

    Tables created:
      - benchmark_qa       (Phase 1 — unchanged)
      - query_clusters     (Phase 2 — cluster centroids + hit counters)
      - query_log          (Phase 2 — per-query audit trail)

    Safe to call multiple times (CREATE TABLE IF NOT EXISTS).
    """
    with get_connection() as conn:
        # ── Phase 1 ────────────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_qa (
                id          TEXT PRIMARY KEY,
                document_id TEXT,
                question    TEXT,
                answer      TEXT,
                source_url  TEXT
            )
        """)

        # ── Phase 2 — query_clusters ───────────────────────────────────────
        # Each row is a semantic cluster of similar user queries.
        # query_embedding: BLOB — running centroid (float32, little-endian).
        # chunk_ids:       TEXT — JSON list of raw doc_ chunk IDs (SU9 guarantees only raw IDs).
        # synthesizing:    INTEGER — concurrency lock (SU12): 1 while Phase 4 is running.
        # last_synthesized_hit_count: INTEGER — SU12 delta gate for re-synthesis.
        conn.execute("""
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
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_qc_hit_count
                ON query_clusters(hit_count DESC, synthesized, synthesizing)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_qc_re_synthesis
                ON query_clusters(pending_re_synthesis, synthesized)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_qc_resynth_delta
                ON query_clusters(hit_count, last_synthesized_hit_count, synthesizing)
        """)

        # ── Phase 2 — query_log ────────────────────────────────────────────
        # Per-query audit trail: every incoming query gets one row.
        # retrieved_chunks: JSON list of raw doc_ IDs that were returned
        #                   (super-node IDs have already been filtered by SU9 middleware).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_query        TEXT    NOT NULL,
                cluster_id       INTEGER REFERENCES query_clusters(id),
                retrieved_chunks TEXT    NOT NULL DEFAULT '[]',
                matched_existing INTEGER NOT NULL DEFAULT 0,
                timestamp        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ql_cluster
                ON query_log(cluster_id, timestamp DESC)
        """)

        # ── Phase 3 — super_node_history migration ───────────────────────────
        migration_path = BASE_DIR / "app" / "db" / "migrations" / "002_super_node_history.sql"
        if migration_path.exists():
            with open(migration_path, "r", encoding="utf-8") as f:
                conn.executescript(f.read())

    logger.info("sqlite_client.init_db: all tables ready")


# ---------------------------------------------------------------------------
# benchmark_qa helpers (Phase 1 — unchanged API)
# ---------------------------------------------------------------------------

def save_benchmark(qa_id: str, document_id: str, question: str,
                   answer: str, source_url: str) -> None:
    """Save a single Q&A pair to the benchmark table."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO benchmark_qa
                (id, document_id, question, answer, source_url)
            VALUES (?, ?, ?, ?, ?)
            """,
            (qa_id, document_id, question, answer, source_url),
        )


# ---------------------------------------------------------------------------
# query_clusters CRUD (Phase 2)
# ---------------------------------------------------------------------------

def _emb_to_blob(embedding: np.ndarray) -> bytes:
    """Serialize a float32 numpy array to raw bytes for BLOB storage."""
    return embedding.astype(np.float32).tobytes()


def _blob_to_emb(blob: bytes, dim: int) -> np.ndarray:
    """Deserialize a BLOB back to a float32 numpy array."""
    return np.frombuffer(blob, dtype=np.float32).reshape(dim)


def insert_cluster(
    canonical_query: str,
    embedding: np.ndarray,
    chunk_ids: list[str],
) -> int:
    """
    INSERT a new cluster row and return its auto-assigned id.

    chunk_ids must contain only raw doc_ IDs (enforced by SU9 middleware
    before this function is called).
    """
    blob = _emb_to_blob(embedding)
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO query_clusters
                (canonical_query, query_embedding, chunk_ids)
            VALUES (?, ?, ?)
            """,
            (canonical_query, blob, json.dumps(chunk_ids)),
        )
        return cur.lastrowid


def get_cluster(cluster_id: int) -> Optional[sqlite3.Row]:
    """Fetch a single cluster row by primary key. Returns None if not found."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM query_clusters WHERE id = ?", (cluster_id,)
        ).fetchone()
    return row


def fetch_all_clusters() -> list[sqlite3.Row]:
    """
    Return every cluster row.

    Used by query_logger for the cosine similarity scan.
    Safe at <10k clusters; replace with FAISS index for larger scale.
    """
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM query_clusters").fetchall()
    return rows


def update_cluster(
    cluster_id: int,
    new_chunk_ids: list[str],
) -> None:
    """
    Increment hit_count, merge chunk_ids, refresh last_hit timestamp.

    new_chunk_ids is the de-duplicated union of the existing list and the
    incoming raw chunk IDs for this query.
    """
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE query_clusters
            SET
                hit_count  = hit_count + 1,
                chunk_ids  = ?,
                last_hit   = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (json.dumps(new_chunk_ids), cluster_id),
        )


def update_cluster_embedding(cluster_id: int, new_embedding: np.ndarray) -> None:
    """
    SU6 — Running Centroid Update.

    Replaces the stored query_embedding BLOB with the newly computed
    weighted-average centroid.  Called by query_logger BEFORE incrementing
    hit_count so that `n` in ((old * n) + new) / (n+1) is correct.
    """
    blob = _emb_to_blob(new_embedding)
    with get_connection() as conn:
        conn.execute(
            "UPDATE query_clusters SET query_embedding = ? WHERE id = ?",
            (blob, cluster_id),
        )


def get_ready_clusters(
    min_hit_count: int,
    resynth_delta: int,
) -> list[sqlite3.Row]:
    """
    SU12 — Re-Synthesis Threshold Counter.

    Returns clusters that are eligible for (re-)synthesis:
      - Not currently synthesizing (lock not held)
      - hit_count has reached the minimum threshold
      - Additional hits since last synthesis >= resynth_delta

    Used by Phase 3 pattern_finder.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM query_clusters
            WHERE synthesizing = 0
              AND synthesis_failed = 0
              AND hit_count >= ?
              AND (hit_count - last_synthesized_hit_count) >= ?
            ORDER BY hit_count DESC
            """,
            (min_hit_count, resynth_delta),
        ).fetchall()
    return rows


def mark_cluster_synthesized(cluster_id: int, super_node_id: str) -> None:
    """
    Called by Phase 4 synthesizer on success.

    Sets synthesized=1, clears synthesizing lock, records super_node_id,
    and snapshots last_synthesized_hit_count = current hit_count (SU12).
    """
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE query_clusters
            SET
                synthesized                = 1,
                synthesizing               = 0,
                super_node_id              = ?,
                last_synthesized_hit_count = hit_count,
                pending_re_synthesis       = 0
            WHERE id = ?
            """,
            (super_node_id, cluster_id),
        )


def set_synthesizing(cluster_id: int, value: bool) -> None:
    """
    SU12 concurrency lock.

    Set synthesizing=1 atomically BEFORE calling the synthesizer to prevent
    double-synthesis on overlapping scheduler ticks.  Set to 0 on failure.
    """
    with get_connection() as conn:
        conn.execute(
            "UPDATE query_clusters SET synthesizing = ? WHERE id = ?",
            (1 if value else 0, cluster_id),
        )


def flag_cluster_for_re_synthesis(cluster_id: int) -> None:
    """
    SU8 / SU13 — mark a cluster as needing re-synthesis.

    Called by:
      - staleness_checker (SU8): source chunk drifted
      - synthesizer (SU13): child node was revised → parent marked stale
    """
    with get_connection() as conn:
        conn.execute(
            "UPDATE query_clusters SET pending_re_synthesis = 1 WHERE id = ?",
            (cluster_id,),
        )


def get_cluster_by_super_node_id(super_node_id: str) -> Optional[sqlite3.Row]:
    """
    SU10 — retrieve a cluster by its associated super_node_id.

    Used by decay_scorer to read the canonical hit_count from SQLite
    (authoritative source) rather than trusting the ChromaDB access_count
    mirror.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM query_clusters WHERE super_node_id = ?",
            (super_node_id,),
        ).fetchone()
    return row


# ---------------------------------------------------------------------------
# query_log helpers (Phase 2)
# ---------------------------------------------------------------------------

def insert_query_log(
    raw_query: str,
    cluster_id: Optional[int],
    retrieved_chunks: list[str],
    matched_existing: bool,
) -> None:
    """
    Write one row to the query_log audit trail.

    retrieved_chunks contains only raw doc_ IDs at this point —
    SU9 middleware has already filtered out any sn_* IDs.
    """
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO query_log
                (raw_query, cluster_id, retrieved_chunks, matched_existing)
            VALUES (?, ?, ?, ?)
            """,
            (
                raw_query,
                cluster_id,
                json.dumps(retrieved_chunks),
                1 if matched_existing else 0,
            ),
        )

# ---------------------------------------------------------------------------
# super_node_history helpers (Phase 3)
# ---------------------------------------------------------------------------

def insert_super_node_history(
    cluster_id: int,
    super_node_id: str,
    revision: int,
    summary: str,
    embedding_hash: str,
    coverage_score: float,
    drift_margin: float,
    chunk_ids_hash: str,
    centroid_hash: str,
    centroid: bytes,
) -> None:
    """
    U1 — Record a full revision of a super-node synthesis to SQLite.
    This provides an immutable audit trail of every generation step.
    """
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO super_node_history
                (cluster_id, super_node_id, revision, summary, embedding_hash,
                 coverage_score, drift_margin, chunk_ids_hash, centroid_hash, centroid)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cluster_id, super_node_id, revision, summary, embedding_hash,
                coverage_score, drift_margin, chunk_ids_hash, centroid_hash, centroid
            ),
        )

def get_latest_history_entry(cluster_id: int) -> Optional[sqlite3.Row]:
    """
    U2 — Fetch the most recent synthesis history entry for a cluster.
    Used by Phase 3 pattern_finder to check if chunk_ids or centroid shifted.
    """
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT * FROM super_node_history
            WHERE cluster_id = ?
            ORDER BY revision DESC
            LIMIT 1
            """,
            (cluster_id,),
        ).fetchone()
    return row

def get_super_node_history(cluster_id: int) -> list[sqlite3.Row]:
    """
    Fetch all historical revisions of a super-node, descending by revision.
    Useful for audit logs and rollback evaluation.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM super_node_history
            WHERE cluster_id = ?
            ORDER BY revision DESC
            """,
            (cluster_id,),
        ).fetchall()
    return rows

# ---------------------------------------------------------------------------
# Backup & Reset (Admin)
# ---------------------------------------------------------------------------

def backup_database(backup_dir: str) -> None:
    """
    Safely backup the SQLite database to the specified directory.
    Uses SQLite's online backup API to ensure a consistent snapshot
    even if writers are active in WAL mode.
    """
    import os
    import shutil
    from pathlib import Path
    
    os.makedirs(backup_dir, exist_ok=True)
    dest_path = Path(backup_dir) / "rag.db"
    
    logger.info("sqlite_client: backing up database to %s", dest_path)
    
    # SQLite backup API handles the WAL + lock concurrency safely.
    with get_connection() as src_conn:
        # We need a raw sqlite3 connection for the destination
        dst_conn = sqlite3.connect(dest_path)
        with dst_conn:
            src_conn.backup(dst_conn)
        dst_conn.close()


def reset_to_ground_truth() -> None:
    """
    Wipe all learned data (Phase 2+) while retaining Phase 1 (benchmark_qa).
    This truncates query_clusters, query_log, maintenance_log, and manual_review_queue.
    """
    logger.warning("sqlite_client: RESETTING TO GROUND TRUTH. Wiping all clusters and logs.")
    with get_connection() as conn:
        for table in ["query_log", "query_clusters", "maintenance_log", "manual_review_queue"]:
            try:
                conn.execute(f"DELETE FROM {table}")
            except sqlite3.OperationalError:
                pass # Table might not be created yet if early in phases
        
        # Reset auto-increment sequences
        try:
            conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('query_log', 'query_clusters', 'maintenance_log', 'manual_review_queue')")
        except sqlite3.OperationalError:
            pass
