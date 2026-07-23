from datetime import datetime, timezone
from pathlib import Path
import json
import logging
import os
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app.config import DATA_DIR
from app.db import sqlite_client, super_node_store
from app.db.chroma_client import ChromaClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])

class BackupRequest(BaseModel):
    version: str = Field("v1", description="Version name (e.g., v1, v2). Use 'v1' to represent the ground truth clean state.")
    message: str = Field(..., description="Commit message summarizing the state of this backup.")

class ResetRequest(BaseModel):
    commit_message: str = Field(..., description="Message summarizing why the reset was triggered. This creates an automatic pre-reset backup.")

class ActionResponse(BaseModel):
    status: str
    message: str
    backup_path: str | None = None


class ConfigUpdateRequest(BaseModel):
    HIERARCHY_MERGE_THRESHOLD: float


@router.post("/backup", response_model=ActionResponse)
def trigger_backup(req: BackupRequest):
    """
    Creates a snapshot backup of both the SQLite database and ChromaDB vectors.
    Saves them in data/backups/{version}/ along with a commit.json manifest.
    """
    try:
        backup_dir = DATA_DIR / "backups" / req.version
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Backup SQLite
        sqlite_client.backup_database(str(backup_dir))
        
        # 2. Backup ChromaDB
        ChromaClient.backup_chroma(str(backup_dir))
        
        # 3. Write commit.json
        commit_data = {
            "version": req.version,
            "message": req.message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with open(backup_dir / "commit.json", "w") as f:
            json.dump(commit_data, f, indent=2)
            
        return ActionResponse(
            status="success",
            message=f"Backup successfully created for version '{req.version}'",
            backup_path=str(backup_dir)
        )
        
    except Exception as e:
        logger.error(f"Backup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Backup failed: {str(e)}")


@router.post("/reset", response_model=ActionResponse)
def trigger_reset(req: ResetRequest):
    """
    Resets the entire RAG system back to the initial ground truth state.
    Wipes all query clusters, logs, and generated super-nodes.
    Automatically creates a 'pre_reset_<timestamp>' backup first.
    """
    try:
        timestamp = int(time.time())
        auto_version = f"pre_reset_{timestamp}"
        
        backup_dir = DATA_DIR / "backups" / auto_version
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Automatic Backup
        sqlite_client.backup_database(str(backup_dir))
        ChromaClient.backup_chroma(str(backup_dir))
        
        commit_data = {
            "version": auto_version,
            "message": f"[AUTO-BACKUP before reset] {req.commit_message}",
            "timestamp": datetime.utcnow().isoformat()
        }
        with open(backup_dir / "commit.json", "w") as f:
            json.dump(commit_data, f, indent=2)
            
        logger.info(f"Auto-backup {auto_version} created before reset.")
        
        # 2. Reset to Ground Truth
        sqlite_client.reset_to_ground_truth()
        ChromaClient.reset_to_ground_truth()
        
        return ActionResponse(
            status="success",
            message="System successfully reset to ground truth. Auto-backup created.",
            backup_path=str(backup_dir)
        )
        
    except Exception as e:
        logger.error(f"Reset failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

@router.post("/trigger-synthesis")
def trigger_synthesis_manual():
    """
    Manually trigger the pattern finder. Useful for testing without
    waiting for the 5-minute scheduler tick.
    """
    from app.scheduler.jobs.pattern_finder import scan_and_trigger
    result = scan_and_trigger()
    return {
        "status": "triggered",
        "clusters_triggered": result["triggered"],
        "clusters_failed": result["failed"],
        "clusters_skipped": result["skipped"],
    }

@router.post("/trigger-maintenance")
def trigger_maintenance_manual(background_tasks: BackgroundTasks):
    """
    Manually trigger the full maintenance pipeline (staleness check, decay, hierarchy merger)
    as a background task.
    """
    from app.scheduler.jobs.maintenance_job import run_staleness_job, run_decay_job, run_merger_job
    
    def run_all():
        run_staleness_job()
        run_decay_job()
        run_merger_job()
        
    background_tasks.add_task(run_all)
    return {"status": "triggered", "message": "Maintenance pipeline triggered in the background."}


@router.get("/config")
def get_config():
    """
    GET /admin/config
    Returns current active configurations.
    """
    from app import config
    return {
        "HIERARCHY_MERGE_THRESHOLD": getattr(config, "HIERARCHY_MERGE_THRESHOLD", 0.88),
    }


@router.post("/config")
def update_config(req: ConfigUpdateRequest):
    """
    POST /admin/config
    Updates active configurations in memory and persists them to the .env file.
    """
    from app import config
    if not (0.0 <= req.HIERARCHY_MERGE_THRESHOLD <= 1.0):
        raise HTTPException(status_code=400, detail="Threshold must be between 0.0 and 1.0")
        
    config.HIERARCHY_MERGE_THRESHOLD = req.HIERARCHY_MERGE_THRESHOLD
    
    # Persist to .env file
    try:
        env_path = config.BASE_DIR / ".env"
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            found = False
            for i, line in enumerate(lines):
                if line.strip().startswith("HIERARCHY_MERGE_THRESHOLD="):
                    lines[i] = f"HIERARCHY_MERGE_THRESHOLD={req.HIERARCHY_MERGE_THRESHOLD}\n"
                    found = True
                    break
            
            if not found:
                lines.append(f"HIERARCHY_MERGE_THRESHOLD={req.HIERARCHY_MERGE_THRESHOLD}\n")
                
            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        else:
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(f"HIERARCHY_MERGE_THRESHOLD={req.HIERARCHY_MERGE_THRESHOLD}\n")
    except Exception as e:
        logger.error("Failed to write updated config to .env: %s", e)
        
    logger.info("Admin updated HIERARCHY_MERGE_THRESHOLD to %s", req.HIERARCHY_MERGE_THRESHOLD)
    return {"status": "success", "message": f"HIERARCHY_MERGE_THRESHOLD updated to {req.HIERARCHY_MERGE_THRESHOLD}."}

@router.get("/scheduler-status")
def get_scheduler_status():
    """
    Returns current scheduler state for monitoring.
    """
    from app.scheduler import scheduler as scheduler_module
    sched = scheduler_module.get_scheduler()
    if sched is None:
        return {"status": "not_initialized"}

    jobs = []
    for job in sched.get_jobs():
        jobs.append({
            "id": job.id,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
        })

    return {
        "status": "running" if sched.running else "stopped",
        "jobs": jobs,
    }


# ---------------------------------------------------------------------------
# Phase 4 — Synthesis endpoints
# ---------------------------------------------------------------------------

class SynthesisResponse(BaseModel):
    status:        str
    super_node_id: str | None = None
    coverage:      float | None = None
    message:       str


class SuperNodeSummary(BaseModel):
    id:              str
    source_query:    str
    cluster_id:      int
    revision:        int
    fact_coverage:   float
    fidelity_flagged: bool
    drift_margin:    float
    is_stale:        bool
    hit_count:       int
    created_at:      str
    last_accessed:   str


class SuperNodeListResponse(BaseModel):
    nodes: list[SuperNodeSummary]
    total: int


@router.post("/synthesize/{cluster_id}", response_model=SynthesisResponse)
def force_synthesize(cluster_id: int):
    """
    POST /admin/synthesize/{cluster_id}

    Force synthesis for a specific cluster, bypassing the scheduler's
    hit_count threshold. Useful for manual testing and debugging.

    Behaviour
    ----------
    - Checks whether the cluster already has a super-node (SU12 re-synthesis).
    - Acquires the synthesizing lock so the scheduler doesn't double-trigger.
    - Runs the full Phase 4 pipeline synchronously.
    - Returns super_node_id + fact_coverage on success.
    """
    from app.synthesis.synthesizer import (
        SynthesisJob,
        SynthesisError,
        ValidationError,
        run as synthesis_run,
    )

    cluster = sqlite_client.get_cluster(cluster_id)
    if cluster is None:
        raise HTTPException(
            status_code=404,
            detail=f"Cluster {cluster_id} not found in SQLite.",
        )

    # Prevent double-synthesis if scheduler is already running this cluster
    if cluster["synthesizing"]:
        raise HTTPException(
            status_code=409,
            detail=f"Cluster {cluster_id} is already being synthesized. Try again shortly.",
        )

    # Check for existing super-node (SU12)
    existing_node = super_node_store.get_by_cluster(cluster_id)
    existing_sn_id = existing_node["id"] if existing_node else None

    # Acquire concurrency lock
    sqlite_client.set_synthesizing(cluster_id, True)

    try:
        chunk_ids = json.loads(cluster["chunk_ids"])
        job = SynthesisJob(
            cluster_id=cluster_id,
            canonical_query=cluster["canonical_query"],
            chunk_ids=chunk_ids,
            hit_count=cluster["hit_count"],
            triggered_at=datetime.now(timezone.utc),
            existing_sn_id=existing_sn_id,
        )

        sn_id = synthesis_run(job)

        # Mark it synthesized so it unlocks and saves the sn_id to the cluster
        sqlite_client.mark_cluster_synthesized(cluster_id, sn_id)

        # Read back the coverage score from ChromaDB metadata
        meta     = super_node_store.get_metadata(sn_id)
        coverage = meta.get("fact_coverage", 0.0)

        logger.info(
            "admin: force_synthesize success | cluster_id=%d | sn_id=%s | coverage=%.3f",
            cluster_id, sn_id, coverage,
        )
        return SynthesisResponse(
            status="success",
            super_node_id=sn_id,
            coverage=coverage,
            message=f"Super-node {'updated' if existing_sn_id else 'created'} successfully.",
        )

    except (SynthesisError, ValidationError) as exc:
        logger.error("admin: force_synthesize failed | cluster_id=%d | %s", cluster_id, exc)
        raise HTTPException(status_code=422, detail=str(exc))

    except Exception as exc:
        logger.error(
            "admin: force_synthesize unexpected error | cluster_id=%d | %s",
            cluster_id, exc, exc_info=True,
        )
        sqlite_client.set_synthesizing(cluster_id, False)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(exc)}")


@router.get("/super-nodes", response_model=SuperNodeListResponse)
def list_super_nodes():
    """
    GET /admin/super-nodes

    Returns all synthesized super-nodes with their metadata.
    Used by the simulation dashboard to visualize the evolution of
    raw clusters into validated knowledge entries.

    Response fields per node
    -------------------------
    id              : ChromaDB document ID (sn_{cluster_id}_{ts})
    source_query    : canonical cluster question
    cluster_id      : originating SQLite cluster row
    revision        : how many times this node has been re-synthesized (SU12)
    fact_coverage   : LLM-judge score (0.0 – 1.0)
    fidelity_flagged: SU14 soft overfit flag
    drift_margin    : SU14 drift_margin research metric
    is_stale        : SU8/SU13 staleness flag
    hit_count       : number of user queries that triggered this cluster
    created_at      : ISO8601 first synthesis timestamp
    last_accessed   : ISO8601 last access timestamp
    """
    try:
        raw_nodes = super_node_store.list_all()
    except Exception as exc:
        logger.error("admin: list_super_nodes failed — %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read super-nodes: {str(exc)}")

    summaries = []
    for node in raw_nodes:
        meta = node.get("metadata", {})
        summaries.append(SuperNodeSummary(
            id=node["id"],
            source_query=meta.get("source_query", ""),
            cluster_id=int(meta.get("cluster_id", 0)),
            revision=int(meta.get("revision", 1)),
            fact_coverage=float(meta.get("fact_coverage", 0.0)),
            fidelity_flagged=bool(meta.get("fidelity_flagged", False)),
            drift_margin=float(meta.get("drift_margin", 0.0)),
            is_stale=bool(meta.get("is_stale", False)),
            hit_count=int(meta.get("hit_count", 0)),
            created_at=str(meta.get("created_at", "")),
            last_accessed=str(meta.get("last_accessed", "")),
        ))

    return SuperNodeListResponse(nodes=summaries, total=len(summaries))


# ---------------------------------------------------------------------------
# Phase 6 — Maintenance & Hierarchy endpoints
# ---------------------------------------------------------------------------

@router.get("/maintenance-log")
def get_maintenance_log(limit: int = 50):
    """
    GET /admin/maintenance-log?limit=N

    Returns the last N maintenance audit log entries from SQLite.
    Used by the Maintenance tab in the dashboard to display audit events.
    """
    try:
        with sqlite_client.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, event_type, super_node_id, reason, decay_score,
                       timestamp
                FROM maintenance_log
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()
        return {
            "events": [
                {
                    "id":            r["id"],
                    "event_type":    r["event_type"],
                    "super_node_id": r["super_node_id"],
                    "reason":        r["reason"],
                    "decay_score":   r["decay_score"],
                    "timestamp":     r["timestamp"],
                }
                for r in rows
            ],
            "total": len(rows),
        }
    except Exception as e:
        logger.error("admin: get_maintenance_log failed — %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/manual-review-queue")
def get_manual_review_queue():
    """
    GET /admin/manual-review-queue

    Returns all unresolved entries in the manual_review_queue.
    Each entry represents a pair of super-nodes that could not be
    auto-merged because the resulting depth would exceed MAX_LINEAGE_DEPTH.
    """
    try:
        rows = sqlite_client.get_manual_review_queue()
        return {
            "queue": [
                {
                    "id":         r["id"],
                    "sn_id_a":    r["sn_id_a"],
                    "sn_id_b":    r["sn_id_b"],
                    "reason":     r["reason"],
                    "resolved":   bool(r["resolved"]),
                    "created_at": r["created_at"],
                }
                for r in rows
            ],
            "total": len(rows),
        }
    except Exception as e:
        logger.error("admin: get_manual_review_queue failed — %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resolve-review/{review_id}")
def resolve_manual_review(review_id: int):
    """
    POST /admin/resolve-review/{review_id}

    Marks a manual review queue entry as resolved so it no longer appears
    in the pending queue. Does not automatically merge the nodes —
    that must be done manually or via a forced synthesis call.
    """
    try:
        with sqlite_client.get_connection() as conn:
            result = conn.execute(
                "UPDATE manual_review_queue SET resolved = 1 WHERE id = ?",
                (review_id,)
            )
            if result.rowcount == 0:
                raise HTTPException(
                    status_code=404,
                    detail=f"Review queue entry {review_id} not found."
                )
        logger.info("admin: resolved manual review entry id=%d", review_id)
        return {"status": "resolved", "id": review_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("admin: resolve_manual_review failed — %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hierarchy-graph")
def get_hierarchy_graph():
    """
    GET /admin/hierarchy-graph

    Builds and returns the full hierarchy as a JSON graph:
      { nodes: [{id, label, type, depth, hit_count, is_stale}],
        edges: [{from, to, label}] }

    Used by the Hierarchy tab to render the force-directed node graph.
    Meta-nodes (type='meta_node') are the parent disease-level entries.
    Super-nodes (type='super_node') are the child sub-topic entries.
    """
    try:
        super_coll = ChromaClient.get_super_nodes_collection()
        if super_coll is None:
            return {"nodes": [], "edges": [], "message": "No super-nodes collection yet."}

        all_nodes = super_coll.get(include=["metadatas"])
        ids       = all_nodes.get("ids", [])
        metas     = all_nodes.get("metadatas", [])

        graph_nodes = []
        graph_edges = []

        for node_id, meta in zip(ids, metas):
            if not meta:
                continue
            node_type   = meta.get("type", "super_node")
            depth       = int(meta.get("lineage_depth", 0))
            hit_count   = int(meta.get("hit_count", 0))
            is_stale    = bool(meta.get("is_stale", False))
            source_query = meta.get("source_query", node_id)

            graph_nodes.append({
                "id":        node_id,
                "label":     source_query[:60] + ("…" if len(source_query) > 60 else ""),
                "type":      node_type,
                "depth":     depth,
                "hit_count": hit_count,
                "is_stale":  is_stale,
                "in_hierarchy": bool(meta.get("in_hierarchy", False)),
                "parent_meta_id": meta.get("parent_meta_id", ""),
            })

            # Build edges from parent to children
            if node_type == "meta_node":
                import json as _json
                try:
                    children = _json.loads(meta.get("children", "[]"))
                    for child_id in children:
                        graph_edges.append({
                            "from":  node_id,
                            "to":    child_id,
                            "label": f"depth {depth}",
                        })
                except Exception:
                    pass

        return {
            "nodes": graph_nodes,
            "edges": graph_edges,
            "total_nodes": len(graph_nodes),
            "total_edges": len(graph_edges),
        }

    except Exception as e:
        logger.error("admin: get_hierarchy_graph failed — %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class LLMConfigRequest(BaseModel):
    model: str
    use_fallback: bool

@router.get("/llm-config")
def get_llm_config():
    from app import config
    from app.generation import llm_client
    return {
        "model": config.SUMMARIZER_LLM_MODEL,
        "use_fallback": llm_client._GLOBAL_USE_FALLBACK,
        "has_fallback_key": bool(config.OPENAI_API_KEY_FALLBACK)
    }

@router.post("/llm-config")
def update_llm_config(req: LLMConfigRequest):
    from app import config
    from app.generation import llm_client
    import os
    import re
    
    # Update in memory
    config.SUMMARIZER_LLM_MODEL = req.model
    llm_client._GLOBAL_USE_FALLBACK = req.use_fallback
    
    # Update .env
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            content = f.read()
        content = re.sub(r"SUMMARIZER_LLM_MODEL=.*", f"SUMMARIZER_LLM_MODEL={req.model}", content)
        with open(env_path, "w") as f:
            f.write(content)
            
    return {"status": "ok"}

@router.get("/llm-status")
def get_llm_status(model: str, use_fallback: bool):
    from app.generation.llm_client import check_token_limits
    return check_token_limits(model, use_fallback)
