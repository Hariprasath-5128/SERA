from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import time
import json
import os
from pathlib import Path
import logging
from datetime import datetime

from app.config import DATA_DIR
from app.db import sqlite_client
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
