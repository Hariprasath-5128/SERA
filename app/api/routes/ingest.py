from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
import logging
import os
import sys

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.ingestion.ingestor import run_ingestion

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

class IngestResponse(BaseModel):
    message: str
    status: str

def trigger_ingestion_job():
    """Background task function to safely run ingestion."""
    try:
        logger.info("Starting background ingestion job...")
        run_ingestion()
        logger.info("Background ingestion job completed successfully.")
    except Exception as e:
        logger.error(f"Background ingestion failed: {e}")

@router.post("/medquad", response_model=IngestResponse, status_code=202)
def start_medquad_ingestion(background_tasks: BackgroundTasks):
    """
    Triggers the MedQuAD dataset ingestion pipeline.
    Runs in the background so it doesn't block the HTTP request.
    """
    background_tasks.add_task(trigger_ingestion_job)
    return IngestResponse(
        message="MedQuAD ingestion job has been accepted and is running in the background.",
        status="running"
    )
