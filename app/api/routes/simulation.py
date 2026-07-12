"""
app/api/routes/simulation.py
-----------------------------
Phase 3 Simulation API — drives the Phase 3 Dashboard UI.

Endpoints
---------
GET  /simulation/benchmark-samples   → random Q&A pairs from benchmark_qa
POST /simulation/run                 → start background simulation job
GET  /simulation/status/{job_id}     → poll progress of a running simulation
GET  /simulation/clusters            → full cluster list for dashboard
GET  /simulation/query-log           → recent query log with cluster info
POST /simulation/reset               → wipe clusters + logs (keep benchmark_qa)
POST /simulation/trigger-synthesis   → manually fire scan_and_trigger()
"""

import json
import logging
import random
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.db import sqlite_client
from app.logging_.query_logger import log_query
from app.retrieval.retriever import Retriever
from app.scheduler.jobs.pattern_finder import scan_and_trigger

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/simulation", tags=["Phase 3 Simulation"])

# ---------------------------------------------------------------------------
# In-memory job store — keyed by job_id
# ---------------------------------------------------------------------------
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SimulationRequest(BaseModel):
    total_queries: int = 50
    biased_query: str
    bias_percent: int = 40        # % of queries that will be the biased one
    fast_mode: bool = True         # skip retriever; only log embedding (faster)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _update_job(job_id: str, **kwargs):
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


def _simulation_worker(job_id: str, req: SimulationRequest):
    """Runs in a background thread so the API returns immediately."""
    _update_job(job_id, status="running", progress=0, results=[], synthesis=None)

    # 1. Gather pool of random queries from benchmark_qa
    try:
        with sqlite_client.get_connection() as conn:
            rows = conn.execute(
                "SELECT question FROM benchmark_qa ORDER BY RANDOM() LIMIT 500"
            ).fetchall()
        pool = [r["question"] for r in rows]
    except Exception as e:
        logger.warning("simulation: could not fetch benchmark_qa rows — %s", e)
        pool = []

    if not pool:
        pool = [req.biased_query]

    # 2. Build shuffled query list with bias
    bias_count  = max(1, int(req.total_queries * req.bias_percent / 100))
    other_count = req.total_queries - bias_count
    other_queries = (pool * (other_count // max(len(pool), 1) + 1))[:other_count]
    all_queries   = [req.biased_query] * bias_count + other_queries
    random.shuffle(all_queries)

    results: List[Dict[str, Any]] = []

    for idx, query in enumerate(all_queries):
        start_ts = time.time()
        entry: Dict[str, Any] = {
            "index": idx + 1,
            "query": query,
            "is_biased": query.strip().lower() == req.biased_query.strip().lower(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        try:
            chunk_ids: List[str] = []
            similarity_score: float = 0.0
            latency_ms: float = 0.0

            if req.fast_mode:
                # Embed-only path: call log_query without hitting ChromaDB retriever
                t0 = time.time()
                cluster_id = log_query(
                    raw_query=query,
                    retrieved_chunk_ids=chunk_ids,
                )
                latency_ms = (time.time() - t0) * 1000
            else:
                # Full retrieval path
                t0 = time.time()
                retrieval_results = Retriever.search(query, top_k=3)
                retrieval_ms = (time.time() - t0) * 1000

                chunk_ids = [
                    r["chunk_id"] for r in retrieval_results
                    if not r["chunk_id"].startswith("sn_")
                ]
                distances = [r["distance"] for r in retrieval_results]
                similarity_score = round(1.0 - min(distances), 4) if distances else 0.0

                t1 = time.time()
                cluster_id = log_query(
                    raw_query=query,
                    retrieved_chunk_ids=chunk_ids,
                )
                latency_ms = retrieval_ms + (time.time() - t1) * 1000

            # Fetch updated cluster info
            cluster_row = sqlite_client.get_cluster(cluster_id)
            hit_count    = cluster_row["hit_count"] if cluster_row else 1
            matched      = cluster_row is not None

            entry.update({
                "cluster_id": cluster_id,
                "latency_ms": round(latency_ms, 2),
                "similarity_score": similarity_score,
                "chunk_ids": chunk_ids,
                "matched_existing": matched,
                "hit_count": hit_count,
                "error": None,
            })

        except Exception as e:
            logger.error("simulation: query %d failed — %s", idx + 1, e)
            entry.update({
                "cluster_id": None,
                "latency_ms": round((time.time() - start_ts) * 1000, 2),
                "similarity_score": 0.0,
                "chunk_ids": [],
                "matched_existing": False,
                "hit_count": 0,
                "error": str(e),
            })

        results.append(entry)
        _update_job(
            job_id,
            progress=int((idx + 1) / req.total_queries * 100),
            results=results,
        )

    # 3. Fire synthesis trigger
    try:
        synthesis = scan_and_trigger()
    except Exception as e:
        logger.error("simulation: synthesis trigger failed — %s", e)
        synthesis = {"triggered": 0, "skipped": 0, "failed": 0, "error": str(e)}

    _update_job(
        job_id,
        status="done",
        progress=100,
        results=results,
        synthesis=synthesis,
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    logger.info("simulation job %s complete — %d queries, synthesis=%s", job_id, len(results), synthesis)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/benchmark-samples")
def get_benchmark_samples(limit: int = 30):
    """Return random questions from benchmark_qa for the query picker."""
    try:
        with sqlite_client.get_connection() as conn:
            rows = conn.execute(
                "SELECT id, question FROM benchmark_qa ORDER BY RANDOM() LIMIT ?",
                (limit,),
            ).fetchall()
        return {"samples": [{"id": r["id"], "question": r["question"]} for r in rows]}
    except Exception as e:
        logger.error("benchmark-samples failed: %s", e)
        return {"samples": []}


@router.post("/run")
def run_simulation(req: SimulationRequest, background_tasks: BackgroundTasks):
    """Start a simulation job. Returns a job_id to poll with /status/{job_id}."""
    job_id = uuid.uuid4().hex[:8]
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "total": req.total_queries,
            "results": [],
            "synthesis": None,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    t = threading.Thread(target=_simulation_worker, args=(job_id, req), daemon=True)
    t.start()
    return {"job_id": job_id, "status": "queued"}


@router.get("/status/{job_id}")
def get_simulation_status(job_id: str):
    """Poll progress of a running simulation."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/chunk/{chunk_id}")
def get_chunk(chunk_id: str):
    """Fetch raw chunk text from ChromaDB by ID."""
    from app.db.chroma_client import ChromaClient
    try:
        col = ChromaClient.get_collection()
        res = col.get(ids=[chunk_id])
        if res and res["documents"] and len(res["documents"]) > 0:
            return {"text": res["documents"][0]}
        return {"text": "Chunk content not found in database."}
    except Exception as e:
        logger.error("get_chunk failed: %s", e)
        return {"text": f"Error loading chunk: {str(e)}"}


@router.get("/clusters")
def get_clusters():
    """Full cluster list with history for the dashboard."""
    rows = sqlite_client.fetch_all_clusters()
    clusters = []
    for row in rows:
        cid = row["id"]
        history_rows = sqlite_client.get_super_node_history(cid)
        history = [
            {
                "revision": h["revision"],
                "summary": h["summary"],
                "coverage_score": h["coverage_score"],
                "drift_margin": h["drift_margin"],
                "chunk_ids_hash": h["chunk_ids_hash"],
                "timestamp": h["timestamp"],
            }
            for h in history_rows
        ]
        clusters.append({
            "id": cid,
            "canonical_query": row["canonical_query"],
            "hit_count": row["hit_count"],
            "chunk_ids": json.loads(row["chunk_ids"] or "[]"),
            "first_seen": row["first_seen"],
            "last_hit": row["last_hit"],
            "synthesized": bool(row["synthesized"]),
            "synthesizing": bool(row["synthesizing"]),
            "synthesis_failed": bool(row["synthesis_failed"]),
            "super_node_id": row["super_node_id"],
            "last_synthesized_hit_count": row["last_synthesized_hit_count"],
            "revision_count": len(history),
            "history": history,
        })
    clusters.sort(key=lambda c: c["hit_count"], reverse=True)
    return {"clusters": clusters, "total": len(clusters)}


@router.get("/query-log")
def get_query_log(limit: int = 100):
    """Recent query log joined with cluster info."""
    try:
        with sqlite_client.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT ql.id, ql.raw_query, ql.cluster_id, ql.retrieved_chunks,
                       ql.matched_existing, ql.timestamp,
                       qc.canonical_query, qc.hit_count
                FROM query_log ql
                LEFT JOIN query_clusters qc ON ql.cluster_id = qc.id
                ORDER BY ql.timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    except Exception as e:
        logger.error("query-log failed: %s", e)
        return {"logs": []}

    return {
        "logs": [
            {
                "id": r["id"],
                "raw_query": r["raw_query"],
                "cluster_id": r["cluster_id"],
                "canonical_query": r["canonical_query"],
                "hit_count": r["hit_count"],
                "retrieved_chunks": json.loads(r["retrieved_chunks"] or "[]"),
                "matched_existing": bool(r["matched_existing"]),
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]
    }


@router.post("/reset")
def reset_simulation():
    """Wipe all Phase 2+ data. Keeps benchmark_qa intact."""
    sqlite_client.reset_to_ground_truth()
    with _jobs_lock:
        _jobs.clear()
    return {"status": "reset", "message": "All clusters, logs, and jobs cleared."}


@router.post("/trigger-synthesis")
def trigger_synthesis():
    """Manually fire the Phase 3 pattern finder."""
    result = scan_and_trigger()
    return {"status": "ok", "synthesis": result}
