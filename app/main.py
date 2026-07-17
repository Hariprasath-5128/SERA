from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import query, ingest, admin, simulation
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.middleware.query_interceptor import QueryInterceptorMiddleware
from app.db import sqlite_client
from app.scheduler import scheduler as scheduler_module
from app import config
import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Startup:  initialise the SQLite database (creates all tables if they
              don't exist yet — safe to run on every boot).
    Shutdown: nothing to clean up for now.
    """
    sqlite_client.init_db()
    _scheduler = scheduler_module.create_scheduler()
    _scheduler.start()
    scheduler_module._scheduler = _scheduler
    logger.info("Phase 3: APScheduler started — pattern_finder every %d min", config.POLL_INTERVAL_MINUTES)
    yield
    _scheduler.shutdown(wait=False)
    logger.info("Phase 3: APScheduler shut down")


app = FastAPI(
    title="SERA AI API",
    description="Backend API for the Self Evolving RAG Assistant (SERA)",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Phase 2: Query Interceptor (SU9 ground-truth filter + query logging) ──────
# Must be registered AFTER CORSMiddleware so it wraps the actual route handler.
# Starlette applies middleware in reverse registration order (last-in, first-run),
# so this must come after CORSMiddleware to intercept the real response body.
app.add_middleware(QueryInterceptorMiddleware)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(query.router)
app.include_router(ingest.router)
app.include_router(admin.router)
app.include_router(simulation.router)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SERA API is running"}

# Serve the frontend directory at the root
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
_frontend_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
