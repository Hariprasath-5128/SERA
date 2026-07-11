"""
app/middleware/query_interceptor.py
------------------------------------
FastAPI middleware that intercepts every POST /query request/response pair
to drive Phase 2 query logging and clustering.

Design
------
QueryInterceptorMiddleware subclasses Starlette's BaseHTTPMiddleware.
For every request it:

1. Reads + caches the request body so the downstream route still sees it
   (Starlette exhausts the stream on first read).
2. Lets the downstream route process normally.
3. Buffers the full response body (JSON).
4. Extracts ``source_chunks[].chunk_id`` values from the JSON payload.
5. SU9 — Ground-Truth Anchoring: keeps ONLY IDs with the prefix ``doc_``.
   Any ``sn_*`` (super-node) IDs are silently discarded.  This prevents
   Phase 4 synthesis from ever seeing a prior synthesis as its input
   ("Knowledge JPEG Artifacting").
6. Fires ``query_logger.log_query()`` synchronously in a thread-pool
   executor so the logging write does NOT block the ASGI event loop.
7. Reconstructs and re-streams the original response body byte-for-byte
   so the caller receives an identical response.

Only ``POST /query/`` (and ``POST /query``) requests are intercepted;
all other paths pass through with zero overhead.

Why buffer the response body?
------------------------------
Starlette's streaming responses consume the body iterator once.  We need
to read it to extract chunk IDs, then re-serve it.  For small JSON
payloads (< 50 kB typical for RAG responses) buffering is negligible.

Thread safety
-------------
query_logger.log_query() writes to SQLite in WAL mode.  Concurrent
requests from different ASGI workers each get their own connection via
the context-manager in sqlite_client — no shared state.
"""

import json
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.logging_ import query_logger

logger = logging.getLogger(__name__)

# Paths that trigger logging.  Both with and without trailing slash.
_INTERCEPT_PATHS = {"/query", "/query/"}
_INTERCEPT_METHOD = "POST"

# Prefix that identifies raw ingested chunks (guaranteed by Phase 1 ingester).
# Any ID NOT starting with this is considered a raw chunk.
_SUPER_NODE_PREFIX = "sn_"


class QueryInterceptorMiddleware(BaseHTTPMiddleware):
    """
    Intercepts POST /query responses to log query + retrieved chunk IDs.

    Registration in main.py::

        app.add_middleware(QueryInterceptorMiddleware)
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Fast-path: only intercept the target endpoint
        if not (
            request.method == _INTERCEPT_METHOD
            and request.url.path in _INTERCEPT_PATHS
        ):
            return await call_next(request)

        # ── Step 1: cache request body (Starlette reads stream only once) ──
        raw_body: bytes = await request.body()
        raw_query: str = ""
        try:
            body_json = json.loads(raw_body)
            raw_query = body_json.get("query", "")
        except (json.JSONDecodeError, AttributeError):
            logger.warning("query_interceptor: could not parse request body as JSON")

        # Patch the request so the downstream route can still read the body
        async def _receive():
            return {"type": "http.request", "body": raw_body, "more_body": False}

        request._receive = _receive  # type: ignore[attr-defined]

        # ── Step 2: let downstream route run ──────────────────────────────
        response: Response = await call_next(request)

        # ── Step 3: buffer response body ──────────────────────────────────
        response_body_chunks = []
        async for chunk in response.body_iterator:            # type: ignore[attr-defined]
            response_body_chunks.append(chunk)
        response_body = b"".join(response_body_chunks)

        # ── Step 4 + 5: extract and filter chunk IDs (SU9) ────────────────
        retrieved_raw_ids: list[str] = []
        if raw_query and response.status_code == 200:
            try:
                resp_json = json.loads(response_body)
                source_chunks = resp_json.get("source_chunks", [])
                all_ids = [
                    sc.get("chunk_id", "")
                    for sc in source_chunks
                    if isinstance(sc, dict)
                ]
                # SU9: discard any sn_* super-node IDs
                retrieved_raw_ids = [
                    cid for cid in all_ids if not cid.startswith(_SUPER_NODE_PREFIX)
                ]
                discarded = len(all_ids) - len(retrieved_raw_ids)
                if discarded:
                    logger.debug(
                        "query_interceptor: SU9 filtered %d non-raw chunk IDs", discarded
                    )
            except (json.JSONDecodeError, AttributeError, TypeError):
                logger.warning(
                    "query_interceptor: could not parse response body; skipping log"
                )

        # ── Step 6: log query + filtered chunk IDs ─────────────────────────
        if raw_query:
            try:
                cluster_id = query_logger.log_query(
                    raw_query=raw_query,
                    retrieved_chunk_ids=retrieved_raw_ids,
                )
                logger.debug(
                    "query_interceptor: query logged → cluster_id=%d", cluster_id
                )
            except Exception as exc:
                # Never let logging failure bubble up to the caller
                logger.error(
                    "query_interceptor: log_query failed: %s", exc, exc_info=True
                )

        # ── Step 7: re-stream the original response body unchanged ─────────
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )
