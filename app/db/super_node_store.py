"""
app/db/super_node_store.py
----------------------------
Dedicated ChromaDB client for the `super_nodes` collection.

Why a separate module from chroma_client.py?
---------------------------------------------
chroma_client.py manages the `raw_chunks` collection — the ground-truth
ingested medical text. super_node_store.py manages the `super_nodes`
collection — the synthesized, validated knowledge entries produced by
Phase 4.

Keeping them separated:
  (a) prevents accidental cross-collection writes
  (b) makes the super_node lifecycle (upsert / metadata patch / list)
      independently testable
  (c) mirrors the ChromaDB collection boundary that SU9 enforces at
      query time (raw vs. synthesized retrieval paths)

Functions
----------
  upsert()          — write or overwrite a super-node (synthesis + re-synthesis)
  get_metadata()    — read metadata dict for one node by sn_id
  update_metadata() — patch metadata fields in-place (used by SU13)
  list_all()        — return all nodes for /admin/super-nodes endpoint
  get_by_cluster()  — return the super-node for a specific cluster_id
"""

import logging
from typing import Optional

import chromadb
import numpy as np

from app.config import CHROMA_PERSIST_PATH, CHROMA_SUPER_COLLECTION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy-loaded singleton client + collection
# ---------------------------------------------------------------------------

_client:     Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection]       = None


def _get_collection() -> chromadb.Collection:
    """
    Returns the cached `super_nodes` ChromaDB collection.
    Uses cosine similarity space — consistent with raw_chunks.
    Creates the collection if it does not yet exist.
    """
    global _client, _collection
    if _collection is None:
        import app.config as config
        if _client is None:
            _client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_PATH)
            logger.info(
                "super_node_store: ChromaDB client initialised at %s",
                config.CHROMA_PERSIST_PATH,
            )
        _collection = _client.get_or_create_collection(
            name=config.CHROMA_SUPER_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "super_node_store: collection '%s' ready",
            config.CHROMA_SUPER_COLLECTION,
        )
    return _collection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upsert(
    sn_id:     str,
    summary:   str,
    embedding: np.ndarray,
    metadata:  dict,
) -> None:
    """
    Insert or overwrite a super-node in the `super_nodes` collection.

    Called by synthesizer.py after:
      - Validation passes (SU2)
      - Fidelity bound passes (SU14)

    On re-synthesis (SU12), the same sn_id is passed, so ChromaDB replaces
    the existing document and embedding in-place.

    Parameters
    ----------
    sn_id     : str   — e.g. "sn_42_1720789200"
    summary   : str   — the validated compressed synthesis text
    embedding : ndarray — 384-dim float32 vector from BAAI/bge-m3
    metadata  : dict  — full metadata dict as specified in the blueprint schema
    """
    collection = _get_collection()

    # ChromaDB metadata values must be str | int | float | bool — no nested dicts/lists
    # source_chunks is already JSON-serialized by synthesizer.py (json.dumps)
    sanitized_meta = {
        k: (bool(v) if isinstance(v, (bool, np.bool_)) else
            float(v) if isinstance(v, (float, np.floating)) else
            int(v)   if isinstance(v, (int, np.integer)) else
            str(v)   if not isinstance(v, str) else v)
        for k, v in metadata.items()
    }

    collection.upsert(
        ids=[sn_id],
        embeddings=[embedding.tolist()],
        documents=[summary],
        metadatas=[sanitized_meta],
    )
    logger.info(
        "super_node_store: upserted sn_id=%s | revision=%s",
        sn_id,
        sanitized_meta.get("revision", "?"),
    )


def get_metadata(sn_id: str) -> dict:
    """
    Fetch the metadata dict for a single super-node by its sn_id.

    Used by:
      - synthesizer.py (SU12): read revision/parent_meta_id/lineage_depth
        when re-synthesizing an existing node
      - synthesizer.py (SU13): read parent node metadata before marking stale

    Returns
    -------
    dict
        Metadata dict as stored in ChromaDB.

    Raises
    ------
    KeyError
        If sn_id does not exist in the collection.
    """
    collection = _get_collection()
    result = collection.get(ids=[sn_id], include=["metadatas"])

    if not result["ids"]:
        raise KeyError(f"super_node_store: sn_id '{sn_id}' not found in collection")

    return result["metadatas"][0]


def update_metadata(sn_id: str, metadata: dict) -> None:
    """
    Patch metadata fields for an existing super-node.

    Used by:
      - synthesizer.py (SU13): flip is_stale=True on a parent node
        when a child is re-synthesized

    Note: This does NOT update the document text or embedding — only metadata.
    ChromaDB's `.update()` preserves the existing document/embedding
    when they are not included in the call.

    Parameters
    ----------
    sn_id    : str  — the super-node ID to update
    metadata : dict — the full metadata dict to replace with
                      (caller is responsible for merging; see SU13 in synthesizer.py)
    """
    collection = _get_collection()

    sanitized_meta = {
        k: (bool(v) if isinstance(v, (bool, np.bool_)) else
            float(v) if isinstance(v, (float, np.floating)) else
            int(v)   if isinstance(v, (int, np.integer)) else
            str(v)   if not isinstance(v, str) else v)
        for k, v in metadata.items()
    }

    collection.update(ids=[sn_id], metadatas=[sanitized_meta])
    logger.debug("super_node_store: metadata updated for sn_id=%s", sn_id)


def list_all() -> list[dict]:
    """
    Return all super-nodes with their metadata.

    Used by:
      - app/api/routes/admin.py → GET /admin/super-nodes

    Returns
    -------
    list[dict]
        Each item has keys: id, document, metadata.
        Empty list if no super-nodes have been synthesized yet.
    """
    collection = _get_collection()
    result = collection.get(include=["documents", "metadatas"])

    nodes = []
    for sn_id, doc, meta in zip(
        result.get("ids", []),
        result.get("documents", []),
        result.get("metadatas", []),
    ):
        nodes.append({
            "id":       sn_id,
            "document": doc,
            "metadata": meta,
        })

    logger.debug("super_node_store: list_all returned %d nodes", len(nodes))
    return nodes


def get_by_cluster(cluster_id: int) -> Optional[dict]:
    """
    Return the super-node for a specific cluster_id, or None if not yet synthesized.

    Used by the admin /synthesize endpoint to check whether a node
    already exists before deciding to create vs. update.

    Returns
    -------
    dict | None
        Same shape as list_all() items, or None.
    """
    collection = _get_collection()
    result = collection.get(
        where={"cluster_id": cluster_id},
        include=["documents", "metadatas"],
    )

    if not result["ids"]:
        return None

    return {
        "id":       result["ids"][0],
        "document": result["documents"][0],
        "metadata": result["metadatas"][0],
    }
