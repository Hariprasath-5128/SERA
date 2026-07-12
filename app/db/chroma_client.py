import chromadb
from typing import List
import logging
from app.config import CHROMA_PERSIST_PATH, CHROMA_RAW_COLLECTION
from app.models.chunk import Chunk

logger = logging.getLogger(__name__)

class ChromaClient:
    """Lazy-loaded wrapper for ChromaDB."""
    _client = None
    _collection = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            # We persist the database to the path defined in config.py
            cls._client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
        return cls._client

    @classmethod
    def get_collection(cls):
        if cls._collection is None:
            client = cls.get_client()
            # We use cosine similarity to find the closest vectors
            cls._collection = client.get_or_create_collection(
                name=CHROMA_RAW_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )
        return cls._collection

    @classmethod
    def insert_chunks(cls, chunks: List[Chunk]):
        """Inserts a list of embedded Chunk objects into ChromaDB."""
        if not chunks:
            return

        collection = cls.get_collection()
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            if not chunk.embedding:
                logger.warning(f"Chunk {chunk.chunk_id} has no embedding. Skipping.")
                continue
                
            ids.append(chunk.chunk_id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.text)
            
            # Store all our rich properties as metadata for filtering later!
            # e.g., collection.query(..., where={"section": "CAUSES"})
            metadatas.append({
                "document_id": chunk.document_id,
                "source_url": chunk.source_url,
                "question_focus": chunk.question_focus,
                "umls_semantic_group": chunk.umls_semantic_group,
                "section": chunk.section,
                "token_count": chunk.token_count
            })
            
        if ids:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Inserted {len(ids)} chunks into ChromaDB.")

    @classmethod
    def backup_chroma(cls, backup_dir: str):
        """
        Exports only the super_nodes collection to a JSON file to save space,
        as the raw chunks are static.
        """
        import os
        import json
        from pathlib import Path
        from app.config import CHROMA_SUPER_COLLECTION
        
        os.makedirs(backup_dir, exist_ok=True)
        dest_path = Path(backup_dir) / "super_nodes_backup.json"
        
        logger.info("chroma_client: exporting super_nodes to %s", dest_path)
        
        try:
            client = cls.get_client()
            col = client.get_or_create_collection(name=CHROMA_SUPER_COLLECTION)
            data = col.get(include=["embeddings", "metadatas", "documents"])
            
            # Convert any ndarrays to lists for JSON serialization if necessary
            embeddings = data.get("embeddings") or []
            if embeddings and hasattr(embeddings[0], "tolist"):
                embeddings = [emb.tolist() for emb in embeddings]
                
            with open(dest_path, "w", encoding="utf-8") as f:
                json.dump({
                    "ids": data.get("ids", []),
                    "embeddings": embeddings,
                    "metadatas": data.get("metadatas", []),
                    "documents": data.get("documents", [])
                }, f)
                
            logger.info("chroma_client: successfully exported %d super nodes", len(data.get("ids", [])))
        except Exception as e:
            logger.error("Failed to backup super_nodes: %s", e)

    @classmethod
    def reset_to_ground_truth(cls):
        """
        Wipe all learned data (Phase 2+) while retaining Phase 1 ground truth.
        This deletes the super_nodes collection entirely.
        """
        from app.config import CHROMA_SUPER_COLLECTION
        client = cls.get_client()
        try:
            client.delete_collection(CHROMA_SUPER_COLLECTION)
            logger.warning("chroma_client: RESETTING TO GROUND TRUTH. Deleted super_nodes collection.")
        except Exception:
            # Collection might not exist yet, which is fine
            pass
