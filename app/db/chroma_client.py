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
        Safely copies the ChromaDB persistence directory to the backup directory.
        """
        import os
        import shutil
        from pathlib import Path
        
        os.makedirs(backup_dir, exist_ok=True)
        dest_path = Path(backup_dir) / "chroma"
        
        logger.info("chroma_client: backing up ChromaDB to %s", dest_path)
        
        if os.path.exists(CHROMA_PERSIST_PATH):
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            shutil.copytree(CHROMA_PERSIST_PATH, dest_path)

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
