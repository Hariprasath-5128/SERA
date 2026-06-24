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
