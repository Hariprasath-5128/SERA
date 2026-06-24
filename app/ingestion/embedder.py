from typing import List
from sentence_transformers import SentenceTransformer
from app.models.chunk import Chunk
from app.config import EMBEDDING_MODEL_NAME

class Embedder:
    _model = None

    @classmethod
    def get_model(cls):
        # Lazy loading: Only loads into RAM when called for the first time!
        if cls._model is None:
            cls._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return cls._model

def encode(chunks: List[Chunk], batch_size: int = 64) -> List[Chunk]:
    """
    Encodes a list of Chunk objects into vector embeddings.
    """
    if not chunks:
        return []
        
    model = Embedder.get_model()
    
    # Extract the raw text from each independent section chunk
    texts = [chunk.text for chunk in chunks]
    
    # Process them in fast batches
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    
    # Attach the generated vector math back to the chunk objects
    for i, chunk in enumerate(chunks):
        chunk.embedding = embeddings[i].tolist()
        
    return chunks
