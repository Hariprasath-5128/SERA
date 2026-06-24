from typing import List
from app.models.chunk import Chunk
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

def split(text: str, document_id: str, source_url: str, question_focus: str, umls_semantic_group: str, section: str = "General") -> List[Chunk]:
    """
    Splits raw text into overlapping chunks.
    This uses a straightforward word-boundary split, approximating tokens
    (1 word roughly equals 1.3 tokens) to respect the CHUNK_SIZE limit.
    """
    words = text.split()
    chunks = []
    chunk_idx = 0
    
    # Rough approximation: 1 word ~ 1.3 tokens
    words_per_chunk = int(CHUNK_SIZE / 1.3)
    words_overlap = int(CHUNK_OVERLAP / 1.3)
    
    if not words:
        return []
        
    start = 0
    while start < len(words):
        end = start + words_per_chunk
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        token_count = int(len(chunk_words) * 1.3)
        
        chunks.append(Chunk(
            chunk_id=f"{document_id}_{section}_{chunk_idx}",
            text=chunk_text,
            document_id=document_id,
            source_url=source_url,
            question_focus=question_focus,
            umls_semantic_group=umls_semantic_group,
            section=section,
            token_count=token_count
        ))
        
        chunk_idx += 1
        start += (words_per_chunk - words_overlap)
        
        # Prevent infinite loops if overlap is misconfigured to be >= chunk size
        if words_per_chunk - words_overlap <= 0:
            break
            
    return chunks
