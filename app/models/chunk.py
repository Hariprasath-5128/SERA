from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Chunk:
    chunk_id: str
    text: str
    document_id: str
    source_url: str
    question_focus: str
    umls_semantic_group: str
    section: str
    token_count: int
    embedding: Optional[List[float]] = None
