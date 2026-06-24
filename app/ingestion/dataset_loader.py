import os
from typing import Iterator
from dataclasses import dataclass
from datasets import load_dataset

@dataclass
class MedQuADRow:
    document_id: str
    document_url: str
    question_focus: str
    umls_semantic_group: str
    question_id: str
    question_type: str
    question: str
    answer: str

def stream_medquad() -> Iterator[MedQuADRow]:
    """
    Streams the MedQuAD dataset from HuggingFace.
    """
    # Uses HF_TOKEN from environment if required for access
    token = os.environ.get("HF_TOKEN")
    
    ds = load_dataset("lavita/MedQuAD", split="train", streaming=True, token=token)
    
    for row in ds:
        yield MedQuADRow(
            document_id=row["document_id"],
            document_url=row["document_url"],
            question_focus=row["question_focus"] or "",
            umls_semantic_group=row["umls_semantic_group"] or "",
            question_id=row["question_id"],
            question_type=row["question_type"] or "",
            question=row["question"],
            answer=row["answer"] or ""
        )


