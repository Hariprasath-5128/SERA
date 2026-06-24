from typing import Set

class UrlExtractor:
    """
    Extracts and deduplicates document URLs.
    MedQuAD contains many question-answer pairs that share the same document_url.
    We only want to crawl each document once.
    """
    def __init__(self):
        self.seen_docs: Set[str] = set()

    def is_new_document(self, document_id: str) -> bool:
        """
        Checks if a document_id has been seen before.
        Returns True if it's new, False if it's a duplicate.
        """
        if document_id in self.seen_docs:
            return False
        
        self.seen_docs.add(document_id)
        return True
