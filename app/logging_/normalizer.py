"""
app/logging_/normalizer.py
--------------------------
Thin text normalisation + embedding helper.

Responsibilities
----------------
- normalize(text): collapse whitespace, strip leading/trailing space,
  lowercase — produces a canonical form for comparison.
- embed(text):     normalize → encode via the shared Embedder singleton
                   → return a 1-D float32 numpy array.

Why keep this separate from query_logger?
  query_logger is stateful (touches SQLite).
  normalizer is a pure function module — easier to unit-test and mock.
"""

import re
import numpy as np
from app.ingestion.embedder import Embedder


def normalize(text: str) -> str:
    """
    Produce a canonical form of *text* suitable for embedding and display
    as a cluster's ``canonical_query``.

    Steps
    -----
    1. Strip leading / trailing whitespace.
    2. Collapse internal whitespace runs to a single space.
    3. Lowercase.

    Examples
    --------
    >>> normalize("  What  IS  Niemann-Pick  disease? ")
    'what is niemann-pick disease?'
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def embed(text: str) -> np.ndarray:
    """
    Normalise *text* then encode it with the shared SentenceTransformer.

    Returns
    -------
    np.ndarray
        1-D float32 array of shape ``(embedding_dim,)``.

    Notes
    -----
    - Uses ``Embedder.get_model()`` which is lazy-loaded on first call and
      then cached as a class variable — no extra model loads per request.
    - ``convert_to_numpy=True`` is set on the model call so we get a proper
      ndarray back, not a tensor.
    """
    canonical = normalize(text)
    model = Embedder.get_model()
    vector = model.encode(
        [canonical],
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    # model.encode returns shape (1, dim) for a list; we want (dim,)
    return vector[0].astype(np.float32)
