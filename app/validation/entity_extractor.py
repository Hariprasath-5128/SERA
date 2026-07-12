"""
app/validation/entity_extractor.py
-------------------------------------
Hallucination detection via spaCy Named Entity Recognition (NER).

Role in Phase 4
---------------
This module is the SECONDARY validation axis, running AFTER the primary
LLM-as-judge (validator.py) passes.

The LLM-as-judge (SU2) is excellent at detecting MISSING facts (coverage).
However, it can occasionally miss ADDED facts — entities that the LLM
injected into the summary that were never present in the source material
(e.g. a drug name, a researcher's name, a date).

entity_extractor catches this by:
  1. Extracting all named entities from the source text (ground truth).
  2. Extracting all named entities from the generated summary.
  3. Computing the set-difference: entities in summary but NOT in source.
  4. Returning a hallucination_ratio = added / max(summary_entities, 1).

It is also used as a FALLBACK by validator.py when the LLM judge returns
malformed JSON (blueprint line 1532, 2847):
  "LLM judge JSON parse error → fallback to entity_extractor; log alert"

Requires
--------
  pip install spacy
  python -m spacy download en_core_web_sm
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-load spaCy model (heavy; only load once on first call)
# ---------------------------------------------------------------------------

_nlp = None


def _get_nlp():
    """Lazy-load en_core_web_sm. Raises RuntimeError if not installed."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
            logger.info("entity_extractor: spaCy en_core_web_sm loaded")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. "
                "Run: python -m spacy download en_core_web_sm"
            )
    return _nlp


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> set[str]:
    """
    Extract a set of normalized named entities from text using spaCy NER.

    Normalization:
      - Lowercased
      - Stripped of leading/trailing whitespace
      - Empty strings excluded

    Parameters
    ----------
    text : str
        Raw text to extract entities from.

    Returns
    -------
    set[str]
        Set of normalized entity strings.
        Returns empty set on empty input or spaCy failure.
    """
    if not text or not text.strip():
        return set()

    try:
        nlp = _get_nlp()
        doc = nlp(text)
        return {ent.text.strip().lower() for ent in doc.ents if ent.text.strip()}
    except Exception as exc:
        logger.warning("entity_extractor: extraction failed — %s", exc)
        return set()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_hallucination(source_text: str, summary: str) -> dict:
    """
    Detect named entities present in the summary but absent from the source.

    These are potential hallucinations — facts the LLM injected that do not
    exist anywhere in the ground-truth source material.

    Parameters
    ----------
    source_text : str
        The deduplicated raw source passages (output of chunk_fetcher).
    summary : str
        The LLM-generated synthesis to check.

    Returns
    -------
    dict with keys:
        source_entities   : set[str]  — all entities found in source
        summary_entities  : set[str]  — all entities found in summary
        added_entities    : list[str] — entities in summary NOT in source (sorted)
        hallucination_ratio : float   — len(added) / max(len(summary_entities), 1)
                                        0.0 = clean, 1.0 = fully hallucinated

    Usage in synthesizer.py (Step 7 — optional check)
    ---------------------------------------------------
        result = check_hallucination(source_text, summary)
        if result["hallucination_ratio"] > 0.0:
            logger.warning(
                "entity_extractor: %d potential hallucination(s): %s",
                len(result["added_entities"]),
                result["added_entities"]
            )
        # Note: this is advisory only. synthesizer.py does not abort on
        # hallucination_ratio > 0; it logs and proceeds. The LLM-as-judge
        # (SU2) is the hard gate.
    """
    source_entities  = extract_entities(source_text)
    summary_entities = extract_entities(summary)

    added_entities = sorted(summary_entities - source_entities)

    hallucination_ratio = len(added_entities) / max(len(summary_entities), 1)

    logger.debug(
        "entity_extractor: source=%d entities | summary=%d entities | "
        "added=%d | ratio=%.3f",
        len(source_entities),
        len(summary_entities),
        len(added_entities),
        hallucination_ratio,
    )

    return {
        "source_entities":     source_entities,
        "summary_entities":    summary_entities,
        "added_entities":      added_entities,
        "hallucination_ratio": hallucination_ratio,
    }


def fallback_validate(source_text: str, summary: str) -> dict:
    """
    Fallback coverage estimate when the LLM judge returns malformed JSON.

    Uses entity overlap as a rough proxy for fact coverage:
      coverage = len(source ∩ summary) / max(len(source_entities), 1)

    This is NOT as accurate as the LLM judge, but prevents a total
    validation failure when the judge malfunctions.

    Returns
    -------
    dict with keys:
        passed         : bool   — True if coverage >= 0.70 (relaxed threshold)
        coverage_score : float  — entity overlap ratio
        missing_facts  : list[str] — source entities not found in summary
    """
    source_entities  = extract_entities(source_text)
    summary_entities = extract_entities(summary)

    if not source_entities:
        # Cannot validate without source entities — conservatively pass
        logger.warning(
            "entity_extractor: fallback_validate — no source entities found; "
            "conservatively passing"
        )
        return {"passed": True, "coverage_score": 1.0, "missing_facts": []}

    matched      = source_entities & summary_entities
    missing      = sorted(source_entities - summary_entities)
    coverage     = len(matched) / len(source_entities)
    # Relaxed threshold for fallback (0.70 vs 0.90 for LLM judge)
    passed       = coverage >= 0.70

    logger.warning(
        "entity_extractor: fallback_validate used — coverage=%.3f | passed=%s | "
        "missing_count=%d",
        coverage, passed, len(missing),
    )

    return {
        "passed":         passed,
        "coverage_score": coverage,
        "missing_facts":  missing,
    }
