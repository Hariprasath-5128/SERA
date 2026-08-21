"""
app/validation/validator.py
-----------------------------
SU2 — LLM-as-Judge validation.

Uses the SUMMARIZER_LLM_MODEL (Qwen-2.5-32B-Instruct) in strict JSON mode
to evaluate the generated summary against the raw source material.

Detects nuanced factual coverage that simple regex or string-matching misses
(e.g., paraphrased facts, implied relationships).

Returns a ValidationResult which determines if the synthesizer accepts the
draft or loops for a retry.
"""

import json
import logging
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError as PydanticValidationError

from app import config
from app.generation.llm_client import llm_call
from app.validation import entity_extractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class ValidationOutput(BaseModel):
    """The strict JSON schema we expect the LLM to return."""
    passed: bool
    coverage_score: float
    missing_facts: list[str]


@dataclass
class ValidationResult:
    """The object returned to the Synthesizer orchestrator."""
    passed: bool
    coverage_score: float
    missing_facts: list[str]
    source_fact_count: int
    summary_fact_count: int


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

LLM_JUDGE_PROMPT = """\
You are a strict grading algorithm.
Compare the Source Material against the Generated Summary.
1. Extract the core facts from the Source Material.
2. Check if EACH fact is present in the Summary.
3. Calculate coverage = (Facts Present / Total Source Facts).
4. Return JSON strictly matching this schema:
{{"passed": true/false (true if >= 0.90), "coverage_score": float, "missing_facts": [list of strings]}}

Source Material:
{source_text}

Summary:
{summary_text}
"""

# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------

def validate_entailment(source_text: str, summary: str) -> ValidationOutput:
    """
    Core LLM-as-judge call.

    Sends the source and summary to the LLM in JSON mode with temperature=0.0
    for deterministic scoring.
    """
    prompt = LLM_JUDGE_PROMPT.format(
        source_text=source_text,
        summary_text=summary
    )

    response_text = llm_call(
        system_prompt="You are a strict factual validation engine.",
        user_prompt=prompt,
        temperature=0.0,
        json_mode=True,
    )

    import re
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```"):
        cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text)
        cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

    try:
        result_dict = json.loads(cleaned_text)
        # Ensure it conforms to our Pydantic schema
        return ValidationOutput(**result_dict)
    except (json.JSONDecodeError, PydanticValidationError) as exc:
        logger.error(
            "validator: LLM returned malformed JSON or failed schema: %s\nResponse: %s",
            exc, response_text
        )
        raise ValueError("Invalid LLM judge output") from exc


def validate(source_text: str, summary: str) -> ValidationResult:
    """
    Main entry point called by synthesizer.py (Step 6).

    Uses the strict LLM-as-judge first. If JSON parsing fails,
    it falls back to the entity_extractor's overlap check.
    """
    logger.debug("validator: running LLM-as-judge coverage check")
    
    try:
        llm_result = validate_entailment(source_text, summary)
        
        logger.info(
            "validator: LLM check finished | passed=%s | score=%.3f | missing=%d",
            llm_result.passed, llm_result.coverage_score, len(llm_result.missing_facts)
        )
        
        return ValidationResult(
            passed=llm_result.passed,
            coverage_score=llm_result.coverage_score,
            missing_facts=llm_result.missing_facts,
            source_fact_count=0,
            summary_fact_count=0,
        )
        
    except ValueError:
        logger.warning("validator: LLM judge failed (malformed JSON). Running strict entity fallback.")
        fallback = entity_extractor.fallback_validate(source_text, summary)
        
        logger.info(
            "validator: entity check finished | passed=%s | score=%.3f | missing=%d",
            fallback["passed"], fallback["coverage_score"], len(fallback["missing_facts"])
        )
        
        return ValidationResult(
            passed=fallback["passed"],
            coverage_score=fallback["coverage_score"],
            missing_facts=fallback["missing_facts"],
            source_fact_count=0,
            summary_fact_count=0,
        )
