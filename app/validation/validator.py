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

    try:
        result_dict = json.loads(response_text)
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

    Includes fallback logic: if the LLM judge fails to return valid JSON,
    it falls back to entity_extractor's rough overlap check to prevent a
    total pipeline collapse.
    """
    logger.debug("validator: starting LLM entailment check")
    
    try:
        output = validate_entailment(source_text, summary)
        
        # Override 'passed' based on our configured threshold, in case the LLM 
        # did the math wrong.
        passed = output.coverage_score >= config.VALIDATION_COVERAGE_THRESHOLD
        
        logger.info(
            "validator: LLM judge finished | passed=%s | score=%.3f | missing=%d",
            passed, output.coverage_score, len(output.missing_facts)
        )
        
        return ValidationResult(
            passed=passed,
            coverage_score=output.coverage_score,
            missing_facts=output.missing_facts,
            source_fact_count=0,   # Not provided by LLM schema; left as 0
            summary_fact_count=0
        )
        
    except Exception as exc:
        logger.warning(
            "validator: LLM entailment failed (%s), falling back to entity extraction",
            exc
        )
        # Fallback SU2 mechanism using the entity_extractor
        fallback = entity_extractor.fallback_validate(source_text, summary)
        
        return ValidationResult(
            passed=fallback["passed"],
            coverage_score=fallback["coverage_score"],
            missing_facts=fallback["missing_facts"],
            source_fact_count=0,
            summary_fact_count=0
        )
