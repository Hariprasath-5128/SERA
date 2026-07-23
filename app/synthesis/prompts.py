"""
app/synthesis/prompts.py
--------------------------
Centralized prompt templates for the Phase 4 Synthesis Agent.

Why a separate module?
-----------------------
Keeping prompts isolated from the orchestration logic in synthesizer.py
allows prompt tuning without touching the core synthesis pipeline.
Any A/B testing, versioning, or domain-specific overrides happen here.

Three templates are defined:

  SYSTEM_PROMPT
    The persona and hard constraints given to the LLM on every call.
    Instructs the model to act as a strict knowledge architect and to
    use ONLY the provided source passages.

  USER_PROMPT_TEMPLATE
    First-attempt synthesis prompt. Receives:
      {canonical_query}     - the representative cluster question
      {source_chunks_text}  - deduplicated source passages (from chunk_fetcher.py)

  RESYNTH_PROMPT_TEMPLATE
    Retry prompt used when LLM-as-judge (validator.py) rejects the summary.
    Receives:
      {missing_facts}       - list of facts flagged as absent by the judge
      {source_chunks_text}  - same deduplicated source passages
    Forces the model to directly address the coverage gaps before outputting
    a new summary.
"""

# ---------------------------------------------------------------------------
# System Prompt — defines the model's persona and hard output constraints
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a highly precise knowledge architect. "
    "Your goal is to create a comprehensive, self-contained knowledge entry "
    "based ONLY on the provided source passages."
)

# ---------------------------------------------------------------------------
# First-attempt synthesis prompt
# ---------------------------------------------------------------------------

USER_PROMPT_TEMPLATE = """\
Task: Users are frequently asking variations of this core question:
"{canonical_query}"

Source Material:
{source_chunks_text}

Instructions:
1. Synthesize a detailed, comprehensive, and self-contained summary that definitively answers the core question.
2. Include ALL key facts, numbers, dates, explanations, and critical nuances found in the source material.
3. There is no strict length limit; prioritize thoroughness and clarity over brevity.
4. Do not include conversational filler or external knowledge.
5. Output ONLY the final synthesis."""

# ---------------------------------------------------------------------------
# Re-synthesis prompt — used when the LLM-as-judge rejects the first draft
# (SU2 retry loop in synthesizer.py)
# ---------------------------------------------------------------------------

RESYNTH_PROMPT_TEMPLATE = """\
Your previous synthesis was rejected. Missing facts: {missing_facts}

Re-synthesize ensuring ALL of the above facts are present.
Source Material:
{source_chunks_text}"""

# ---------------------------------------------------------------------------
# Meta-Node Prompt — used ONLY when hierarchy_merger creates a parent node
# from two or more child super-nodes (SU11 Ground-Truth Re-Synthesis)
#
# This structured prompt enforces a disease-centric hierarchical layout:
#   Disease Name → [ Overview | Symptoms | Causes | Effects | Treatments | ... ]
# so that parent meta-nodes serve as organised disease encyclopaedia entries.
# ---------------------------------------------------------------------------

META_NODE_SYSTEM_PROMPT = (
    "You are a senior medical knowledge architect. "
    "Your task is to merge multiple related medical knowledge entries into a single, "
    "well-structured disease encyclopaedia entry. "
    "Use ONLY information present in the provided source passages. "
    "Never add external knowledge or invented facts."
)

META_NODE_PROMPT_TEMPLATE = """\
Multiple related medical super-nodes are being merged into a single parent knowledge entry.

Child Node Topics:
{child_queries}

Combined Source Material:
{source_chunks_text}

Instructions:
You MUST structure your output using ALL of the following section headings.
If information for a section is not present in the source material, write "Information not available in source material."

## Overview
Provide a concise definition and general description of the disease or medical topic.

## Symptoms
List and describe all symptoms, signs, and clinical presentations mentioned in the source.

## Causes
Describe all known causes, risk factors, genetic factors, and mechanisms of disease.

## Effects
Describe the physiological effects, complications, long-term impact, and prognosis.

## Treatments
Describe all treatment options, medications, surgical interventions, and management strategies.

## Related Information
Include any additional relevant information such as epidemiology, diagnosis methods, or prevention strategies.

Output ONLY the structured encyclopaedia entry using the headings above. Do not include conversational filler."""
