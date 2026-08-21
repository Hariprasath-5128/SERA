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
    "You are a highly precise medical knowledge architect. "
    "Your goal is to create a comprehensive, self-contained knowledge entry "
    "based ONLY on the provided source passages. "
    "CRITICAL RULE: You MUST preserve ALL of the following exactly as they "
    "appear in the source material — drug names, medication dosages, numerical "
    "values, percentages, measurement units, specific symptom names, side effect "
    "names, procedure names, and clinical terminology. "
    "Do NOT abstract, generalize, paraphrase, or omit any of these entities. "
    "If a source says '5mg of Lisinopril', your output must also say '5mg of Lisinopril'. "
    "Summarizing granular facts into vague language will result in rejection."
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
1. Synthesize a detailed, comprehensive, and self-contained medical knowledge entry that definitively answers the core question.
2. MANDATORY: Preserve ALL exact medical entities verbatim from the source — drug names, dosages (e.g. "10mg", "500ml"), percentages, symptom names, side-effect names, procedure names, lab values, and clinical measurements.
3. Include ALL key facts, numbers, dates, explanations, and critical nuances found in the source material.
4. Do NOT abstract specific facts into generic language. "Nausea, dizziness, and a dry cough" is correct. "Various side effects" is NOT acceptable.
5. There is no strict length limit; prioritize thoroughness and clinical precision over brevity.
6. Do not include conversational filler or external knowledge.
7. Output ONLY the final synthesis."""

# ---------------------------------------------------------------------------
# Re-synthesis prompt — used when the LLM-as-judge rejects the first draft
# (SU2 retry loop in synthesizer.py)
# ---------------------------------------------------------------------------

RESYNTH_PROMPT_TEMPLATE = """\
Your previous synthesis was rejected. Missing facts: {missing_facts}

Re-synthesize ensuring ALL of the above facts are present.
CRITICAL: Every specific medical entity (drug names, exact dosages, numerical values,
symptom names, side-effect names) from the source material MUST appear verbatim in your output.
Do not abstract or paraphrase any clinical fact.

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
    "Never add external knowledge or invented facts. "
    "CRITICAL RULE: ALL specific medical entities — drug names, exact dosages "
    "(e.g. '10mg', '2x daily'), numerical lab values, percentages, specific symptom "
    "names, side-effect names, and clinical procedure names — MUST be preserved "
    "exactly as they appear in the source passages. Do NOT generalize them into vague "
    "language. Every specific clinical fact must survive the merge verbatim."
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

CRITICAL PRESERVATION RULE: Every specific medical entity mentioned in the source material —
including drug names, exact dosages, numerical values, symptom names, and side-effect names —
MUST appear verbatim in the relevant section. Do not replace specific facts with vague summaries.

## Overview
Provide a concise definition and general description of the disease or medical topic.
Include any specific prevalence statistics or epidemiological numbers from the source.

## Symptoms
List and describe ALL symptoms, signs, and clinical presentations mentioned in the source.
Use the exact clinical names as they appear in the source (e.g., "dyspnea", "diaphoresis").

## Causes
Describe all known causes, risk factors, genetic factors, and mechanisms of disease.
Preserve all specific names (e.g., gene names, pathogen names, exact risk percentages).

## Effects
Describe the physiological effects, complications, long-term impact, and prognosis.
Include all specific organ systems, exact complication rates, and numerical outcomes.

## Treatments
Describe ALL treatment options, medications (with exact dosages), surgical interventions,
and management strategies. Drug names and doses must be exact (e.g., "Metformin 500mg twice daily").

## Related Information
Include any additional relevant information such as epidemiology, diagnosis methods, or prevention strategies.
Preserve all specific diagnostic thresholds (e.g., "HbA1c > 6.5%") verbatim.

Output ONLY the structured encyclopaedia entry using the headings above. Do not include conversational filler."""

