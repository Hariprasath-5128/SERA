# SERA Evaluation Results

This folder contains all benchmark evaluation runs organized by purpose.

| Folder | Queries | Purpose | Status |
|--------|---------|---------|--------|
| 01_legacy_pilot_runs__less_than_20q | ~15q | Early pilot tests (pre-200q benchmark) | Legacy - do NOT use in paper |
| 02_k_sensitivity__200q__k1_to_k9 | 200q | K-sensitivity sweep: static vs dynamic at k=1..9 | VALID - primary paper experiment |
| 03_equal_budget__WRONG_PAIRS__200q__s1d3_s2d6_s3d9 | 200q | Equal budget with INVERTED pairs (dynamic had 5x token advantage) | INVALID - do NOT use in paper |
| 04_equal_budget__WRONG_RATIO__200q__s3d1_s6d2_s9d3 | 200q | Equal budget with assumed 3x ratio (actual ratio is 2.23x) | INCORRECT RATIO - reference only |
| 05_equal_budget__CORRECT__150q__s2d1_s4d2_s9d4 | 150q | Equal budget with data-driven ratio (2.23x from ChromaDB stats) | VALID - use in paper |
| 06_additional_info_SAIR_UAIR__200q | 200q | LLM-judge SAIR/UAIR claim validation at k=5 | VALID - use in paper |
| 07_legacy_phase_experiments__pre_supernode | varies | Early phase-by-phase evaluation before full synthesis | Legacy reference only |
| 08_test_runs__incomplete_and_partial | <10q | Aborted or partial test runs | DO NOT USE |

## Key Metrics
- **ROUGE-L F1**: Lexical overlap against ground truth
- **BioBERT F1**: Semantic similarity (domain-specific)
- **Concept Recall**: Answer concept coverage = |answer_kw ∩ ref_kw| / |ref_kw|
- **RAIV**: Ref-Aligned Info Volume = |context_kw ∩ ref_kw| (retrieval quality)
- **SAIR**: Source-Anchored Information Rate = % claims supported by context
- **UAIR**: Unsupported Addition Information Rate = % hallucinated claims

## Equal Budget Pair Derivation
From actual ChromaDB stats (56,407 raw chunks, 336 super nodes):
- Avg raw chunk: **176 words**
- Avg super node: **394 words**
- Ratio: **2.23x** (1 super node = 2.23 raw chunks)

Correct pairs: s2_d1 (~10% diff), s4_d2 (~10% diff), s9_d4 (~0.8% diff - near-perfect)
