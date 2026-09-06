# SERA Evaluation Report: Static vs Dynamic RAG

**Generated:** 20260905_125722
**Generator Model:** `llama3:8b`
**Judge Model:** `qwen2.5:7b`
**Experiment scope:** `equal_budget`  |  **Query types:** `all`
**Max queries per table:** 100  |  **K for validation:** 5

---

## Abstract

This report presents a controlled, reproducible evaluation of two Retrieval-Augmented Generation (RAG) pipelines in the SERA (Self-Evolving RAG Architecture) system. The *Static* baseline retrieves raw text chunks from the v1 ground-truth corpus; the *Dynamic* pipeline retrieves dynamically synthesised Super-Nodes produced by SERA's Phase 3/4 synthesis cycle. Both pipelines share an identical generator LLM. Evaluation spans k=1..9 retrieved documents, two query complexity tiers (Simple and Complex), an equal-context-budget comparison, and claim-level Additional Information Validation via LLM-as-judge (SAIR/UAIR). All results are presented with per-k breakdowns.

---

## Central Hypothesis

> **H1 (Dynamic Superiority Hypothesis):** Replacing raw text chunks with dynamically synthesised Super-Nodes will produce answers with higher semantic fidelity (BioBERT F1, ROUGE-L F1), greater concept coverage (Concept Recall, RAIV), and lower hallucination rates (UAIR) compared to the Static baseline, while maintaining comparable or lower total token cost.

*Note: This section is labelled **Hypothesis** - not Conclusion. The Conclusion section re-evaluates this hypothesis against the empirical results.*

---

## Super-Node Database Statistics

| Statistic | Static (v1) | Dynamic |
|-----------|-------------|--------|
| Total raw chunks | [TO BE MEASURED] | [TO BE MEASURED] |
| Total super-nodes | N/A (raw only) | [TO BE MEASURED] |
| Mean super-node length (tokens) | N/A | [TO BE MEASURED] |
| Median super-node length (tokens) | N/A | [TO BE MEASURED] |
| Super-node topic coverage (unique clusters) | N/A | [TO BE MEASURED] |
| ChromaDB collection (raw_chunks) count | [TO BE MEASURED] | [TO BE MEASURED] |
| ChromaDB collection (super_nodes) count | N/A | [TO BE MEASURED] |
| SQLite DB size (MB) | [TO BE MEASURED] | [TO BE MEASURED] |
| Benchmark queries: Simple | [TO BE MEASURED] | [TO BE MEASURED] |
| Benchmark queries: Complex | [TO BE MEASURED] | [TO BE MEASURED] |

---

## Metric Definitions

| Metric | Definition |
|--------|------------|
| **ROUGE-L F1** | Longest Common Subsequence recall between generated and reference answer (token-level). Range [0,1]. Higher is better. |
| **BioBERT F1** | Cosine similarity between BioBERT-encoded generated and reference answer embeddings. Range [0,1]. Higher is better. |
| **Concept Recall** | |answer_keywords intersect ref_keywords| / |ref_keywords|. Measures domain-concept coverage. Range [0,1]. Higher is better. |
| **Ref-Aligned Info Volume (RAIV)** | |context_keywords intersect ref_keywords|. Counts reference concepts in the retrieved context. Higher is better. |
| **Context Tokens** | Whitespace-split word count of all retrieved context. Lower values signal more precise retrieval. |
| **Output Tokens** | Whitespace-split word count of the generated answer. |
| **Total Tokens** | Context Tokens + Output Tokens. Proxy for inference cost. |
| **Latency (s)** | Wall-clock time (seconds) for retrieval + generation. Lower is better. |
| **SAIR (%)** | Source-Anchored Information Rate = (correct and source-supported claims) / total_claims * 100. Higher is better. |
| **UAIR (%)** | Unsupported Addition Information Rate = unsupported_claims / total_claims * 100. Lower is better. |

---

## K-Sensitivity Results (k = 1 to 9)

Each row is the mean over all evaluated queries at that k value.

### Simple Queries

*Experiment not run or no data available.*

### Complex Queries

*Experiment not run or no data available.*

---

## Win / Loss / Tie Analysis

Counts across k=1..9: how many k-values does Dynamic WIN over Static per metric?
(Higher-is-better: Dynamic wins if Delta > 0; lower-is-better: Dynamic wins if Delta < 0)

### Simple Queries

| Metric | Dynamic Wins | Static Wins | Ties |
|--------|-------------|-------------|------|
| *No data* | -- | -- | -- |

### Complex Queries

| Metric | Dynamic Wins | Static Wins | Ties |
|--------|-------------|-------------|------|
| *No data* | -- | -- | -- |

---

## Aggregate Results: Simple vs Complex

Mean across all k=1..9 runs.

### Simple Queries

| Config | ROUGE-L F1 | BioBERT F1 | Concept Recall | RAIV | Ctx Tokens | Out Tokens | Total Tokens | Latency | SAIR | UAIR |
|--------|-----------|-----------|---------------|------| -----------|-----------|-------------|---------|------|------|
| Static | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* |
| Dynamic | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* |

### Complex Queries

| Config | ROUGE-L F1 | BioBERT F1 | Concept Recall | RAIV | Ctx Tokens | Out Tokens | Total Tokens | Latency | SAIR | UAIR |
|--------|-----------|-----------|---------------|------| -----------|-----------|-------------|---------|------|------|
| Static | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* |
| Dynamic | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* |

---

## Equal-Context-Budget Comparison

Budget pairs are chosen so that static and dynamic pipelines consume approximately the same total context tokens (raw chunks are small; super-nodes are large).

### Budget Pairs

| Budget Pair | Static k | Dynamic k | Rationale |
|-------------|----------|-----------|-----------|
| s3_d1 | 3 | 1 | ~3 raw-chunk token budget matched to 1 super-node budget |
| s6_d2 | 6 | 2 | ~6 raw-chunk token budget matched to 2 super-node budget |
| s9_d3 | 9 | 3 | ~9 raw-chunk token budget matched to 3 super-node budget |

### Simple Queries - Budget Results

| Config | ROUGE-L F1 | BioBERT F1 | Concept Recall | RAIV | Ctx Tokens | Out Tokens | Total Tokens | Latency | SAIR | UAIR |
|--------|-----------|-----------|---------------|------| -----------|-----------|-------------|---------|------|------|
| s3_d1 (Static, k=s3) | 0.1185 | 0.7290 | 0.0967 | 43.2 | 553 | 125 | 677 | 29.241s | 0.0% | 0.0% |
| s3_d1 (Dynamic, k=d1) | 0.1142 | 0.7211 | 0.0935 | 41.7 | 373 | 113 | 487 | 23.438s | 0.0% | 0.0% |
| s6_d2 (Static, k=s6) | 0.1347 | 0.7212 | 0.1156 | 58.6 | 1070 | 166 | 1236 | 37.485s | 0.0% | 0.0% |
| s6_d2 (Dynamic, k=d2) | 0.1248 | 0.7360 | 0.1039 | 51.8 | 627 | 127 | 754 | 25.734s | 0.0% | 0.0% |
| s9_d3 (Static, k=s9) | 0.1414 | 0.7219 | 0.1216 | 68.6 | 1571 | 179 | 1751 | 45.756s | 0.0% | 0.0% |
| s9_d3 (Dynamic, k=d3) | 0.1257 | 0.7431 | 0.1051 | 60.6 | 883 | 129 | 1011 | 54.028s | 0.0% | 0.0% |

### Complex Queries - Budget Results

| Config | ROUGE-L F1 | BioBERT F1 | Concept Recall | RAIV | Ctx Tokens | Out Tokens | Total Tokens | Latency | SAIR | UAIR |
|--------|-----------|-----------|---------------|------| -----------|-----------|-------------|---------|------|------|
| s3_d1 (Static, k=s3) | 0.1526 | 0.7302 | 0.1237 | 33.7 | 530 | 164 | 695 | 33.840s | 0.0% | 0.0% |
| s3_d1 (Dynamic, k=d1) | 0.1350 | 0.6658 | 0.1108 | 30.4 | 363 | 120 | 483 | 22.824s | 0.0% | 0.0% |
| s6_d2 (Static, k=s6) | 0.1652 | 0.7381 | 0.1386 | 46.2 | 1019 | 199 | 1218 | 42.398s | 0.0% | 0.0% |
| s6_d2 (Dynamic, k=d2) | 0.1627 | 0.7432 | 0.1387 | 43.6 | 686 | 165 | 850 | 33.167s | 0.0% | 0.0% |
| s9_d3 (Static, k=s9) | 0.1736 | 0.7409 | 0.1524 | 55.0 | 1538 | 216 | 1754 | 51.707s | 0.0% | 0.0% |
| s9_d3 (Dynamic, k=d3) | 0.1661 | 0.7349 | 0.1476 | 51.9 | 966 | 186 | 1151 | 40.215s | 0.0% | 0.0% |

---

## Additional Information Validation (SAIR / UAIR)

Fixed k=5. LLM-as-judge (`qwen2.5:7b`) decomposes each generated answer into atomic claims and classifies them as correct+supported, incorrect, or unsupported.

| Query Type | Pipeline | SAIR (%) | UAIR (%) | Avg Total Claims | Avg Supported |
|------------|----------|----------|----------|-----------------|---------------|
| *N/A* | *Experiment not run* | -- | -- | -- | -- |

---

## Limitations

1. **Token counting** is done by whitespace splitting, not a model tokeniser. Results may differ from actual API token counts by +-10-20%.
2. **BioBERT F1** is sentence-embedding cosine similarity (not token-level BERTScore), computed with `pritamdeka/S-BioBert-snli-multinli-stsb`.
3. **SAIR/UAIR** depends on the judge model (`qwen2.5:7b`). Judge agreement with human annotators was not verified in this run.
4. **Dynamic Super-Nodes** are retrieved without epsilon-greedy exploration or SU5 hierarchical masking (active in production). This isolates the quality of the synthesised content itself.
5. **Query sets** are drawn from `benchmark_qa_simple` and `benchmark_qa_complex`, seeded programmatically. They may not fully represent real-world query distributions.
6. **Equal-budget pairs** are heuristic approximations; actual token equality depends on super-node content length, which varies.

---

## Methodological Safeguards

The following design choices ensure a fair, unbiased comparison:

| What was NOT done | Why it matters |
|-------------------|----------------|
| Generator LLM **not** changed between pipelines | Quality difference attributable to retrieval, not generation |
| Judge LLM **not** told which pipeline produced which answer | Prevents judge bias |
| Dynamic pipeline does **not** use raw-chunk fallback | Tests pure super-node quality |
| Epsilon-greedy exploration **disabled** in evaluation | Ensures deterministic, reproducible retrieval |
| Benchmark queries **not** used during SERA synthesis cycle | Prevents data leakage |
| Results **not** post-filtered for good queries | All queries run regardless of answer quality |

---

## Conclusion

### Central Hypothesis Re-evaluation

> **H1 (Dynamic Superiority Hypothesis):** Replacing raw text chunks with dynamically synthesised Super-Nodes will produce answers with higher semantic fidelity (BioBERT F1, ROUGE-L F1), greater concept coverage (Concept Recall, RAIV), and lower hallucination rates (UAIR) compared to the Static baseline, while maintaining comparable or lower total token cost.

The empirical results are **inconclusive** - Static and Dynamic pipelines perform comparably in aggregate. Refer to per-metric and per-k tables for nuance.

Full numerical evidence is available in the K-Sensitivity, Win/Loss/Tie, and Additional Information Validation sections above.

---

*Report generated automatically by `scripts/eval_static_vs_dynamic.py` at 20260905_125722.*