# Self-Evolving Write-Back RAG — Engineering Implementation Blueprint
## VERSION: v3.1 (14-improvement upgrade — adds SU14 Synthesis Fidelity Bound check)

> **Role:** Senior ML Systems Architect  
> **Stack:** Python · FastAPI · ChromaDB · SQLite/PostgreSQL · SentenceTransformers · OpenAI API · APScheduler · Docker  
> **Audience:** Engineer building from scratch, research-grade quality

---

SERA/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── README.md
├── .env.example
│
├── app/
│   ├── main.py
│   ├── config.py
│   │
│   ├── api/
│   │   └── routes/
│   │       ├── ingest.py
│   │       ├── query.py
│   │       └── admin.py
│   │
│   ├── db/
│   │   ├── chroma_client.py
│   │   ├── sqlite_client.py
│   │   ├── super_node_store.py
│   │   └── migrations/
│   │       └── 001_initial.sql
│   │
│   ├── ingestion/
│   │   ├── dataset_loader.py
│   │   ├── url_extractor.py
│   │   ├── web_fetcher.py
│   │   ├── html_cleaner.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── benchmark_loader.py
│   │   └── ingestor.py
│   │
│   ├── retrieval/
│   │   ├── retriever.py
│   │   ├── router.py
│   │   └── score_booster.py
│   │
│   ├── generation/
│   │   └── generator.py
|   |   └── llm_client.py
│   │
│   ├── logging_/
│   │   ├── normalizer.py
│   │   └── query_logger.py
│   │
│   ├── middleware/
│   │   └── query_interceptor.py
│   │
│   ├── patterns/
│   │   └── finder.py
│   │
│   ├── scheduler/
│   │   ├── scheduler.py
│   │   └── jobs/
│   │       ├── pattern_finder.py
│   │       └── maintenance_job.py
│   │
│   ├── synthesis/
│   │   ├── prompts.py
│   │   ├── chunk_fetcher.py
│   │   └── synthesizer.py
│   │
│   ├── validation/
│   │   ├── validator.py
│   │   └── entity_extractor.py
│   │
│   ├── maintenance/
│   │   ├── decay_scorer.py
│   │   ├── staleness_checker.py
│   │   └── hierarchy_merger.py
│   │
│   ├── models/
│   │   ├── chunk.py
│   │   ├── query_cluster.py
│   │   ├── super_node.py
│   │   ├── benchmark_pair.py
│   │   ├── synthesis_job.py
│   │   └── retrieval_result.py
│   │
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── logger.py
│   │   ├── timers.py
│   │   └── helpers.py
│   │
│   └── exceptions/
│       ├── synthesis_error.py
│       ├── validation_error.py
│       └── retrieval_error.py
│
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   └── sample_medquad_rows.json
│   │
│   ├── test_phase1_ingestion.py
│   ├── test_phase2_logging.py
│   ├── test_phase3_scheduler.py
│   ├── test_phase4_synthesis.py
│   ├── test_phase5_retrieval.py
│   └── test_phase6_maintenance.py
│
├── scripts/
│   ├── benchmark.py
│   ├── seed_data.py
│   ├── export_container.sh
│   └── rebuild_indexes.py
│
├── data/
│   ├── chroma/
│   ├── sqlite/
│   │   └── rag.db
│   │
│   ├── cache/
│   │   ├── hf_cache/
│   │   ├── embeddings/
│   │   └── responses/
│   │
│   ├── logs/
|   └── results
│   └── checkpoints/
│
├── models/
│   ├── embedding/
│   │   └── BAAI/bge-m3/
│   │
│   ├── summarizer/
│   │   ├── llama/
│   │   ├── qwen/
│   │   └── mistral/
│   │
│   └── reranker/
│
├── notebooks/
│   ├── experiments.ipynb
│   └── ablations.ipynb
│
└── docs/
├── architecture/
│   ├── phase1.md
│   ├── phase2.md
│   ├── phase3.md
│   ├── phase4.md
│   ├── phase5.md
│   └── phase6.md
│
├── diagrams/
└── benchmark_results/



## UPGRADE MANIFEST (v1 → v3)

| SU# | Title | Affected Phase(s) | Affected File(s) |
|-----|-------|-------------------|-----------------|
| SU1 | Token Window Collapse — Semantic Deduplication | Phase 4 | `synthesis/chunk_fetcher.py` |
| SU2 | Rigid Fact Validation — LLM-as-Judge | Phase 4 | `validation/validator.py`, `validation/entity_extractor.py` |
| SU3 | Linear Time Decay Flaw — Half-Life Decay | Phase 6 | `maintenance/decay_scorer.py` |
| SU4 | Self-Fulfilling Feedback Loop — ε-Greedy Retrieval | Phase 5 | `retrieval/retriever.py` |
| SU5 | Parent-Child Vector Collision — Hierarchical Masking | Phase 5, Phase 6 | `retrieval/retriever.py` |
| SU6 | First-Query Centroid Drift — Running Centroid Update | Phase 2 | `logging_/query_logger.py` |
| SU7 | Infinite Merger Loop — in_hierarchy Lock Flag | Phase 6 | `maintenance/hierarchy_merger.py` |
| SU8 | Cascading Amnesia — Soft-Staleness Validation | Phase 6 | `maintenance/staleness_checker.py` |
| SU9 | Knowledge JPEG Artifacting — Ground-Truth Anchoring | Phase 2, Phase 4 | `middleware/query_interceptor.py` |
| SU10 | ChromaDB Concurrency Illusion — SQLite Atomic Counters | Phase 5, Phase 6 | `retrieval/retriever.py`, `maintenance/decay_scorer.py` |
| SU11 | Summary-of-Summaries Merger — Ground-Truth Re-Synthesis in Hierarchy Merger | Phase 6 | `maintenance/hierarchy_merger.py` |
| SU12 | Frozen Snapshot on Boolean Flag — Re-Synthesis Threshold Counter | Phase 3, Phase 4 | `db schema`, `patterns/finder.py`, `synthesis/synthesizer.py` |
| SU13 | Upward Staleness Blindness — Child-Revision Propagation | Phase 4, Phase 6 | `synthesis/synthesizer.py`, `maintenance/staleness_checker.py` |
| SU14 | Semantic Drift from Source — Synthesis Fidelity Bound Check | Phase 4 | `synthesis/synthesizer.py` |

---

## PHASE 1 — Standard RAG Baseline (Ingestion)

### Goal
Establish the foundational retrieval layer using the **MedQuAD HuggingFace dataset** as the knowledge source. Each row carries a `document_url` pointing to a NLM/NIH article; the pipeline crawls that URL, cleans the HTML, chunks the article text, embeds it, and stores it in ChromaDB. The `question`/`answer` columns are **not** ingested — they are stored separately as evaluation pairs in a `benchmark_pairs` SQLite table for Recall@k, BLEU, ROUGE, and factual coverage scoring.

### Components
```
app/
  ingestion/
    dataset_loader.py   # HF datasets → stream MedQuAD rows
    url_extractor.py    # Extract + deduplicate document_url per row
    web_fetcher.py      # HTTP GET article HTML (rate-limited, retried)
    html_cleaner.py     # Strip nav/ads/boilerplate → clean article text
    chunker.py          # Semantic + recursive text splitter
    embedder.py         # SentenceTransformer wrapper
    ingestor.py         # Orchestrates full pipeline (with Idempotent Deduplication) → ChromaDB
    benchmark_loader.py # Write question/answer pairs → SQLite benchmark_pairs
  retrieval/
    retriever.py        # Top-k ChromaDB retrieval
  generation/
    generator.py        # LLM answer generation (OpenAI)
  api/
    routes/ingest.py    # POST /ingest/medquad (trigger dataset load)
    routes/query.py     # POST /query
  db/
    chroma_client.py    # ChromaDB singleton
  config.py             # All hyperparameters
```

### Inputs
- MedQuAD dataset stream: `datasets.load_dataset("lavita/MedQuAD", split="train", streaming=True)`
- Each row fields used:
  - `document_url` → crawl target
  - `document_id` → dedup key
  - `question_focus` → article topic tag (metadata)
  - `umls_semantic_group` → domain tag (metadata, e.g. "Disorders")
  - `question` + `answer` → benchmark pairs only (NOT ingested into ChromaDB)

### Outputs
- ChromaDB collection: `raw_chunks` (vectors + metadata)
- SQLite table: `benchmark_pairs` (question, answer, document_url, question_type)
- JSON status payload from ingest endpoint

### Database Schema

**ChromaDB Collection: `raw_chunks`**
```
Collection: raw_chunks
  id:          str   → "{document_id}_{chunk_idx}"
  embedding:   List[float] (1024-dim, BAAI/bge-m3)
  document:    str   → clean article text chunk
  metadata:
    document_id:          str   → MedQuAD document_id (e.g. "0000613")
    source_url:           str   → original document_url
    question_focus:       str   → e.g. "Mabry syndrome"
    umls_semantic_group:  str   → e.g. "Disorders"
    chunk_idx:            int
    type:                 str   = "raw_chunk"
    created_at:           ISO8601 timestamp
    token_count:          int
    updated_at:           ISO8601 timestamp  ← for staleness check
```

**SQLite Table: `benchmark_pairs`**
```sql
CREATE TABLE IF NOT EXISTS benchmark_pairs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id      TEXT NOT NULL UNIQUE,   -- e.g. "0000613-1"
    document_id      TEXT NOT NULL,
    document_url     TEXT NOT NULL,
    question_focus   TEXT,
    question_type    TEXT,                   -- "information"|"frequency"|"treatment" etc.
    question         TEXT NOT NULL,
    answer           TEXT NOT NULL,
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_bp_doc ON benchmark_pairs(document_id);
CREATE INDEX IF NOT EXISTS idx_bp_type ON benchmark_pairs(question_type);
```

### Data Structures
```python
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

@dataclass
class Chunk:
    chunk_id: str              # "{document_id}_{chunk_idx}"
    text: str
    document_id: str
    source_url: str
    question_focus: str
    umls_semantic_group: str
    token_count: int
    embedding: Optional[List[float]] = None

@dataclass
class BenchmarkPair:
    question_id: str
    document_id: str
    document_url: str
    question_focus: str
    question_type: str
    question: str
    answer: str

@dataclass
class QueryRequest:
    query: str
    top_k: int = 5

@dataclass
class QueryResponse:
    answer: str
    source_chunks: List[str]
    latency_ms: float
```

### Algorithm
```
INGESTION PIPELINE (Knowledge Source):
1. Stream MedQuAD from HuggingFace:
   datasets.load_dataset("lavita/MedQuAD", split="train", streaming=True)
2. For each row:
   a. Extract document_url, document_id, question_focus, umls_semantic_group
   b. Deduplicate on document_id (many rows share same URL → one crawl per doc)
   c. Store question/answer → INSERT INTO benchmark_pairs (skip if exists)
3. For each unique document_url (deduplicated set):
   a. web_fetcher.get(url) → raw HTML (rate-limit: 1 req/sec, retry 3x)
   b. html_cleaner.clean(html) → article plain text (strip nav, footer, ads)
   c. chunker.split(text) → List[Chunk] (~500 tokens, 50 overlap)
   d. embedder.encode([c.text for c in chunks], batch_size=64)
   e. chroma_client.raw_chunks.upsert(chunks)

QUERY PIPELINE:
4. Embed query → BAAI/bge-m3
5. ChromaDB top-k retrieval from raw_chunks
6. Concatenate chunk texts as context
7. Call Meta-Llama-3.1-8B-Instruct with context + query
8. Return answer + source metadata + latency
```

### Pseudocode
```python
# dataset_loader.py
def stream_medquad() -> Iterator[MedQuADRow]:
    ds = load_dataset("lavita/MedQuAD", split="train", streaming=True)
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

# ingestor.py
def run_medquad_ingestion(limit: Optional[int] = None) -> dict:
    seen_docs: set[str] = set()
    chunks_total, pairs_total = 0, 0

    for i, row in enumerate(dataset_loader.stream_medquad()):
        if limit and i >= limit:
            break

        # Always store benchmark pair (question/answer)
        benchmark_loader.upsert_pair(BenchmarkPair(**asdict(row)))
        pairs_total += 1

        # Skip if document already ingested
        if row.document_id in seen_docs:
            continue
        seen_docs.add(row.document_id)

        # Crawl → clean → chunk → embed → store
        try:
            html = web_fetcher.get(row.document_url)          # rate-limited
            text = html_cleaner.clean(html)
            if len(text) < 100:
                log.warning("thin_content", url=row.document_url)
                continue
            chunks = chunker.split(text, document_id=row.document_id,
                                   source_url=row.document_url,
                                   question_focus=row.question_focus,
                                   umls_semantic_group=row.umls_semantic_group)
            embeddings = embedder.encode([c.text for c in chunks], batch_size=64)
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
            chroma_client.raw_chunks.upsert(
                ids=[c.chunk_id for c in chunks],
                embeddings=[c.embedding for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=[build_metadata(c) for c in chunks]
            )
            chunks_total += len(chunks)
        except FetchError as e:
            log.error("fetch_failed", url=row.document_url, error=str(e))

    return {"chunks_ingested": chunks_total, "benchmark_pairs": pairs_total,
            "unique_docs": len(seen_docs)}

# web_fetcher.py
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def get(url: str) -> str:
    time.sleep(1.0 / config.CRAWL_RATE_PER_SEC)   # global rate limit
    resp = httpx.get(url, timeout=15, headers={"User-Agent": config.USER_AGENT},
                     follow_redirects=True)
    resp.raise_for_status()
    return resp.text

# html_cleaner.py
def clean(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["nav","footer","script","style","aside","header"]):
        tag.decompose()
    # NLM/NIH pages: main content lives in <div id="section-body"> or <main>
    main = soup.find("main") or soup.find("div", {"id": "section-body"}) or soup.body
    text = main.get_text(separator="\n", strip=True)
    # Collapse blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# retriever.py
def retrieve(query: str, top_k: int = 5) -> List[dict]:
    q_emb = embedder.encode([query])[0]
    results = chroma_client.raw_chunks.query(
        query_embeddings=[q_emb], n_results=top_k
    )
    return results
```

### API Endpoints
```
POST /api/v1/ingest/medquad
  Body: { limit: Optional[int] }   # null = full dataset
  Response: { chunks_ingested: int, benchmark_pairs: int, unique_docs: int }
  Note: Runs as background task; returns 202 immediately with job_id.

GET /api/v1/ingest/status/{job_id}
  Response: { status: "running"|"done"|"failed", progress: int, total: int }

POST /api/v1/query
  Body: { query: str, top_k: int = 5 }
  Response: { answer: str, sources: List[str], latency_ms: float }

GET /api/v1/health
  Response: { status: "ok", chroma_size: int, benchmark_pairs: int }
```

### Background Jobs
None in Phase 1.

### Error Handling
| Failure | Recovery |
|---------|----------|
| HuggingFace dataset timeout | Retry with `datasets` built-in retry; checkpoint `seen_docs` to resume |
| HTTP 404 on document_url | Skip URL, log `fetch_skip`, continue to next doc |
| HTTP 429 / rate-limit from NLM | Exponential backoff via tenacity; honour `Retry-After` header |
| HTML yields <100 chars after clean | Skip chunk, log `thin_content` warning |
| ChromaDB upsert timeout | Retry 3x with exponential backoff |
| Embedding model OOM | Reduce batch size to 16, retry |
| OpenAI rate limit | Retry with jitter (tenacity) |
| Empty chunk after split | Skip, log warning |
| Duplicate document_id | `upsert` (not insert) — idempotent by design |

### Metrics
- `ingest_docs_processed` (counter)
- `ingest_docs_skipped` (counter — fetch errors, thin content)
- `ingest_chunks_total` (counter)
- `fetch_latency_ms` (histogram — per URL crawl)
- `chunks_per_doc` (histogram)
- `query_latency_ms` (p50, p95, p99)
- `chroma_collection_size` (gauge)
- `benchmark_pairs_total` (gauge)
- `openai_tokens_used` (counter)

### Unit Tests
```python
# test_phase1_ingestion.py

def test_dataset_loader_yields_rows():
    rows = list(itertools.islice(dataset_loader.stream_medquad(), 5))
    assert len(rows) == 5
    assert all(r.document_url.startswith("http") for r in rows)

def test_web_fetcher_gets_html(respx_mock):
    respx_mock.get("https://ghr.nlm.nih.gov/condition/mabry-syndrome").mock(
        return_value=httpx.Response(200, text="<main>Mabry syndrome content</main>")
    )
    html = web_fetcher.get("https://ghr.nlm.nih.gov/condition/mabry-syndrome")
    assert "Mabry" in html

def test_html_cleaner_strips_nav():
    html = "<nav>skip</nav><main>Keep this content about the disease.</main>"
    text = html_cleaner.clean(html)
    assert "skip" not in text
    assert "Keep this content" in text

def test_html_cleaner_rejects_thin_content():
    text = html_cleaner.clean("<html><body>404 Not Found</body></html>")
    assert len(text) < 100  # caller should skip

def test_chunker_respects_token_limit():
    chunks = chunker.split(LONG_TEXT, document_id="0000613", source_url="http://x.com",
                            question_focus="Mabry", umls_semantic_group="Disorders")
    assert all(c.token_count <= 550 for c in chunks)
    assert all(c.chunk_id.startswith("0000613_") for c in chunks)

def test_ingest_writes_benchmark_pairs():
    run_medquad_ingestion(limit=10)
    pairs = sqlite_client.fetch_all_benchmark_pairs()
    assert len(pairs) == 10
    assert all(p.question and p.answer for p in pairs)

def test_ingest_deduplicates_docs():
    # MedQuAD has 5 rows for doc 0000613 (same URL)
    result = run_medquad_ingestion(limit=5)
    assert result["unique_docs"] == 1  # only 1 URL crawled

def test_query_returns_answer():
    response = client.post("/api/v1/query", json={"query": "What is Mabry syndrome?"})
    assert response.status_code == 200
    assert len(response.json()["answer"]) > 10
```

### Future Scalability
- Replace synchronous crawl loop with async `asyncio` + `httpx.AsyncClient` for 10x throughput
- Add crawl checkpoint: persist `seen_docs` set to SQLite so ingestion is resumable after crash
- Replace SentenceTransformer CPU with GPU batch inference (torch.cuda)
- Swap ChromaDB for Weaviate/Qdrant when collection > 1M vectors
- Add Redis cache for repeated queries (exact match)

---

**NEXT FILES TO IMPLEMENT (Phase 1):**
1. `config.py` — env vars, hyperparams
2. `db/chroma_client.py` — singleton
3. `db/sqlite_client.py` — also handles benchmark_pairs table
4. `ingestion/embedder.py`
5. `ingestion/chunker.py`
6. `ingestion/web_fetcher.py`
7. `ingestion/html_cleaner.py`
8. `ingestion/dataset_loader.py`
9. `ingestion/benchmark_loader.py`
10. `ingestion/ingestor.py`
11. `retriever.py`
12. `generation/generator.py`
13. `api/routes/ingest.py`
14. `api/routes/query.py`
15. `main.py`

**EXACT CODE DEPENDENCIES:**
```
config.py → (none)
chroma_client.py → config.py
sqlite_client.py → config.py
embedder.py → config.py
chunker.py → (none)
web_fetcher.py → config.py, tenacity, httpx
html_cleaner.py → beautifulsoup4, lxml
dataset_loader.py → datasets (HuggingFace)
benchmark_loader.py → sqlite_client.py
ingestor.py → dataset_loader, web_fetcher, html_cleaner, chunker, embedder, chroma_client, benchmark_loader
retriever.py → embedder, chroma_client
generator.py → config.py (openai key)
routes/ingest.py → ingestor
routes/query.py → retriever, generator
main.py → all routes
```

---

## PHASE 2 — Query Logging and Hashing

### Goal
Intercept every query. Embed it. If cosine sim > 0.92 against an existing cluster in SQLite, increment that cluster's `hit_count`, update the cluster centroid using a running weighted average (SU6), and record which **raw chunks only** were retrieved (SU9 — ground-truth anchoring). This is the data collection layer that makes self-evolution possible.

### Components
```
app/
  logging_/
    query_logger.py     # Core log/upsert logic + centroid update (SU6)
    normalizer.py       # Strip → lowercase → embed
  db/
    sqlite_client.py    # SQLite connection + migrations
    models.py           # ORM models
  middleware/
    query_interceptor.py  # FastAPI middleware + ground-truth filter (SU9)
```

### Inputs
- Raw query string
- Retrieved chunk IDs (from Phase 1 retriever) — **raw_chunk IDs only** (SU9)

### Outputs
- SQLite row: new query cluster OR incremented hit_count + updated centroid
- `retrieved_chunk_ids` list (raw_chunks only) appended to cluster log

### Database Schema (SQLite)
```sql
CREATE TABLE query_clusters (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_query TEXT NOT NULL,
    query_embedding BLOB NOT NULL,      -- serialized numpy float32 array (running centroid)
    hit_count     INTEGER DEFAULT 1,
    chunk_ids     TEXT NOT NULL,        -- JSON array: ["doc_chunk_1","doc_chunk_48"] ← raw only
    first_seen    DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_hit      DATETIME DEFAULT CURRENT_TIMESTAMP,
    synthesized   BOOLEAN DEFAULT 0,    -- 1 = super-node already created
    super_node_id TEXT                  -- FK to ChromaDB super-node ID
);

CREATE INDEX idx_hit_count ON query_clusters(hit_count DESC);
CREATE INDEX idx_synthesized ON query_clusters(synthesized, hit_count);

-- Individual query log (raw, for audit)
CREATE TABLE query_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_query     TEXT NOT NULL,
    cluster_id    INTEGER REFERENCES query_clusters(id),
    retrieved_chunks TEXT,              -- JSON array (raw_chunk IDs only)
    timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Data Structures
```python
@dataclass
class QueryCluster:
    id: Optional[int]
    canonical_query: str
    query_embedding: np.ndarray    # shape (384,) — running centroid, not first-query locked
    hit_count: int
    chunk_ids: List[str]           # union of all retrieved raw_chunk IDs (no super_node IDs)
    first_seen: datetime
    last_hit: datetime
    synthesized: bool
    super_node_id: Optional[str]

@dataclass
class QueryLogEntry:
    raw_query: str
    cluster_id: int
    retrieved_chunks: List[str]    # raw_chunk IDs only
    timestamp: datetime
```

### Algorithm
```
1. Query arrives at /query endpoint
2. Normalize: strip, lowercase, embed → q_emb (384-dim)
3. Fetch all query_clusters from SQLite (embedding BLOB)
4. Compute cosine similarity: q_emb vs each cluster.query_embedding (running centroid)
5. If max_sim >= 0.92:
   a. cluster = argmax cluster
   b. cluster.hit_count += 1
   c. cluster.chunk_ids = union(cluster.chunk_ids, new_raw_retrieved_ids)  ← raw only (SU9)
   d. cluster.last_hit = now()
   e. UPDATE cluster centroid via running weighted average (SU6):
      new_centroid = ((old_centroid * n) + q_emb) / (n + 1)
   f. UPDATE query_clusters SET ...
6. Else (new cluster):
   a. INSERT new row: canonical_query=q_normalized, embedding=q_emb
      (centroid starts as first-query embedding; updated on next match)
7. INSERT query_log row
```

### Pseudocode
```python
# query_logger.py

def log_query(raw_query: str, retrieved_chunk_ids: List[str]) -> int:
    """
    retrieved_chunk_ids must contain only raw_chunk IDs (prefixed "doc_").
    Super-node IDs are filtered upstream in query_interceptor.py (SU9).
    """
    q_emb = normalizer.embed(raw_query)
    clusters = sqlite_client.fetch_all_clusters()  # List[QueryCluster]

    if clusters:
        sims = cosine_similarity([q_emb], [c.query_embedding for c in clusters])[0]
        best_idx, best_sim = np.argmax(sims), np.max(sims)

        if best_sim >= COSINE_THRESHOLD:  # 0.92
            cluster = clusters[best_idx]
            merged_ids = list(set(cluster.chunk_ids + retrieved_chunk_ids))

            # SU6: Update centroid via running weighted average
            update_cluster_centroid(cluster.id, q_emb)

            sqlite_client.update_cluster(
                cluster_id=cluster.id,
                chunk_ids=merged_ids,
                hit_count=cluster.hit_count + 1
            )
            return cluster.id

    # New cluster
    cluster_id = sqlite_client.insert_cluster(
        canonical_query=raw_query,
        query_embedding=q_emb,
        chunk_ids=retrieved_chunk_ids
    )
    return cluster_id


def update_cluster_centroid(cluster_id: int, new_query_emb: np.ndarray):
    """
    SU6 — Running Weighted Average Centroid Update.
    Prevents centroid stagnation when initial query was poorly phrased.
    Called BEFORE incrementing hit_count so n = count before this query.
    """
    cluster = sqlite_client.get_cluster(cluster_id)
    current_emb = np.frombuffer(cluster.query_embedding, dtype=np.float32)
    n = cluster.hit_count  # current size before incrementing

    # Running weighted average: New Centroid = ((Current * n) + New) / (n + 1)
    updated_emb = ((current_emb * n) + new_query_emb) / (n + 1)

    sqlite_client.update_cluster_embedding(
        cluster_id=cluster_id,
        query_embedding=updated_emb.astype(np.float32).tobytes()
    )
```

### Middleware — Ground-Truth Anchoring (SU9)
```python
# middleware/query_interceptor.py

@app.middleware("http")
async def log_query_middleware(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/api/v1/query":
        body = await request.json()

        # SU9 — Ground-Truth Filter:
        # Only log IDs belonging to original raw documents.
        # Raw chunks are prefixed "doc_"; super_node IDs are prefixed "sn_".
        # This prevents super-node summaries from entering the synthesis lineage
        # and causing generative artifacting (Knowledge JPEG degradation).
        raw_grounding_ids = [
            chunk_id for chunk_id in response_ctx["chunk_ids"]
            if chunk_id.startswith("doc_")
        ]

        query_logger.log_query(body["query"], raw_grounding_ids)
    return response
```

### Background Jobs
None in Phase 2. (Triggered in Phase 3.)

### Error Handling
| Failure | Recovery |
|---------|----------|
| SQLite locked (concurrent writes) | WAL mode + retry queue |
| cosine_similarity OOM (10k+ clusters) | Batch compute in chunks of 1000; migrate to FAISS index |
| Embedding mismatch dim | Assert 384 on insert, raise ValueError |
| Cluster chunk_ids corrupt (bad JSON) | Catch JSONDecodeError, reset to new_ids, log alert |
| Centroid BLOB corrupt | Fallback to raw q_emb for this update; log alert |

### ⚠️ Hidden Engineering Problem
**Scalability of in-memory cosine search:** Loading all cluster embeddings into RAM to compute cosine similarity works at <10k clusters. Beyond that, you need a FAISS or Annoy index over cluster embeddings. Build the `sqlite_client.fetch_all_clusters()` interface now so you can swap the backend later without touching `query_logger.py`.

### Metrics
- `clusters_total` (Gauge)
- `cluster_hit_rate` = queries that matched existing cluster / total queries
- `avg_hit_count` across clusters
- `new_clusters_per_hour`
- `sqlite_write_latency_ms`
- `centroid_update_count` (counter — SU6)

### Unit Tests
```python
def test_new_query_creates_cluster():
    log_query("what is the penalty for late work", ["doc_c1","doc_c2"])
    clusters = sqlite_client.fetch_all_clusters()
    assert len(clusters) == 1

def test_similar_query_increments_hit():
    log_query("what is the penalty for late work", ["doc_c1","doc_c2"])
    log_query("what happens if I submit tomorrow", ["doc_c1","doc_c3"])
    clusters = sqlite_client.fetch_all_clusters()
    assert len(clusters) == 1
    assert clusters[0].hit_count == 2
    assert "doc_c3" in clusters[0].chunk_ids

def test_dissimilar_query_creates_new_cluster():
    log_query("penalty for late work", ["doc_c1"])
    log_query("what is the refund policy", ["doc_c5"])
    assert len(sqlite_client.fetch_all_clusters()) == 2

# SU6 tests
def test_centroid_updates_on_match():
    log_query("what is the penalty for late work", ["doc_c1"])
    cluster_before = sqlite_client.fetch_all_clusters()[0]
    emb_before = np.frombuffer(cluster_before.query_embedding, dtype=np.float32)
    log_query("what happens if I submit tomorrow", ["doc_c1"])
    cluster_after = sqlite_client.fetch_all_clusters()[0]
    emb_after = np.frombuffer(cluster_after.query_embedding, dtype=np.float32)
    assert not np.allclose(emb_before, emb_after), "Centroid must shift after new matching query"

# SU9 tests
def test_interceptor_filters_super_node_ids(mocker):
    mock_log = mocker.patch("query_logger.log_query")
    response_ctx["chunk_ids"] = ["doc_c1", "sn_45_1700000", "doc_c2"]
    # Trigger middleware
    client.post("/api/v1/query", json={"query": "test"})
    logged_ids = mock_log.call_args[0][1]
    assert "sn_45_1700000" not in logged_ids
    assert "doc_c1" in logged_ids and "doc_c2" in logged_ids
```

### Future Scalability
- Swap full-table scan for FAISS IVF index on cluster embeddings (>10k clusters)
- Migrate SQLite → PostgreSQL with pgvector extension
- Add Bloom filter for exact-match deduplication before embedding

### Changes Introduced

**SU6 — Running Centroid Update**
- *Old behavior:* The cluster embedding was permanently locked to the first query that created the cluster. Over time, as hundreds of semantically similar but better-phrased queries arrived, the centroid drifted away from the geometric mean of the cluster, causing valid matches to fall below the 0.92 cosine threshold (Centroid Stagnation).
- *New behavior:* Every time a query matches an existing cluster, `update_cluster_centroid()` applies a running weighted average formula: `new = ((old * n) + q_emb) / (n + 1)`. The centroid continuously tracks the true semantic center of all matched queries.
- *Reason:* Eliminates centroid stagnation. Benchmark impact: improves cluster_hit_rate over long-running simulations and prevents valid patterns from being missed.

**SU9 — Ground-Truth Anchoring in Middleware**
- *Old behavior:* `query_interceptor.py` logged all chunk IDs returned by the retriever, including super-node IDs (`sn_*`). This meant super-node summaries fed back into the synthesis lineage, causing each new synthesis to summarize a prior summary (Knowledge JPEG degradation).
- *New behavior:* The middleware filters `response_ctx["chunk_ids"]` to retain only IDs prefixed `doc_` (raw chunks) before passing them to `query_logger.log_query()`. Super-node IDs are silently discarded from the logging path.
- *Reason:* The LLM must always synthesize from ground-truth raw chunks, never from previously generated summaries. This prevents nuance erosion and hallucination multiplication over repeated synthesis cycles.

---

**NEXT FILES TO IMPLEMENT (Phase 2):**
1. `db/sqlite_client.py` — add `update_cluster_embedding()` method
2. `logging_/normalizer.py`
3. `logging_/query_logger.py` — includes `update_cluster_centroid()`
4. `middleware/query_interceptor.py` — includes ground-truth filter

**EXACT CODE DEPENDENCIES:**
```
sqlite_client.py → config.py
normalizer.py → embedder.py
query_logger.py → sqlite_client, normalizer
query_interceptor.py → query_logger
routes/query.py → query_interceptor (inject)
```

---

## PHASE 3 — Synthesis Trigger (Pattern Finder + APScheduler)

### Goal
Background job polls SQLite every N minutes. When a cluster's `hit_count >= threshold` AND `synthesized == 0`, extract the canonical query + union chunk IDs (raw chunks only — guaranteed by Phase 2 SU9) and push to Phase 4 synthesis pipeline.

### Components
```
app/
  scheduler/
    scheduler.py        # APScheduler setup
    jobs/
      pattern_finder.py # Reads SQLite, fires synthesis
  patterns/
    finder.py           # Co-occurrence mining logic
```

### Inputs
- SQLite `query_clusters` table
- Config: `HIT_COUNT_THRESHOLD=10`, `POLL_INTERVAL_MINUTES=5`

### Outputs
- `SynthesisJob` dataclass pushed to Phase 4 pipeline
- SQLite `synthesized=1` on completion

### Data Structures
```python
@dataclass
class SynthesisJob:
    cluster_id: int
    canonical_query: str
    chunk_ids: List[str]        # Union of raw_chunk IDs only (guaranteed by SU9)
    hit_count: int
    triggered_at: datetime

@dataclass
class PatternResult:
    cluster_id: int
    canonical_query: str
    chunk_ids: List[str]
    hit_count: int
```

### Algorithm
```
1. APScheduler fires every N minutes → pattern_finder.scan()
2. Query:
   SELECT * FROM query_clusters
   WHERE synthesizing = 0
     AND hit_count >= MIN_HIT_COUNT
     AND (hit_count - last_synthesized_hit_count) >= RESYNTH_DELTA
   (SU12: replaces the old `synthesized = 0` boolean gate with a counter comparison
    so clusters re-trigger synthesis every RESYNTH_DELTA new hits, not just once.)
3. For each qualifying cluster:
   a. chunk_ids = JSON.parse(cluster.chunk_ids)  # already a union of raw_chunk IDs
   b. Build SynthesisJob (carry existing sn_id so synthesizer can upsert, not insert)
   c. SET synthesizing=1 atomically BEFORE calling synthesizer (prevents double-run)
   d. Call synthesizer.run(job)  [Phase 4]
   e. On success: UPDATE last_synthesized_hit_count=hit_count, synthesizing=0
4. Log: clusters_triggered, synthesis_duration
```

### Pseudocode
```python
# jobs/pattern_finder.py
def scan_and_trigger():
    ready = sqlite_client.get_ready_clusters(
        min_hit_count=config.HIT_COUNT_THRESHOLD,
        resynth_delta=config.RESYNTH_DELTA         # SU12
    )
    for cluster in ready:
        job = SynthesisJob(
            cluster_id=cluster.id,
            canonical_query=cluster.canonical_query,
            chunk_ids=cluster.chunk_ids,
            hit_count=cluster.hit_count,
            triggered_at=datetime.utcnow(),
            existing_sn_id=cluster.super_node_id     # SU12: upsert same ID, not new
        )
        try:
            # SU12: Lock cluster as synthesizing BEFORE calling synthesizer
            sqlite_client.set_synthesizing(cluster.id, True)
            super_node_id = synthesizer.run(job)
            sqlite_client.mark_synthesized(cluster.id, super_node_id)   # sets last_synthesized_hit_count
            metrics.increment("synthesis_success")
        except SynthesisError as e:
            sqlite_client.set_synthesizing(cluster.id, False)
            log.error(f"Synthesis failed for cluster {cluster.id}: {e}")
            metrics.increment("synthesis_failure")

# scheduler/scheduler.py
def create_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        scan_and_trigger,
        trigger=IntervalTrigger(minutes=config.POLL_INTERVAL_MINUTES),
        id="pattern_finder",
        max_instances=1,          # CRITICAL: prevent overlapping runs
        coalesce=True
    )
    return scheduler
```

### API Endpoints
```
POST /api/v1/admin/trigger-synthesis
  # Manual trigger for testing
  Response: { clusters_triggered: int }

GET /api/v1/admin/scheduler-status
  Response: { next_run: ISO8601, last_run: ISO8601, jobs: List[str] }
```

### Background Jobs
| Job | Trigger | Interval | Max Instances |
|-----|---------|----------|---------------|
| `pattern_finder` | Interval | 5 min | 1 |

### Error Handling
| Failure | Recovery |
|---------|----------|
| Scheduler crashes | FastAPI lifespan restarts scheduler on app restart |
| Synthesis timeout | Set `job_defaults={'misfire_grace_time': 30}` |
| Overlapping runs | `max_instances=1` prevents double-synthesis |
| DB locked during scan | Read with `isolation_level=None` (autocommit read) |

### ⚠️ Hidden Engineering Problem
**Double synthesis:** Without `max_instances=1`, two scheduler ticks can both read the same cluster and launch duplicate synthesis jobs. Always set `max_instances=1` **and** mark cluster `synthesizing=1` (SU12: this column replaces the `synthesized` boolean as the concurrency lock) atomically BEFORE kicking off Phase 4, not after.

Add a `last_synthesized_hit_count` column to `query_clusters` (SU12) to enable periodic re-synthesis:
```sql
ALTER TABLE query_clusters ADD COLUMN last_synthesized_hit_count INTEGER DEFAULT 0;
-- Pattern finder triggers when:
--   hit_count >= MIN_HIT_COUNT
--   AND (hit_count - last_synthesized_hit_count) >= RESYNTH_DELTA
-- On synthesis success: SET last_synthesized_hit_count = hit_count
```

### Metrics
- `pattern_finder_run_duration_ms`
- `clusters_eligible` (per run)
- `synthesis_triggered` (counter)
- `synthesis_success_rate`
- `re_synthesis_triggered` (counter — SU12: counts re-synthesis cycles, not just initial)
- `avg_revision_count` (gauge — SU12: mean number of times each cluster has been re-synthesized)

### Unit Tests
```python
def test_pattern_finder_triggers_above_threshold():
    insert_cluster(hit_count=10, last_synthesized_hit_count=0)
    jobs = pattern_finder.get_ready_clusters(min_hit_count=10, resynth_delta=10)
    assert len(jobs) == 1

def test_pattern_finder_skips_below_threshold():
    insert_cluster(hit_count=5, last_synthesized_hit_count=0)
    jobs = pattern_finder.get_ready_clusters(min_hit_count=10, resynth_delta=10)
    assert len(jobs) == 0

# SU12: Periodic re-synthesis tests
def test_pattern_finder_retrigggers_after_delta():
    # Cluster was synthesized at hit=10; now at hit=20; delta=10 → should re-trigger
    insert_cluster(hit_count=20, last_synthesized_hit_count=10)
    jobs = pattern_finder.get_ready_clusters(min_hit_count=10, resynth_delta=10)
    assert len(jobs) == 1

def test_pattern_finder_skips_below_delta():
    # hit=15, last_synthesized=10 → delta=5, below RESYNTH_DELTA=10 → skip
    insert_cluster(hit_count=15, last_synthesized_hit_count=10)
    jobs = pattern_finder.get_ready_clusters(min_hit_count=10, resynth_delta=10)
    assert len(jobs) == 0

def test_no_double_synthesis(mocker):
    mocker.patch("synthesizer.run", side_effect=lambda j: "sn_1")
    insert_cluster(hit_count=10, last_synthesized_hit_count=0)
    scan_and_trigger()
    scan_and_trigger()
    assert synthesizer.run.call_count == 1
```

---

**NEXT FILES TO IMPLEMENT (Phase 3):**
1. `scheduler/scheduler.py`
2. `patterns/finder.py`
3. `scheduler/jobs/pattern_finder.py` — SU12: uses `last_synthesized_hit_count` delta query

**EXACT CODE DEPENDENCIES:**
```
scheduler.py → config.py, apscheduler
finder.py → sqlite_client.py
pattern_finder.py (job) → finder.py, synthesizer.py (Phase 4)
main.py → scheduler.py (start on lifespan)
```

**SU12 — Re-Synthesis Threshold Counter**
- *Old behavior:* Pattern finder used `WHERE synthesized = 0` — a permanent boolean gate. After first synthesis, the cluster was frozen forever regardless of how many more hits arrived.
- *New behavior:* `WHERE (hit_count - last_synthesized_hit_count) >= RESYNTH_DELTA` triggers re-synthesis every `RESYNTH_DELTA` new hits. Each re-synthesis upserts the same `sn_id` (not a new one) and updates `last_synthesized_hit_count = hit_count`.
- *Reason:* A cluster queried 40 times should have a fresher, evidence-richer super-node than one queried 10 times. The boolean flag silently froze the earliest, most evidence-poor snapshot as permanent. The counter keeps knowledge current without creating duplicate nodes.

---

## PHASE 4 — Summarizer Agent & Validation Engine

### Goal
Fetch raw chunks, apply semantic deduplication to prevent token window collapse (SU1), run LLM synthesis with strict prompt, validate factual fidelity using an LLM-as-judge entailment model (SU2), and upsert Super-Node to ChromaDB. The core intelligence of the system.

### Components
```
app/
  synthesis/
    chunk_fetcher.py     # Fetch + semantically deduplicate chunks (SU1)
    synthesizer.py       # LLM synthesis orchestrator
    prompts.py           # System + user prompt templates
  validation/
    validator.py         # LLM-as-judge entailment engine (SU2)
    entity_extractor.py  # Retained for hallucination detection (additions check)
  db/
    super_node_store.py  # ChromaDB upsert for super-nodes
```

### Inputs
- `SynthesisJob`: canonical_query, chunk_ids (raw_chunk IDs only — guaranteed by SU9), hit_count

### Outputs
- ChromaDB upsert: Super-Node document with full metadata
- Returns: `super_node_id: str` OR raises `SynthesisError`

### Database Schema (ChromaDB Collection: `super_nodes`)
```
Collection: super_nodes
  id:          "sn_{cluster_id}_{timestamp}"  ← set at first synthesis; REUSED on re-synthesis (SU12)
  embedding:   List[float] (384-dim)
  document:    str  → compressed synthesis text
  metadata:
    type:           "super_node"
    source_query:   str   → canonical query
    source_chunks:  str   → JSON list of raw_chunk IDs (never sn_* IDs)
    hit_count:      int
    created_at:     ISO8601
    last_accessed:  ISO8601
    access_count:   int = 0   ← NOTE: canonical count lives in SQLite (SU10)
    decay_score:    float = 1.0
    cluster_id:     int
    fact_coverage:  float  → LLM-judge validation score
    in_hierarchy:   bool = False   ← merger lock flag (SU7)
    is_stale:       bool = False   ← soft-staleness flag (SU8)
    parent_meta_id:   str   = ""     ← set by merger; used by SU13 upward propagation
    lineage_depth:    int   = 0      ← merger increments; capped at MAX_LINEAGE_DEPTH (SU11)
    revision:         int   = 1      ← incremented on each re-synthesis (SU12)
    fidelity_flagged: bool  = False  ← SU14: summary drifted toward query more than source supports
    drift_margin:     float = 0.0    ← SU14: sim_summary_to_query minus max(sim_raw_to_query)
```

### Data Structures
```python
@dataclass
class SuperNode:
    id: str
    text: str
    embedding: List[float]
    source_query: str
    source_chunks: List[str]    # raw_chunk IDs only
    hit_count: int
    created_at: datetime
    last_accessed: datetime
    access_count: int           # mirrored from SQLite for ChromaDB queries
    decay_score: float
    cluster_id: int
    fact_coverage: float
    in_hierarchy: bool = False  # SU7
    is_stale: bool = False      # SU8
    parent_meta_id: str = ""    # SU13: set by merger; checked on re-synthesis for upward propagation
    lineage_depth: int = 0      # SU11: merger increments; enforces MAX_LINEAGE_DEPTH cap
    revision: int = 1           # SU12: incremented on each re-synthesis

@dataclass
class SynthesisJob:
    cluster_id: int
    canonical_query: str
    chunk_ids: List[str]        # Union of raw_chunk IDs only (guaranteed by SU9)
    hit_count: int
    triggered_at: datetime
    existing_sn_id: Optional[str] = None  # SU12: if set, upsert this ID; else create new

@dataclass
class ValidationResult:
    passed: bool
    coverage_score: float          # facts_in_summary / facts_in_source (LLM-scored)
    missing_facts: List[str]       # facts dropped during synthesis
    source_fact_count: int
    summary_fact_count: int

class SynthesisError(Exception):
    pass

class ValidationError(SynthesisError):
    pass
```

### Algorithm
```
1. Fetch raw texts + embeddings for all chunk_ids from ChromaDB
2. SEMANTIC DEDUPLICATION (SU1):
   a. If len(chunks) > 5:
      - Cluster with AgglomerativeClustering (cosine, distance_threshold=0.15)
      - Select one representative chunk per cluster
   b. Enforce token budget: truncate to max_tokens * 4 chars
3. Build synthesis prompt (canonical_query + deduplicated source_material)
4. Call LLM (Qwen-2.5-32B-Instruct, temperature=0.1, max_tokens=1000)
5. Extract candidate summary from response
6. LLM-AS-JUDGE VALIDATION (SU2):
   a. Send (source_material, summary) to Qwen-2.5-32B-Instruct in JSON mode
   b. LLM extracts facts, checks coverage mathematically
   c. Returns: { passed: bool, coverage_score: float, missing_facts: list }
   d. IF coverage < 0.90:
      - Retry with re-synthesis prompt (max 3 retries)
      - If still failing: ABORT, mark cluster as synthesis_failed
7. (Optional hallucination check) entity_extractor: flag entities in summary NOT in source
8. Embed summary → 384-dim vector
9. SYNTHESIS FIDELITY BOUND CHECK (SU14):
   a. Compute source_centroid = mean(source_chunk_embeddings)
   b. sim_summary_to_source = cosine_sim(summary_emb, source_centroid)
   c. sim_summary_to_query  = cosine_sim(summary_emb, query_emb)
   d. sim_raw_to_query_max  = max(cosine_sim(chunk_emb, query_emb) for each chunk)
   e. drift_margin = sim_summary_to_query - sim_raw_to_query_max
   f. IF drift_margin > MAX_DRIFT_MARGIN (0.08):
      - Flag: fidelity_flagged=True (summary sounds more relevant than source justifies)
      - log warning + increment synthesis_fidelity_flagged metric
      - DO NOT abort — allow serving, but mark for human spot-check
   g. IF sim_summary_to_source < MIN_SOURCE_ANCHOR (0.55):
      - Hard reject: force re-synthesis with stricter prompt or smaller chunk batch
      - If still failing after MAX_SYNTHESIS_RETRIES: raise SynthesisError
10. Upsert to ChromaDB super_nodes collection (includes fidelity_flagged, drift_margin)
11. SU13 upward staleness propagation (if revision > 1 and parent_meta_id)
12. Return super_node_id
```

### Updated Component: `synthesis/chunk_fetcher.py` (SU1)
```python
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from typing import List

def fetch_and_deduplicate(chunk_ids: List[str], max_tokens: int = 4000) -> str:
    """
    SU1 — Semantic Deduplication.
    Fetches raw chunks from ChromaDB, clusters them to identify redundant
    semantic content, and returns one representative chunk per cluster.
    Prevents token window collapse when many queries share the same intent
    and the chunk_ids union grows to 40+ redundant entries.
    """
    # 1. Fetch raw chunks + embeddings from ChromaDB
    raw_results = chroma_client.raw_chunks.get(
        ids=chunk_ids,
        include=["documents", "embeddings"]
    )

    docs = raw_results["documents"]
    embs = np.array(raw_results["embeddings"])

    # If small enough, skip clustering — no dedup needed
    if len(docs) <= 5:
        final_text = "\n\n---\n\n".join(docs)
        max_chars = max_tokens * 4
        return final_text[:max_chars]

    # 2. Cluster chunks to find redundancies
    # distance_threshold=0.15 in cosine space ≈ similarity > 0.85
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.15,
        metric="cosine",
        linkage="average"
    )
    labels = clustering.fit_predict(embs)

    # 3. Pick one representative document per cluster (first encountered)
    unique_docs = []
    seen_labels = set()
    for i, label in enumerate(labels):
        if label not in seen_labels:
            unique_docs.append(docs[i])
            seen_labels.add(label)

    # 4. Enforce token limit (~4 chars per token)
    final_text = "\n\n---\n\n".join(unique_docs)
    max_chars = max_tokens * 4
    return final_text[:max_chars]
```

### Updated Component: `validation/validator.py` (SU2)
```python
import json
from pydantic import BaseModel

class ValidationOutput(BaseModel):
    passed: bool
    coverage_score: float
    missing_facts: list[str]

LLM_JUDGE_PROMPT = """
You are a strict grading algorithm.
Compare the Source Material against the Generated Summary.
1. Extract the core facts from the Source Material.
2. Check if EACH fact is present in the Summary.
3. Calculate coverage = (Facts Present / Total Source Facts).
4. Return JSON strictly matching this schema:
{"passed": true/false (true if >= 0.90), "coverage_score": float, "missing_facts": [list of strings]}

Source Material:
{source_text}

Summary:
{summary_text}
"""

def validate_entailment(source_text: str, summary: str) -> ValidationOutput:
    """
    SU2 — LLM-as-Judge Entailment Validation.
    Replaces regex + spaCy NER with semantic entailment scoring via Qwen-2.5-32B-Instruct
    in strict JSON mode. Captures nuanced factual coverage that surface-form
    string matching cannot detect (e.g. paraphrased facts, implied relationships).
    """
    response = client.chat.completions.create(
        model="Qwen-2.5-32B-Instruct",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": LLM_JUDGE_PROMPT.format(
                source_text=source_text, summary_text=summary
            )}
        ],
        temperature=0.0   # Deterministic scoring
    )

    result_dict = json.loads(response.choices[0].message.content)
    return ValidationOutput(**result_dict)


# Backward-compatible shim: synthesizer.py calls validate(), not validate_entailment()
def validate(source_text: str, summary: str) -> ValidationResult:
    output = validate_entailment(source_text, summary)
    return ValidationResult(
        passed=output.passed,
        coverage_score=output.coverage_score,
        missing_facts=output.missing_facts,
        source_fact_count=0,   # Not available from LLM response; set to 0
        summary_fact_count=0
    )
```

### Note on `entity_extractor.py` (Retained for Hallucination Check)
`entity_extractor.py` is **not removed**. It is retained for the optional hallucination check in Step 7 of the algorithm (entities present in summary but absent from source = injected facts). It is no longer the primary coverage mechanism.

### Synthesis Prompt (Unchanged)
```python
SYSTEM_PROMPT = """You are a strict, highly precise knowledge architect.
Your goal is to create a compressed, self-contained knowledge entry
based ONLY on the provided source passages."""

USER_PROMPT_TEMPLATE = """
Task: Users are frequently asking variations of this core question:
"{canonical_query}"

Source Material:
{source_chunks_text}

Instructions:
1. Synthesize a single, dense, self-contained summary that definitively answers the core question.
2. Include ALL key facts, numbers, dates, and critical nuances found in the source material.
3. Do not include conversational filler, introductions, or external knowledge.
4. Output ONLY the final compressed summary.
"""

RESYNTH_PROMPT_TEMPLATE = """
Your previous synthesis was rejected. Missing facts: {missing_facts}

Re-synthesize ensuring ALL of the above facts are present.
Source Material:
{source_chunks_text}
"""
```

### Pseudocode
```python
# synthesizer.py
def run(job: SynthesisJob, max_retries: int = 3) -> str:
    # Step 1: Fetch + deduplicate chunks (SU1)
    source_text = chunk_fetcher.fetch_and_deduplicate(job.chunk_ids, max_tokens=4000)
    if not source_text:
        raise SynthesisError(f"No chunks found for IDs: {job.chunk_ids}")

    # Step 2: Synthesize with retry
    summary, attempt = None, 0
    missing_facts = []
    result = None

    while attempt < max_retries:
        if attempt == 0:
            prompt = USER_PROMPT_TEMPLATE.format(
                canonical_query=job.canonical_query,
                source_chunks_text=source_text
            )
        else:
            prompt = RESYNTH_PROMPT_TEMPLATE.format(
                missing_facts=missing_facts,
                source_chunks_text=source_text
            )

        summary = llm_call(SYSTEM_PROMPT, prompt, temperature=0.1)

        # Step 3: Validate via LLM-as-judge (SU2)
        result = validator.validate(source_text, summary)
        if result.passed:
            break

        missing_facts = result.missing_facts
        attempt += 1
        log.warning(f"Validation failed (attempt {attempt}): coverage={result.coverage_score:.2f}")

    if not result or not result.passed:
        raise ValidationError(f"Max retries exceeded. Final coverage: {result.coverage_score:.2f}")

    # Step 4: Embed + Upsert
    embedding = embedder.encode([summary])[0]

    # SU12: Reuse existing sn_id if this is a re-synthesis; create new ID on first synthesis
    if job.existing_sn_id:
        sn_id = job.existing_sn_id
        existing_meta = chroma_client.super_nodes.get(
            ids=[sn_id], include=["metadatas"]
        )["metadatas"][0]
        revision = existing_meta.get("revision", 1) + 1
        parent_meta_id = existing_meta.get("parent_meta_id", "")
        lineage_depth = existing_meta.get("lineage_depth", 0)
        created_at = existing_meta.get("created_at", datetime.utcnow().isoformat())
    else:
        sn_id = f"sn_{job.cluster_id}_{int(datetime.utcnow().timestamp())}"
        revision = 1
        parent_meta_id = ""
        lineage_depth = 0
        created_at = datetime.utcnow().isoformat()

    chroma_client.super_nodes.upsert(
        ids=[sn_id],
        embeddings=[embedding.tolist()],
        documents=[summary],
        metadatas=[{
            "type": "super_node",
            "source_query": job.canonical_query,
            "source_chunks": json.dumps(job.chunk_ids),  # raw_chunk IDs only
            "hit_count": job.hit_count,
            "created_at": created_at,
            "last_accessed": datetime.utcnow().isoformat(),
            "access_count": 0,      # canonical count stored in SQLite (SU10)
            "decay_score": 1.0,
            "cluster_id": job.cluster_id,
            "fact_coverage": result.coverage_score,
            "in_hierarchy": False,  # SU7: merger lock flag
            "is_stale": False,      # SU8: soft-staleness flag
            "parent_meta_id": parent_meta_id,   # SU13: preserved from previous synthesis
            "lineage_depth": lineage_depth,      # SU11: preserved; merger manages increments
            "revision": revision,               # SU12: tracks how many times re-synthesized
            "fidelity_flagged": fidelity_flagged,             # SU14: overfit-to-query flag
            "drift_margin": fidelity["drift_margin"],         # SU14: research metric
            "sim_summary_to_source": fidelity["sim_summary_to_source"],  # SU14: anchor score
        }]
    )

    # SU14 — Synthesis Fidelity Bound Check
    # Runs AFTER validation passes, BEFORE upsert.
    # Catches semantic drift that the fact-coverage validator cannot detect:
    # (a) summary "sounds more relevant" than the source ever was (overfit to query)
    # (b) summary has drifted away from its own source entirely (compounding-summarization)
    q_emb = embedder.encode([job.canonical_query])[0]
    source_chunk_records = chroma_client.raw_chunks.get(
        ids=job.chunk_ids, include=["embeddings"]
    )
    source_chunk_embs = [np.array(e) for e in source_chunk_records["embeddings"]]

    fidelity = check_fidelity_bound(
        summary_emb=embedding,
        source_chunk_embs=source_chunk_embs,
        source_query_emb=q_emb
    )
    log.info("fidelity_bound_result", cluster_id=job.cluster_id, **fidelity)

    fidelity_flagged = False
    if fidelity["suspected_disconnect"]:
        # Summary has drifted away from its source — hard failure
        raise SynthesisError(
            f"Fidelity bound failed: sim_to_source={fidelity['sim_summary_to_source']:.3f} "
            f"< MIN_SOURCE_ANCHOR={config.MIN_SOURCE_ANCHOR}. Summary disconnected from ground truth."
        )
    if fidelity["suspected_overfit"]:
        # Summary sounds more confident than source justifies — soft flag, still serve
        log.warning("fidelity_overfit_flagged", cluster_id=job.cluster_id,
                    drift_margin=fidelity["drift_margin"])
        metrics.increment("synthesis_fidelity_flagged")
        fidelity_flagged = True

    # Inject fidelity fields into upsert metadata
    # (These are added to the metadatas dict before the chroma_client.super_nodes.upsert() call)

    # SU13 — Upward Staleness Propagation
    # If this is a re-synthesis (revision > 1) AND the node has a parent meta-node,
    # mark the parent is_stale so it gets rebuilt with our updated content.
    if revision > 1 and parent_meta_id:
        try:
            parent_meta = chroma_client.super_nodes.get(
                ids=[parent_meta_id], include=["metadatas"]
            )["metadatas"][0]
            chroma_client.super_nodes.update(
                ids=[parent_meta_id],
                metadatas=[{**parent_meta, "is_stale": True, "stale_reason": "child_revised"}]
            )
            # Reuse SU8's existing re-synthesis queue — same mechanism, new trigger condition
            parent_cluster_id = parent_meta.get("cluster_id")
            if parent_cluster_id:
                sqlite_client.flag_cluster_for_re_synthesis(parent_cluster_id)
            log.info("parent_marked_stale_by_child",
                     parent_id=parent_meta_id, child_id=sn_id, revision=revision)
        except Exception as e:
            log.warning("su13_upward_propagation_failed",
                        parent_id=parent_meta_id, error=str(e))

    return sn_id


def check_fidelity_bound(summary_emb: np.ndarray,
                         source_chunk_embs: list[np.ndarray],
                         source_query_emb: np.ndarray) -> dict:
    """
    SU14 — Synthesis Fidelity Bound Check.

    Flags synthesis drift that the fact-coverage validator cannot catch.

    A summary should sit BETWEEN its source chunks and the query in semantic space:
    it is allowed to be a cleaner expression of the source's content, but it should
    not be MORE similar to the query than the source material genuinely supports.

    If sn_similarity_to_query significantly exceeds max(raw_chunk_similarity_to_query),
    the synthesis likely over-fit to "sounding like a good answer" rather than
    "faithfully compressing what the source actually says."

    This is also a research metric: avg_drift_margin across the benchmark set tells you
    whether your synthesis prompt is systematically over-fitting to retrievability versus
    faithfully compressing. If drift_margin creeps upward as MAX_LINEAGE_DEPTH increases,
    that is empirical evidence supporting SU11's ground-truth re-anchoring design.
    """
    source_centroid = np.mean(source_chunk_embs, axis=0)

    sim_summary_to_source = cosine_similarity([summary_emb], [source_centroid])[0][0]
    sim_summary_to_query  = cosine_similarity([summary_emb], [source_query_emb])[0][0]
    sim_raw_to_query_max  = max(
        cosine_similarity([e], [source_query_emb])[0][0] for e in source_chunk_embs
    )

    drift_margin = sim_summary_to_query - sim_raw_to_query_max

    return {
        "sim_summary_to_source": float(sim_summary_to_source),
        "sim_summary_to_query":  float(sim_summary_to_query),
        "sim_raw_to_query_max":  float(sim_raw_to_query_max),
        "drift_margin":          float(drift_margin),
        # Summary is suspiciously MORE relevant than source ever was → possible overfit
        "suspected_overfit":     drift_margin > config.MAX_DRIFT_MARGIN,    # default 0.08
        # Summary has drifted AWAY from its own source → compounding-summarization symptom
        "suspected_disconnect":  sim_summary_to_source < config.MIN_SOURCE_ANCHOR,  # default 0.55
    }
```

### API Endpoints
```
POST /api/v1/admin/synthesize/{cluster_id}
  # Force synthesis for a specific cluster
  Response: { super_node_id: str, coverage: float }

GET /api/v1/admin/super-nodes
  Response: { nodes: List[SuperNodeSummary], total: int }
```

### Error Handling
| Failure | Recovery |
|---------|----------|
| LLM API error | Retry 3x with exponential backoff (tenacity) |
| Coverage < 0.90 after 3 retries | Set `synthesis_failed=1` in cluster, alert |
| ChromaDB upsert failure | Retry 3x; if fails, rollback `synthesizing` flag |
| LLM judge returns malformed JSON | Catch JSONDecodeError; fallback to entity_extractor; log warning |
| Empty source chunks after deduplication | Abort synthesis, log error |
| All chunks collapsed into 1 cluster | Return single chunk as source_text; skip dedup |

### ⚠️ Hidden Engineering Problems
1. **LLM judge cost:** Each validation call to Qwen-2.5-32B-Instruct costs compute/tokens. For high-volume synthesis, consider a local cross-encoder (e.g. `cross-encoder/nli-deberta-v3-small`) as a fast first-pass filter before escalating to Qwen-2.5-32B-Instruct.
2. **spaCy still required:** `entity_extractor.py` is retained for the hallucination additions check. Keep `RUN python -m spacy download en_core_web_sm` in the Dockerfile.
3. **AgglomerativeClustering requires sklearn ≥ 1.0** for `metric="cosine"`. Verified compatible with `scikit-learn==1.5.2` in requirements.txt.
4. **Token budget:** The deduplication step (SU1) replaces the old truncation-only approach, but the final `max_chars` guard is still in place as a safety net.

### Metrics
- `synthesis_duration_ms` (histogram)
- `synthesis_retries` (histogram)
- `validation_coverage_score` (histogram)
- `synthesis_abort_rate`
- `super_node_count` (gauge)
- `avg_fact_coverage`
- `dedup_chunks_removed` (counter — SU1: chunks eliminated per synthesis run)
- `llm_judge_calls` (counter — SU2)
- `synthesis_fidelity_flagged` (counter — SU14: nodes flagged for query overfit)
- `avg_drift_margin` (gauge — SU14: benchmark metric; rising value signals systematic overfit)
- `avg_sim_to_source` (gauge — SU14: falling value signals compounding-summarization drift)
- `fidelity_disconnect_aborts` (counter — SU14: hard re-synthesis forced by sim_to_source < MIN_SOURCE_ANCHOR)

### Unit Tests
```python
def test_synthesizer_creates_super_node():
    job = SynthesisJob(cluster_id=1, canonical_query="...", chunk_ids=["doc_c1","doc_c2"], hit_count=10)
    sn_id = synthesizer.run(job)
    assert sn_id.startswith("sn_1_")
    result = chroma_client.super_nodes.get(ids=[sn_id])
    assert result["metadatas"][0]["type"] == "super_node"
    assert result["metadatas"][0]["in_hierarchy"] == False  # SU7 init
    assert result["metadatas"][0]["is_stale"] == False      # SU8 init
    assert result["metadatas"][0]["revision"] == 1          # SU12 init
    assert result["metadatas"][0]["lineage_depth"] == 0     # SU11 init
    assert result["metadatas"][0]["parent_meta_id"] == ""   # SU13 init

# SU12 tests
def test_re_synthesis_reuses_same_sn_id():
    job1 = SynthesisJob(cluster_id=1, canonical_query="...", chunk_ids=["doc_c1"], hit_count=10)
    sn_id_v1 = synthesizer.run(job1)

    job2 = SynthesisJob(cluster_id=1, canonical_query="...",
                        chunk_ids=["doc_c1", "doc_c2"], hit_count=20,
                        existing_sn_id=sn_id_v1)   # pass existing ID
    sn_id_v2 = synthesizer.run(job2)

    assert sn_id_v1 == sn_id_v2, "Re-synthesis must reuse the same sn_id"
    result = chroma_client.super_nodes.get(ids=[sn_id_v2])
    assert result["metadatas"][0]["revision"] == 2, "Revision counter must increment"

# SU13 tests
def test_re_synthesis_marks_parent_stale():
    # Setup: super-node in a hierarchy, with parent meta-node
    job1 = SynthesisJob(cluster_id=1, canonical_query="...", chunk_ids=["doc_c1"], hit_count=10)
    sn_id = synthesizer.run(job1)

    # Simulate merger assigning a parent
    chroma_client.super_nodes.update(
        ids=[sn_id],
        metadatas=[{**chroma_client.super_nodes.get(ids=[sn_id])["metadatas"][0],
                    "parent_meta_id": "mn_999",
                    "in_hierarchy": True}]
    )

    # Re-synthesis should mark the parent stale
    job2 = SynthesisJob(cluster_id=1, canonical_query="...",
                        chunk_ids=["doc_c1", "doc_c2"], hit_count=20,
                        existing_sn_id=sn_id)
    synthesizer.run(job2)

    parent = chroma_client.super_nodes.get(ids=["mn_999"])
    assert parent["metadatas"][0]["is_stale"] == True
    assert parent["metadatas"][0]["stale_reason"] == "child_revised"

# SU1 + SU2 tests (unchanged from before)
def test_chunk_fetcher_deduplicates_redundant_chunks():
    ids = insert_near_identical_chunks(10)
    text = chunk_fetcher.fetch_and_deduplicate(ids)
    separator_count = text.count("\n\n---\n\n")
    assert separator_count < 5

def test_chunk_fetcher_skips_dedup_under_threshold():
    ids = insert_diverse_chunks(4)
    text = chunk_fetcher.fetch_and_deduplicate(ids)
    assert text.count("\n\n---\n\n") == 3

def test_validator_rejects_low_coverage(mocker):
    mocker.patch("client.chat.completions.create", return_value=mock_response(
        '{"passed": false, "coverage_score": 0.40, "missing_facts": ["$500", "2024-01-15"]}'
    ))
    result = validator.validate(
        source_text="The fee is $500. Deadline is 2024-01-15.",
        summary="The deadline is approaching."
    )
    assert not result.passed
    assert result.coverage_score < 0.90

def test_validator_accepts_high_coverage(mocker):
    mocker.patch("client.chat.completions.create", return_value=mock_response(
        '{"passed": true, "coverage_score": 0.97, "missing_facts": []}'
    ))
    result = validator.validate(
        source_text="The fee is $500. Deadline is 2024-01-15. Contact John Smith.",
        summary="Fee: $500, due 2024-01-15, contact John Smith."
    )
    assert result.passed

def test_synthesis_aborts_after_max_retries(mocker):
    mocker.patch("validator.validate", return_value=ValidationResult(passed=False, coverage_score=0.5, missing_facts=["x"], source_fact_count=2, summary_fact_count=1))
    with pytest.raises(ValidationError):
        synthesizer.run(job)

# SU14 tests
def test_fidelity_bound_normal_passes():
    """Faithful summary: sits close to source, modest drift margin."""
    summary_emb    = make_emb_between(source_centroid, query_emb, alpha=0.5)
    chunk_embs     = [source_centroid]  # single chunk for simplicity
    result = check_fidelity_bound(summary_emb, chunk_embs, query_emb)
    assert not result["suspected_overfit"]
    assert not result["suspected_disconnect"]

def test_fidelity_bound_overfit_flagged():
    """Summary closer to query than source ever was → suspected_overfit=True."""
    # Place summary 0.15 cosine closer to query than any raw chunk
    summary_emb = query_emb + np.random.rand(len(query_emb)) * 0.01  # near query
    chunk_embs  = [source_centroid]   # far from query
    result = check_fidelity_bound(summary_emb, chunk_embs, query_emb)
    assert result["suspected_overfit"], f"drift_margin={result['drift_margin']:.3f}"
    assert result["drift_margin"] > config.MAX_DRIFT_MARGIN

def test_fidelity_bound_disconnect_raises():
    """Summary far from source centroid → SynthesisError raised in synthesizer.run()."""
    mocker.patch("synthesizer.check_fidelity_bound", return_value={
        "sim_summary_to_source": 0.30,  # < MIN_SOURCE_ANCHOR
        "drift_margin": 0.01,
        "suspected_overfit": False,
        "suspected_disconnect": True,
    })
    with pytest.raises(SynthesisError, match="Fidelity bound failed"):
        synthesizer.run(job)

def test_fidelity_overfit_does_not_abort_synthesis():
    """Overfit flag is soft — node is upserted and marked fidelity_flagged=True."""
    mocker.patch("synthesizer.check_fidelity_bound", return_value={
        "sim_summary_to_source": 0.80,
        "drift_margin": 0.12,           # > MAX_DRIFT_MARGIN
        "suspected_overfit": True,
        "suspected_disconnect": False,
    })
    sn_id = synthesizer.run(job)        # should NOT raise
    node = chroma_client.super_nodes.get(ids=[sn_id])
    assert node["metadatas"][0]["fidelity_flagged"] == True
    assert node["metadatas"][0]["drift_margin"] > config.MAX_DRIFT_MARGIN

def test_fidelity_bound_drift_margin_increases_with_depth(mocker):
    """
    Research metric test: avg_drift_margin should be higher for nodes synthesized at
    MAX_LINEAGE_DEPTH=2 vs depth=0, empirically validating SU11's ground-truth
    re-anchoring design.
    """
    depth_0_margins = measure_drift_margins(lineage_depth=0, n=20)
    depth_2_margins = measure_drift_margins(lineage_depth=2, n=20)
    # If SU11 is OFF (summary-of-summary), depth_2 margins are significantly higher
    # With SU11 ON (raw re-synthesis), margins should be statistically similar
    assert np.mean(depth_2_margins) - np.mean(depth_0_margins) < 0.02, (
        "SU11 ground-truth re-anchoring is failing: drift margin grows with depth"
    )
```

### Changes Introduced

**SU1 — Semantic Deduplication in `chunk_fetcher.py`**
- *Old behavior:* `chunk_fetcher.fetch()` returned all chunk texts concatenated in order. In automated benchmarks with 50 similar queries, the `chunk_ids` union grew to 40+ entries, feeding massive redundancy into the LLM prompt and triggering the "lost in the middle" degradation effect.
- *New behavior:* `fetch_and_deduplicate()` embeds all fetched chunks and runs `AgglomerativeClustering` (cosine distance, threshold=0.15). Only one representative chunk per cluster is passed to synthesis. A hard character limit enforces the token budget as a safety net.
- *Reason:* Prevents token window collapse in long-running benchmarks and reduces LLM API costs by eliminating semantically redundant input.

**SU2 — LLM-as-Judge Validation in `validator.py`**
- *Old behavior:* `validate()` used regex number/date extraction plus spaCy NER to build sets of `source_facts` and `summary_facts`, computing coverage as set intersection ratio. This approach passed summaries that hallucinated relationships as long as the surface-form entities matched.
- *New behavior:* `validate_entailment()` sends source and summary to `Qwen-2.5-32B-Instruct` in strict JSON mode with a fact-extraction + coverage-calculation prompt. The LLM reasons over semantic meaning rather than string surface forms. `entity_extractor.py` is retained for hallucination addition detection (entities in summary not present in source).
- *Reason:* Regex/NER cannot detect paraphrased facts, implied relationships, or fabricated connections between correctly extracted entities. LLM-as-judge validates the "Fact Coverage" metric with the semantic precision required for a research paper claim of ≥90% factual fidelity.

**SU12 — Re-Synthesis Threshold Counter in `synthesizer.py`**
- *Old behavior:* First synthesis created `sn_{cluster_id}_{timestamp}` with a fresh ID each time. The pattern finder's boolean gate meant synthesis only ever ran once; subsequent runs would create orphan duplicate nodes.
- *New behavior:* `synthesizer.run()` accepts `existing_sn_id` from `SynthesisJob`. If present, it reads the current metadata (preserving `parent_meta_id`, `lineage_depth`, `created_at`), increments `revision`, and upserts to the same ChromaDB ID. First synthesis still generates a new ID.
- *Reason:* Prevents duplicate node proliferation while allowing content to evolve. The `revision` counter is a research metric tracking knowledge volatility (a frequently re-synthesized node is a signal that the topic is actively changing).

**SU13 — Upward Staleness Propagation in `synthesizer.py`**
- *Old behavior:* When a super-node was re-synthesized, its parent meta-node had no awareness of the change. The meta-node continued serving content that referenced the pre-revision version of its child's `source_chunks` scope.
- *New behavior:* At the end of `synthesizer.run()`, if `revision > 1` and `parent_meta_id` is set, the code marks the parent's `is_stale=True` with `stale_reason="child_revised"` and calls `sqlite_client.flag_cluster_for_re_synthesis(parent_cluster_id)`. This reuses SU8's existing soft-staleness queue — no new machinery.
- *Reason:* Without upward propagation, Fix 11 + Fix 12 together create a gap: children are rebuilt from ground-truth, but the meta-node overhead is stale relative to its own subtree. Fix 13 closes this gap by making the staleness signal travel upward, not just downward.

**SU14 — Synthesis Fidelity Bound Check in `synthesizer.py`**
- *Old behavior:* The only post-synthesis quality gate was the LLM-as-judge fact-coverage check (SU2). That validator answers "did all facts survive?" but cannot answer "did the summary anchor itself to the source, or drift toward sounding like a better answer?". A summary can pass 90% fact coverage while still being semantically re-oriented away from the source's actual evidence base and toward the query — a form of articulation drift invisible to any fact-extraction approach.
- *New behavior:* `check_fidelity_bound()` runs immediately after validation passes and before the ChromaDB upsert. It computes three cosine similarities: (1) summary vs source centroid, (2) summary vs query, (3) max(raw chunks vs query). The `drift_margin = sim_summary_to_query − max(sim_raw_to_query)` measures whether the summary has "jumped ahead" of the source in query-space. If `drift_margin > MAX_DRIFT_MARGIN` (0.08): soft flag — node is upserted with `fidelity_flagged=True` and `drift_margin` stored in metadata for human spot-check. If `sim_summary_to_source < MIN_SOURCE_ANCHOR` (0.55): hard reject — `SynthesisError` is raised and re-synthesis is forced, exactly like a coverage failure.
- *Research metric:* `avg_drift_margin` across the benchmark set is a paper-grade signal. A rising `avg_drift_margin` as `MAX_LINEAGE_DEPTH` increases is empirical evidence that SU11's ground-truth re-anchoring at every merge is necessary: without it, each merge generation drifts further from the source evidence base. With SU11 active, drift margins at depth=2 should be statistically indistinguishable from depth=0.
- *Reason:* Closes the detection gap between "facts present" (SU2) and "semantically anchored to source" (SU14). Together they form a two-axis quality guarantee: high coverage AND low drift.

---

**NEXT FILES TO IMPLEMENT (Phase 4):**
1. `validation/entity_extractor.py` (retained for hallucination check)
2. `validation/validator.py` — replace with LLM-as-judge (SU2)
3. `synthesis/prompts.py`
4. `synthesis/chunk_fetcher.py` — replace with deduplicating version (SU1)
5. `synthesis/synthesizer.py` — includes `check_fidelity_bound()` (SU14)
6. `db/super_node_store.py`

---

## PHASE 5 — Preferential Retrieval

### Goal
Modify the retrieval router to query BOTH `raw_chunks` and `super_nodes` collections. Apply an ε-greedy score boost to prevent filter bubble formation (SU4). Mask child super-nodes when a parent meta-node is retrieved to prevent context window flooding (SU5). Log access counts atomically to SQLite rather than ChromaDB to prevent concurrency counter loss (SU10). Fall back gracefully if no super_node exists.

### Components
```
app/
  retrieval/
    retriever.py          # Updated: ε-greedy dual-collection search (SU4) +
                          #          hierarchical masking (SU5) +
                          #          SQLite atomic access logging (SU10)
    score_booster.py      # Score boost logic (called conditionally by ε-greedy)
    router.py             # Route to super_node or raw_chunk
```

### Algorithm
```
1. Embed query → q_emb
2. Query super_nodes collection (top_k=3) + fetch IDs
3. Query raw_chunks collection (top_k=5)
4. HIERARCHICAL MASKING (SU5):
   a. First pass: identify any retrieved meta_nodes; collect their children IDs
   b. Second pass: exclude child IDs from candidate list
5. ε-GREEDY BOOST (SU4):
   a. Draw random float r in [0, 1)
   b. If r > epsilon (0.10) → Exploitation: apply 0.85 multiplier to super_node distances
   c. If r <= epsilon (0.10) → Exploration: apply 1.0 multiplier (no boost)
6. Merge + re-rank by boosted distance (ascending)
7. Return top_k documents from merged list
8. ATOMIC ACCESS LOGGING (SU10):
   a. For each super_node in results: execute atomic SQL UPDATE hit_count + 1
   b. Do NOT write access_count back to ChromaDB during query path
```

### Updated Component: `retrieval/retriever.py` (SU4 + SU5 + SU10)
```python
import random
import json

def retrieve(query: str, top_k: int = 5,
             epsilon: float = config.EXPLORATION_EPSILON) -> RetrievalResult:
    """
    Unified retrieval function combining:
    - SU4: ε-Greedy Exploration to prevent filter bubble on super-nodes
    - SU5: Hierarchical masking to prevent parent/child vector collision
    - SU10: Atomic SQLite access logging to prevent concurrency counter loss
    """
    q_emb = embedder.encode([query])[0]

    # Dual collection search
    sn_results = chroma_client.super_nodes.query(
        query_embeddings=[q_emb], n_results=top_k,
        include=["documents", "metadatas", "distances", "ids"]
    )
    raw_results = chroma_client.raw_chunks.query(
        query_embeddings=[q_emb], n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # ── SU5: Hierarchical Masking ────────────────────────────────────────────
    # First pass: collect children IDs of any retrieved meta_nodes
    masked_node_ids = set()
    for meta in sn_results["metadatas"][0]:
        if meta.get("type") == "meta_node":
            children_ids = json.loads(meta.get("children", "[]"))
            masked_node_ids.update(children_ids)

    # ── SU4: ε-Greedy Strategy ───────────────────────────────────────────────
    apply_boost = random.random() > epsilon
    multiplier = config.SUPER_NODE_SCORE_MULTIPLIER if apply_boost else 1.0
    # Log which strategy was chosen for observability
    log.info("retrieval_strategy", mode="exploit" if apply_boost else "explore",
             epsilon=epsilon)

    # ── Build candidate list ─────────────────────────────────────────────────
    candidates = []
    super_node_ids_in_results = []

    for doc, meta, dist, id_ in zip(
        sn_results["documents"][0],
        sn_results["metadatas"][0],
        sn_results["distances"][0],
        sn_results["ids"][0]
    ):
        # SU5: Skip child nodes shadowed by their retrieved parent meta_node
        if id_ in masked_node_ids:
            continue

        boosted_score = dist * multiplier
        candidates.append({
            "doc": doc, "meta": meta, "score": boosted_score,
            "type": meta.get("type", "super_node"), "id": id_
        })
        if meta.get("type") == "super_node":
            super_node_ids_in_results.append(id_)

    for doc, meta, dist in zip(
        raw_results["documents"][0],
        raw_results["metadatas"][0],
        raw_results["distances"][0]
    ):
        candidates.append({
            "doc": doc, "meta": meta, "score": dist,
            "type": "raw_chunk", "id": None
        })

    # Sort ascending (lower L2 = better)
    candidates.sort(key=lambda x: x["score"])
    top = candidates[:top_k]

    # ── SU10: Atomic SQLite Access Logging ───────────────────────────────────
    # Never write hit_count to ChromaDB during the query path.
    # SQLite provides atomic increment; ChromaDB does not.
    if super_node_ids_in_results:
        log_super_node_access(super_node_ids_in_results)

    return RetrievalResult(
        documents=[c["doc"] for c in top],
        types=[c["type"] for c in top],
        scores=[c["score"] for c in top]
    )


def log_super_node_access(super_node_ids: list[str]):
    """
    SU10 — Atomic SQLite Counter Update.
    Replaces ChromaDB metadata write for access_count.
    SQLite's atomic UPDATE prevents the read-increment-write race condition
    that loses counter increments under concurrent benchmark load.
    """
    placeholders = ",".join("?" * len(super_node_ids))
    sqlite_client.execute(
        f"""
        UPDATE query_clusters
        SET hit_count = hit_count + 1, last_hit = CURRENT_TIMESTAMP
        WHERE super_node_id IN ({placeholders})
        """,
        super_node_ids
    )
```

### ⚠️ Hidden Engineering Problems
1. **Score boost interaction with L2 vs cosine distance:** ChromaDB returns L2 distance by default. Multiplying by 0.85 works for L2 but breaks for cosine (where distance is already 0–2). Explicitly set `space="cosine"` when creating collections and verify distance semantics before applying multiplier.
2. **Exploration rate in benchmarks:** Setting `epsilon=0.10` means 10% of queries skip the boost. In a controlled 200-query benchmark, ~20 queries will use exploration mode. Track `retrieval_strategy` label in metrics to separate exploitation vs exploration results when computing super-node hit rates.
3. **Hierarchical masking and top_k:** Masking children may reduce the effective super_node candidate count below `top_k`. The fill is provided by raw_chunk candidates, which is the correct fallback behavior.

### Metrics
- `retrieval_type_distribution` — super_node vs raw_chunk vs meta_node (per query)
- `super_node_hit_rate` — % of queries where a super_node was top result
- `query_latency_ms` — compare Phase 1 vs Phase 5 (target: <200ms)
- `context_token_reduction` — avg tokens Phase 1 vs Phase 5
- `exploration_rate` — actual % of queries in exploration mode (should ≈ epsilon)
- `masked_children_count` — per query, nodes excluded by hierarchical masking (SU5)

### Changes Introduced

**SU4 — ε-Greedy Retrieval Router**
- *Old behavior:* Super-nodes received a deterministic 0.85 L2 distance multiplier on every query, unconditionally elevating them above raw chunks. Over time this created a filter bubble: high-access super-nodes monopolized retrieval, suppressing newly ingested raw chunks regardless of their relevance.
- *New behavior:* On 10% of queries (`epsilon=0.10`), the multiplier is set to 1.0 (no boost), forcing the router to rank super-nodes and raw chunks purely by semantic similarity. This surfaces newly ingested documents that may have overtaken stale super-nodes in relevance.
- *Reason:* Eliminates algorithmic filter bubble bias. Exploration queries provide honest feedback on whether the super-node is still the best answer, enabling the decay scorer to eventually prune nodes no longer competitive on raw merit.

**SU5 — Hierarchical Masking**
- *Old behavior:* When a meta-node and its child super-nodes were all retrieved for the same query, the context window was flooded with redundant content (meta-node text + child summaries covering the same topic). This inflated context token counts and invalidated "Context Token Economy" benchmark metrics.
- *New behavior:* A first-pass scan of super-node results identifies any `meta_node` entries and collects their `children` IDs. In the second pass, any candidate whose ID is in `masked_node_ids` is skipped. The top_k slots vacated by masked children are filled by raw_chunk candidates.
- *Reason:* Prevents parent-child redundancy from corrupting token economy metrics and degrading synthesis quality.

**SU10 — Atomic SQLite Access Counter**
- *Old behavior:* `update_access()` wrote `access_count + 1` back to ChromaDB metadata as an async task. Under concurrent benchmark load (50 threads), all threads read the same value simultaneously, incremented it in memory, and wrote the same value back — losing all but one increment per batch.
- *New behavior:* `log_super_node_access()` executes a single atomic SQL `UPDATE hit_count = hit_count + 1` statement in SQLite. SQLite's row-level locking ensures no increments are lost. ChromaDB `access_count` in metadata is a stale mirror updated only during the maintenance pass (Phase 6).
- *Reason:* Eliminates concurrency counter loss. SQLite's atomicity guarantees that all 50 concurrent increments are recorded, ensuring decay scores are computed on accurate utilization data.

---

## PHASE 6 — Maintenance, Decay & Hierarchy

### Goal
Prevent database bloat and semantic drift. Four sub-systems: (1) Soft Staleness Check (SU8), (2) Half-Life Decay Scorer (SU3, SU10), (3) Hierarchy Merger with lock flag (SU7), (4) Cascade-safe maintenance coordination.

### Components
```
app/
  maintenance/
    staleness_checker.py   # Soft-staleness: mark is_stale, trigger async re-synthesis (SU8)
    decay_scorer.py        # Half-life exponential decay + SQLite counter sync (SU3, SU10)
    hierarchy_merger.py    # Merge similar super-nodes with in_hierarchy lock (SU7)
  scheduler/
    jobs/
      maintenance_job.py   # APScheduler job wrapping all 3
```

### Data Structures
```python
@dataclass
class DecayScore:
    super_node_id: str
    score: float
    access_count: int         # sourced from SQLite (SU10)
    days_since_last_access: int   # based on last_accessed, not created_at (SU3)
    should_prune: bool

@dataclass
class MetaNode:
    id: str                    # "mn_{timestamp}"
    text: str
    child_super_node_ids: List[str]
    source_queries: List[str]
    created_at: datetime
```

### Algorithm

#### Soft Staleness Check (SU8)
```
1. For each super_node in ChromaDB:
   a. source_chunks = JSON.parse(meta["source_chunks"])
   b. Query raw_chunks for all source_chunk IDs
   c. match_ratio = len(existing_ids) / len(source_chunk_ids)
   d. If match_ratio < 1.0 (some chunks shifted or deleted):
      - Set meta["is_stale"] = True  ← DO NOT DELETE
      - UPDATE ChromaDB metadata (preserve hit_count history)
      - Flag cluster for async re-synthesis in SQLite
      - The stale super-node continues serving queries until replacement is ready
2. Background re-synthesis rebuilds the super-node from updated raw chunks
3. On successful re-synthesis: replace old super-node, clear is_stale flag
```

#### Half-Life Decay Scorer (SU3 + SU10)
```
Formula: score = min(access_count, 10) * exp(-days_since_last_access / HALF_LIFE_DAYS)
         HALF_LIFE_DAYS = 7 (configurable)
         access_count sourced from SQLite (SU10), not ChromaDB metadata

1. For each super_node:
   a. Pull true access_count from SQLite (atomic source of truth)
   b. Compute days_since_last_access (from meta["last_accessed"])
   c. Compute half-life decay score
   d. UPDATE ChromaDB metadata decay_score (sync)
   e. UPDATE SQLite last_hit to match ChromaDB last_accessed
   f. If decay_score < PRUNE_THRESHOLD (0.05):
      - DELETE from ChromaDB
      - Log pruning event
```

#### Hierarchy Merger with Lock Flag and Ground-Truth Re-Synthesis (SU7 + SU11)
```
1. Fetch ONLY super_nodes where in_hierarchy == False
   (SU7: Prevents re-pairing already-merged children)
2. Skip nodes with lineage_depth >= MAX_LINEAGE_DEPTH (SU11: depth cap)
3. Compute pairwise cosine similarity matrix
4. Find pairs with sim > 0.88
5. For each qualifying pair (sn_A, sn_B):
   a. SU11 — Ground-Truth Re-Synthesis:
      - Union(sn_A.source_chunks, sn_B.source_chunks) → all raw chunk IDs
      - DO NOT read sn_A.document or sn_B.document as synthesis input
      - Build SynthesisJob(chunk_ids=unioned_raw_ids)
      - Call synthesizer.run(job) → full Phase 4 pipeline (dedup + LLM + validate)
   b. Upsert resulting super-node as meta_node (type="meta_node", children=[sn_A.id, sn_B.id])
   c. Set lineage_depth = max(sn_A.lineage_depth, sn_B.lineage_depth) + 1
   d. UPDATE sn_A and sn_B: set in_hierarchy=True, parent_meta_id=meta_id, lineage_depth preserved
      (SU7: Locks them out of future merger passes)
6. Retrieval order: meta_node → super_node → raw_chunk
```

### Updated Component: `maintenance/staleness_checker.py` (SU8)
```python
def run_soft_staleness_check():
    """
    SU8 — Soft-Staleness Validation.
    Replaces hard DELETE on stale super-nodes with a soft is_stale flag.
    Stale nodes continue serving queries while async re-synthesis runs,
    preserving months of hit_count and access history accumulated from
    organic user intent signals.
    """
    all_sn = chroma_client.super_nodes.get(
        where={"type": {"$eq": "super_node"}},
        include=["metadatas", "ids"]
    )

    for sn_id, meta in zip(all_sn["ids"], all_sn["metadatas"]):
        source_chunk_ids = json.loads(meta["source_chunks"])

        # Verify how many original IDs still exist in the raw collection
        try:
            existing = chroma_client.raw_chunks.get(
                ids=source_chunk_ids, include=["ids"]
            )
            match_ratio = len(existing["ids"]) / len(source_chunk_ids)
        except Exception as e:
            log.warning("staleness_check_error", sn_id=sn_id, error=str(e))
            continue

        if match_ratio < 1.0:
            # Soft flag: mark stale without deleting historical access metrics
            updated_meta = {**meta, "is_stale": True}
            chroma_client.super_nodes.update(ids=[sn_id], metadatas=[updated_meta])

            # Queue background re-synthesis via SQLite flag
            sqlite_client.flag_cluster_for_re_synthesis(meta["cluster_id"])

            log.info("super_node_marked_stale", sn_id=sn_id,
                     match_ratio=match_ratio, cluster_id=meta["cluster_id"])
```

### Updated Component: `maintenance/decay_scorer.py` (SU3 + SU10)
```python
import math
from datetime import datetime

def compute_exponential_decay(meta: dict, half_life_days: int = 7) -> float:
    """
    SU3 — Half-Life Exponential Decay.
    Replaces the linear formula score = access_count / (days_since_created + 1).
    That formula guaranteed terminal decay for all nodes regardless of recent usage.
    This formula judges relevancy based on recency of access, not total age.
    """
    last_accessed = datetime.fromisoformat(meta["last_accessed"])
    days_since_last_access = (datetime.utcnow() - last_accessed).days

    # Exponential half-life decay
    decay_factor = math.exp(-days_since_last_access / half_life_days)

    # Cap access_count at 10 to prevent runaway scores on viral nodes
    base_utility = min(meta["access_count_sqlite"], 10)

    return base_utility * decay_factor


def compute_and_apply_decay():
    """
    SU10 integration: pulls true access_count from SQLite before computing decay.
    Syncs the authoritative SQLite hit_count into ChromaDB metadata as access_count.
    """
    all_sn = chroma_client.super_nodes.get(
        where={"type": {"$eq": "super_node"}},
        include=["metadatas", "ids"]
    )

    for sn_id, meta in zip(all_sn["ids"], all_sn["metadatas"]):
        # SU10: Pull true access count from SQLite (atomic source of truth)
        sqlite_row = sqlite_client.get_cluster_by_super_node_id(sn_id)
        true_access_count = sqlite_row.hit_count if sqlite_row else meta.get("access_count", 0)

        # Inject SQLite count for decay computation
        meta_with_sqlite = {**meta, "access_count_sqlite": true_access_count}
        score = compute_exponential_decay(meta_with_sqlite,
                                          half_life_days=config.HALF_LIFE_DAYS)

        # Sync count back to ChromaDB metadata for query-time inspection
        updated_meta = {**meta, "decay_score": score, "access_count": true_access_count}
        chroma_client.super_nodes.update(ids=[sn_id], metadatas=[updated_meta])

        if score < config.PRUNE_THRESHOLD:
            chroma_client.super_nodes.delete(ids=[sn_id])
            log.info("node_pruned", sn_id=sn_id, score=score,
                     days_since_access=(datetime.utcnow() -
                         datetime.fromisoformat(meta["last_accessed"])).days)
```

### Updated Component: `maintenance/hierarchy_merger.py` (SU7 + SU11)
```python
def merge_similar_nodes():
    """
    SU7 + SU11 — Hierarchy Merger with Ground-Truth Re-Synthesis.

    SU7: Only processes super_nodes not already in a hierarchy (in_hierarchy=False).
    SU11: NEVER reads .documents (generated text) as synthesis input.
          Instead, unions source_chunks (raw chunk IDs) and re-runs the full
          Phase 4 synthesizer pipeline (fetch raw → dedup → LLM → validate).
          Enforces MAX_LINEAGE_DEPTH cap to prevent unbounded meta-node nesting.
    """
    # SU7: Only pull super_nodes NOT already in a hierarchy
    all_sn = chroma_client.super_nodes.get(
        where={
            "$and": [
                {"type": {"$eq": "super_node"}},
                {"in_hierarchy": {"$eq": False}}  # Exclude already-merged nodes
            ]
        },
        include=["metadatas", "embeddings", "ids"]
        # NOTE: "documents" intentionally excluded — SU11: we never read generated text here
    )

    if len(all_sn["ids"]) < 2:
        return

    embs = np.array(all_sn["embeddings"])
    sim_matrix = cosine_similarity(embs)
    np.fill_diagonal(sim_matrix, 0)
    pairs = np.argwhere(sim_matrix > config.MERGE_THRESHOLD)  # 0.88
    pairs = [(a, b) for a, b in pairs if a < b]   # deduplicate

    for i, j in pairs:
        meta_i = all_sn["metadatas"][i]
        meta_j = all_sn["metadatas"][j]

        # SU11: Depth cap — refuse to create meta-nodes beyond MAX_LINEAGE_DEPTH
        depth_i = meta_i.get("lineage_depth", 0)
        depth_j = meta_j.get("lineage_depth", 0)
        merged_depth = max(depth_i, depth_j) + 1
        if merged_depth > config.MAX_LINEAGE_DEPTH:
            log.warning("merger_depth_cap_reached",
                        sn_a=all_sn["ids"][i], sn_b=all_sn["ids"][j],
                        merged_depth=merged_depth, cap=config.MAX_LINEAGE_DEPTH)
            # Flag for manual review instead of auto-merging
            sqlite_client.flag_for_manual_review(
                all_sn["ids"][i], all_sn["ids"][j],
                reason=f"lineage_depth_cap: would reach depth {merged_depth}"
            )
            continue

        # SU11: Union source_chunks IDs from both children — these are always raw doc_ IDs
        # (guaranteed by SU9 ground-truth anchoring at logging time)
        source_ids_i = json.loads(meta_i.get("source_chunks", "[]"))
        source_ids_j = json.loads(meta_j.get("source_chunks", "[]"))
        unioned_raw_ids = list(set(source_ids_i + source_ids_j))

        # Assert no super-node IDs leaked into lineage (defensive check)
        assert all(id_.startswith("doc_") for id_ in unioned_raw_ids), \
            f"Super-node IDs detected in source_chunks union — SU9 violated!"

        # SU11: Run full Phase 4 synthesis pipeline on raw IDs — NOT on .documents text
        merge_query = f"Synthesis of: {meta_i['source_query']} AND {meta_j['source_query']}"
        job = SynthesisJob(
            cluster_id=None,        # meta-nodes don't belong to a single cluster
            canonical_query=merge_query,
            chunk_ids=unioned_raw_ids,
            hit_count=meta_i.get("hit_count", 0) + meta_j.get("hit_count", 0),
            triggered_at=datetime.utcnow(),
            existing_sn_id=None     # Always create a new meta-node ID
        )
        try:
            meta_id = synthesizer.run(job)  # Returns sn_id which we'll treat as meta_node
        except SynthesisError as e:
            log.error("merger_synthesis_failed", sn_a=all_sn["ids"][i], sn_b=all_sn["ids"][j],
                      error=str(e))
            continue

        # Rename the synthesized node to a meta_node and set its metadata
        merged_meta = chroma_client.super_nodes.get(
            ids=[meta_id], include=["metadatas"]
        )["metadatas"][0]
        chroma_client.super_nodes.update(
            ids=[meta_id],
            metadatas=[{**merged_meta,
                        "type": "meta_node",
                        "children": json.dumps([all_sn["ids"][i], all_sn["ids"][j]]),
                        "lineage_depth": merged_depth,
                        "source_chunks": json.dumps(unioned_raw_ids),  # raw IDs preserved
                        "in_hierarchy": False   # meta_nodes themselves are not children
                       }]
        )

        # SU7: Lock children out of future merger passes; record parent link for SU13
        for idx in [i, j]:
            child_id = all_sn["ids"][idx]
            child_meta = all_sn["metadatas"][idx]
            updated_child_meta = {
                **child_meta,
                "in_hierarchy": True,
                "parent_meta_id": meta_id   # SU13 uses this to propagate upward on re-synthesis
            }
            chroma_client.super_nodes.update(ids=[child_id], metadatas=[updated_child_meta])

        log.info("meta_node_created", meta_id=meta_id,
                 children=[all_sn["ids"][i], all_sn["ids"][j]],
                 lineage_depth=merged_depth,
                 unioned_raw_chunks=len(unioned_raw_ids))
```

### Background Jobs
| Job | Trigger | Interval |
|-----|---------|----------|
| `staleness_check` | Interval | 1 hour |
| `decay_scorer` | Interval | 6 hours |
| `hierarchy_merger` | Interval | 24 hours |

### ⚠️ Hidden Engineering Problems
1. **Pairwise cosine on 10k super-nodes = 10^8 operations.** Cap hierarchy merger at 500 most-accessed super-nodes per run (filter by SQLite hit_count before fetching from ChromaDB), or use FAISS to find approximate neighbors.
2. **Soft-staleness and active queries:** A node marked `is_stale=True` continues to be returned by the retriever. Add an optional `is_stale` filter to retrieval if research requirements demand fresh-only results.
3. **Meta-node creation loop (capped by SU7):** The `in_hierarchy` flag prevents child nodes from being re-paired. However, if two meta-nodes are themselves similar, they will be paired. Cap hierarchy at depth=2 by checking `depth` in metadata; skip merging if `depth >= 2`.
4. **SQLite-ChromaDB sync consistency:** The decay scorer is the only place where ChromaDB `access_count` is synced from SQLite. If the scheduler crashes mid-sync, some nodes may have stale `access_count` in ChromaDB. This is acceptable — ChromaDB `access_count` is advisory only; SQLite is authoritative.

### Metrics
- `staleness_check_duration_ms`
- `stale_nodes_flagged` (counter — SU8, not deleted — distinguishes from v1 hard deletes)
- `re_synthesis_queued` (counter — SU8)
- `decay_score_distribution` (histogram — SU3)
- `nodes_pruned` (counter)
- `meta_nodes_created` (counter — SU7)
- `nodes_locked_in_hierarchy` (counter — SU7)
- `sqlite_chroma_access_count_delta` (gauge — SU10 sync accuracy)

### Unit Tests
```python
# SU8 tests
def test_staleness_check_marks_stale_not_deletes():
    sn_id = create_super_node(source_chunks=["doc_c1", "doc_c2"])
    chroma_client.raw_chunks.delete(ids=["doc_c1"])  # Simulate upstream edit
    run_soft_staleness_check()
    node = chroma_client.super_nodes.get(ids=[sn_id])
    assert node["metadatas"][0]["is_stale"] == True   # Marked, not deleted
    assert len(node["ids"]) == 1                       # Still exists

def test_staleness_check_queues_re_synthesis():
    sn_id = create_super_node(source_chunks=["doc_c1"])
    chroma_client.raw_chunks.delete(ids=["doc_c1"])
    run_soft_staleness_check()
    pending = sqlite_client.get_clusters_pending_re_synthesis()
    assert any(c.super_node_id == sn_id for c in pending)

# SU3 tests
def test_decay_score_recent_access_survives():
    meta = {"last_accessed": datetime.utcnow().isoformat(), "access_count_sqlite": 5}
    score = compute_exponential_decay(meta, half_life_days=7)
    assert score > config.PRUNE_THRESHOLD  # Recently accessed → not pruned

def test_decay_score_old_node_pruned():
    old_date = (datetime.utcnow() - timedelta(days=60)).isoformat()
    meta = {"last_accessed": old_date, "access_count_sqlite": 1}
    score = compute_exponential_decay(meta, half_life_days=7)
    assert score < config.PRUNE_THRESHOLD

def test_linear_decay_terminal_trap_absent():
    # A node created 365 days ago but accessed yesterday should survive
    meta = {
        "last_accessed": (datetime.utcnow() - timedelta(days=1)).isoformat(),
        "access_count_sqlite": 8
    }
    score = compute_exponential_decay(meta, half_life_days=7)
    assert score > 1.0  # High recent utility overrides age

# SU7 tests
def test_merger_skips_nodes_already_in_hierarchy():
    sn_a = create_super_node(in_hierarchy=True)  # Already merged
    sn_b = create_super_node(in_hierarchy=False)
    merge_similar_nodes()
    meta_nodes = chroma_client.super_nodes.get(where={"type": {"$eq": "meta_node"}})
    assert len(meta_nodes["ids"]) == 0  # No pairing possible with only 1 eligible node

def test_merger_locks_children_after_merge():
    sn_a = create_near_identical_super_nodes(2)
    merge_similar_nodes()
    for sn_id in sn_a:
        node = chroma_client.super_nodes.get(ids=[sn_id])
        assert node["metadatas"][0]["in_hierarchy"] == True

# SU11 tests
def test_merger_uses_raw_chunk_ids_not_documents():
    """SU11: merger must never read .documents (generated text) to feed synthesis."""
    sn_a = create_super_node(source_chunks=["doc_c1", "doc_c2"], in_hierarchy=False)
    sn_b = create_super_node(source_chunks=["doc_c3", "doc_c4"], in_hierarchy=False)
    # Make them similar enough to trigger merge
    mock_similar_embeddings(sn_a, sn_b)

    merge_similar_nodes()

    meta_nodes = chroma_client.super_nodes.get(where={"type": {"$eq": "meta_node"}},
                                                include=["metadatas"])
    assert len(meta_nodes["ids"]) == 1
    # meta-node's source_chunks must be the union of raw IDs, NOT any sn_* IDs
    meta_source = json.loads(meta_nodes["metadatas"][0]["source_chunks"])
    assert all(id_.startswith("doc_") for id_ in meta_source), \
        "SU11 violated: meta-node source_chunks contains non-doc_ IDs"
    assert set(meta_source) == {"doc_c1", "doc_c2", "doc_c3", "doc_c4"}

def test_merger_enforces_depth_cap():
    """SU11: merges that would exceed MAX_LINEAGE_DEPTH are refused and flagged."""
    # Create two super-nodes already at depth = MAX_LINEAGE_DEPTH
    sn_a = create_super_node(lineage_depth=config.MAX_LINEAGE_DEPTH, in_hierarchy=False)
    sn_b = create_super_node(lineage_depth=config.MAX_LINEAGE_DEPTH, in_hierarchy=False)
    mock_similar_embeddings(sn_a, sn_b)

    merge_similar_nodes()

    meta_nodes = chroma_client.super_nodes.get(where={"type": {"$eq": "meta_node"}})
    assert len(meta_nodes["ids"]) == 0, "Merger should refuse to create node beyond depth cap"
    flagged = sqlite_client.get_manual_review_queue()
    assert len(flagged) == 1

# SU10 tests
def test_decay_scorer_uses_sqlite_not_chroma_counter():
    sn_id = create_super_node()
    # Set ChromaDB access_count to stale value
    chroma_client.super_nodes.update(ids=[sn_id], metadatas=[{"access_count": 1}])
    # Set SQLite to true value
    sqlite_client.execute("UPDATE query_clusters SET hit_count = 50 WHERE super_node_id = ?", [sn_id])
    compute_and_apply_decay()
    node = chroma_client.super_nodes.get(ids=[sn_id])
    assert node["metadatas"][0]["access_count"] == 50  # Synced from SQLite
```

### Changes Introduced

**SU3 — Half-Life Exponential Decay**
- *Old behavior:* `score = access_count / (days_since_created + 1)`. Because `days_since_created` grows unboundedly, all nodes are mathematically guaranteed to eventually fall below `PRUNE_THRESHOLD`, even if they were accessed yesterday. A long-running benchmark would see massive false-positive pruning of still-valuable nodes.
- *New behavior:* `score = min(access_count, 10) * exp(-days_since_last_access / half_life_days)`. Decay is anchored to `last_accessed`, not `created_at`. A node accessed recently scores high regardless of total age; a node untouched for 4+ half-lives naturally approaches zero.
- *Reason:* Aligns pruning logic with human memory models and radioactive decay mathematics. Ensures nodes are pruned for disuse, not longevity — a critical distinction for research accuracy.

**SU7 — Infinite Merger Loop Prevention**
- *Old behavior:* `merge_similar_nodes()` fetched ALL super-nodes with `type == "super_node"`. After merging sn_A and sn_B into meta_node MN, both sn_A and sn_B remained as children with `cosine_sim > 0.88`. On the next daily run, they were re-paired and a duplicate MN was created. Over weeks: exponential meta_node duplication.
- *New behavior:* The ChromaDB `where` filter explicitly excludes nodes with `in_hierarchy == True`. After a merge, `hierarchy_merger.py` sets `in_hierarchy = True` and `parent_meta_id = meta_id` on both children before returning. They are permanently excluded from future pairing passes.
- *Reason:* Prevents database bloat from exponential meta-node duplication and maintains a clean, non-redundant hierarchy structure.

**SU8 — Cascading Amnesia Prevention**
- *Old behavior:* If any source chunk ID was missing, the staleness checker executed `DELETE` on the super-node and reset `synthesized=0` in SQLite. A minor upstream typo fix would trigger re-chunking, generate new hash IDs, and cause the staleness checker to wipe months of hit_count and access metrics.
- *New behavior:* Instead of deletion, the checker sets `is_stale=True` in ChromaDB metadata and flags the cluster for async re-synthesis in SQLite. The stale node continues serving queries with its intact history until the new synthesis completes and replaces it.
- *Reason:* Preserves organic user intent signals accumulated over months. The re-synthesis path is triggered asynchronously, so there is no service disruption and no metric loss from minor upstream edits.

**SU11 — Ground-Truth Re-Synthesis in Hierarchy Merger**
- *Old behavior:* `merge_similar_nodes()` called `synthesis_merge(sn_A.document, sn_B.document)` — feeding the generated summaries directly into a new LLM call. This is a summary-of-summaries pattern: each node has already dropped up to 10% of source facts (by the 90% validation threshold). Merging them compounds this loss. At hierarchy depth ≥2, validation still reports green but is scoring coverage-of-a-lossy-summary, not coverage-of-original-facts. The degradation is structurally invisible.
- *New behavior:* The merger reads `source_chunks` (raw chunk ID lists) from both children, takes their union, and runs `synthesizer.run(job)` — the exact same Phase 4 pipeline (fetch raw → SU1 dedup → LLM synthesize → SU2 LLM-judge validate). The `.documents` (generated text) field is never consumed as synthesis input anywhere in the pipeline. A `lineage_depth` counter tracks merge depth; `MAX_LINEAGE_DEPTH` (default 2) caps automatic merging and flags deeper candidates for manual review.
- *Reason:* Guarantees that no matter how many merge generations deep a meta-node is, its synthesis always traces back to a flat list of original `doc_`-prefixed chunk IDs. The validation score at every depth is a real score against real source material, not a score against a prior compression.

---

## COMPLETE REPOSITORY STRUCTURE

```
self-evolving-rag/
├── README.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── pyproject.toml
│
├── app/
│   ├── main.py                        # FastAPI app + lifespan
│   ├── config.py                      # All hyperparameters + env vars
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── ingest.py              # POST /ingest/medquad
│   │       ├── query.py               # POST /query
│   │       └── admin.py               # Admin endpoints
│   │
│   ├── db/
│   │   ├── chroma_client.py           # ChromaDB singleton
│   │   ├── sqlite_client.py           # SQLite: query_clusters + benchmark_pairs
│   │   │                              #   + update_cluster_embedding() (SU6)
│   │   │                              #   + flag_cluster_for_re_synthesis() (SU8)
│   │   │                              #   + get_cluster_by_super_node_id() (SU10)
│   │   └── migrations/
│   │       └── 001_initial.sql
│   │
│   ├── ingestion/
│   │   ├── dataset_loader.py          # HF MedQuAD streaming loader
│   │   ├── url_extractor.py           # Dedup document_url per document_id
│   │   ├── web_fetcher.py             # Rate-limited HTTP crawler (httpx + tenacity)
│   │   ├── html_cleaner.py            # Strip boilerplate → plain text (BS4)
│   │   ├── chunker.py                 # ~500-token semantic chunker
│   │   ├── embedder.py                # SentenceTransformer wrapper
│   │   ├── benchmark_loader.py        # Write Q/A pairs → benchmark_pairs table
│   │   └── ingestor.py                # Orchestrates full pipeline
│   │
│   ├── retrieval/
│   │   ├── retriever.py               # ε-greedy + hierarchical masking + SQLite logging
│   │   │                              #   (SU4 + SU5 + SU10)
│   │   └── score_booster.py           # Score boost logic (called conditionally)
│   │
│   ├── generation/
│   │   └── generator.py
│   │
│   ├── logging_/
│   │   ├── query_logger.py            # + update_cluster_centroid() (SU6)
│   │   └── normalizer.py
│   │
│   ├── middleware/
│   │   └── query_interceptor.py       # + ground-truth doc_ filter (SU9)
│   │
│   ├── patterns/
│   │   └── finder.py
│   │
│   ├── synthesis/
│   │   ├── chunk_fetcher.py           # fetch_and_deduplicate() (SU1)
│   │   ├── synthesizer.py
│   │   └── prompts.py
│   │
│   ├── validation/
│   │   ├── validator.py               # LLM-as-judge entailment (SU2)
│   │   └── entity_extractor.py        # Retained for hallucination additions check
│   │
│   ├── maintenance/
│   │   ├── staleness_checker.py       # Soft-staleness flag (SU8)
│   │   ├── decay_scorer.py            # Half-life decay + SQLite sync (SU3 + SU10)
│   │   └── hierarchy_merger.py        # in_hierarchy lock flag (SU7)
│   │
│   └── scheduler/
│       ├── scheduler.py
│       └── jobs/
│           ├── pattern_finder.py
│           └── maintenance_job.py
│
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   └── sample_medquad_rows.json   # 20 rows for offline tests
│   ├── test_phase1_ingestion.py
│   ├── test_phase2_logging.py         # + centroid update (SU6) + ground-truth filter (SU9)
│   ├── test_phase3_scheduler.py
│   ├── test_phase4_synthesis.py       # + dedup (SU1) + LLM-judge (SU2)
│   ├── test_phase5_retrieval.py       # + ε-greedy (SU4) + masking (SU5) + atomic (SU10)
│   └── test_phase6_maintenance.py     # + soft-stale (SU8) + half-life (SU3) + lock (SU7)
│
├── scripts/
│   ├── benchmark.py                   # Benchmarking suite (uses benchmark_pairs)
│   └── seed_data.py                   # Ingest N rows from MedQuAD for dev
│
└── data/
    ├── chroma/                        # ChromaDB persistence
    └── sqlite/
        └── rag.db                     # query_clusters + benchmark_pairs + logs
```

---

## DATABASE SCHEMAS (Complete SQL — v2)

```sql
-- benchmark_pairs: MedQuAD Q/A pairs for evaluation ONLY (not ingested into vector DB)
CREATE TABLE IF NOT EXISTS benchmark_pairs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id      TEXT NOT NULL UNIQUE,
    document_id      TEXT NOT NULL,
    document_url     TEXT NOT NULL,
    question_focus   TEXT,
    question_type    TEXT,
    question         TEXT NOT NULL,
    answer           TEXT NOT NULL,
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_bp_document ON benchmark_pairs(document_id);
CREATE INDEX IF NOT EXISTS idx_bp_type ON benchmark_pairs(question_type);

-- query_clusters: core pattern tracking table
CREATE TABLE IF NOT EXISTS query_clusters (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_query  TEXT NOT NULL,
    query_embedding  BLOB NOT NULL,       -- running centroid (SU6), not first-query locked
    hit_count        INTEGER NOT NULL DEFAULT 1,   -- atomic source of truth (SU10)
    chunk_ids        TEXT NOT NULL DEFAULT '[]',   -- raw_chunk IDs only (SU9)
    first_seen       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_hit         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    synthesized      BOOLEAN NOT NULL DEFAULT 0,   -- legacy; kept for migration compatibility
    synthesizing     BOOLEAN NOT NULL DEFAULT 0,   -- concurrency lock (SU12)
    synthesis_failed BOOLEAN NOT NULL DEFAULT 0,
    super_node_id    TEXT,
    retry_count      INTEGER NOT NULL DEFAULT 0,
    pending_re_synthesis BOOLEAN NOT NULL DEFAULT 0,  -- set by soft-staleness check (SU8) or SU13 upward propagation
    last_synthesized_hit_count INTEGER NOT NULL DEFAULT 0  -- SU12: re-trigger when (hit_count - this) >= RESYNTH_DELTA
);

CREATE INDEX IF NOT EXISTS idx_hit_count_synth
    ON query_clusters(hit_count DESC, synthesized, synthesizing);
CREATE INDEX IF NOT EXISTS idx_re_synthesis
    ON query_clusters(pending_re_synthesis, synthesized);
CREATE INDEX IF NOT EXISTS idx_resynth_delta
    ON query_clusters(hit_count, last_synthesized_hit_count, synthesizing);  -- SU12: delta query

-- query_log: raw query audit trail
CREATE TABLE IF NOT EXISTS query_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_query        TEXT NOT NULL,
    normalized_query TEXT,
    cluster_id       INTEGER REFERENCES query_clusters(id),
    retrieved_chunks TEXT DEFAULT '[]',   -- raw_chunk IDs only (SU9)
    retrieval_type   TEXT,                -- 'super_node' | 'raw_chunk' | 'mixed' | 'explore'
    latency_ms       REAL,
    timestamp        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_query_log_cluster
    ON query_log(cluster_id, timestamp DESC);

-- synthesis_log: audit every synthesis attempt
CREATE TABLE IF NOT EXISTS synthesis_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id       INTEGER REFERENCES query_clusters(id),
    attempt          INTEGER NOT NULL DEFAULT 1,
    coverage_score   REAL,
    passed           BOOLEAN,
    super_node_id    TEXT,
    error_message    TEXT,
    duration_ms      REAL,
    timestamp        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- maintenance_log: track decay and pruning
CREATE TABLE IF NOT EXISTS maintenance_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type       TEXT NOT NULL,  -- 'prune'|'soft_stale'|'merge'|'re_synthesis_queued'|'depth_cap_refused'|'manual_review_flagged'
    super_node_id    TEXT,
    reason           TEXT,           -- 'child_revised' (SU13) | 'source_chunk_drift' (SU8) | etc.
    decay_score      REAL,
    timestamp        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- manual_review_queue: SU11 depth-cap violations flagged for human inspection
CREATE TABLE IF NOT EXISTS manual_review_queue (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    sn_id_a          TEXT NOT NULL,
    sn_id_b          TEXT NOT NULL,
    reason           TEXT NOT NULL,
    flagged_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved         BOOLEAN NOT NULL DEFAULT 0
);
```

---

## HYPERPARAMETERS (v2)

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `CHUNK_SIZE_TOKENS` | 500 | Balances context density vs retrieval precision |
| `CHUNK_OVERLAP_TOKENS` | 50 | Prevents boundary information loss |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 1024-dim, high-accuracy multi-vector baseline |
| `EMBEDDING_BATCH_SIZE` | 64 | GPU-efficient batch size |
| `CRAWL_RATE_PER_SEC` | 1 | NLM rate-limit compliance (1 req/sec) |
| `CRAWL_TIMEOUT_SEC` | 15 | Per-URL HTTP timeout |
| `CRAWL_MAX_RETRIES` | 3 | Tenacity retry limit per URL |
| `USER_AGENT` | `"MedRAG-Research/1.0"` | Identify crawler to NLM servers |
| `HF_DATASET_CACHE_DIR` | `"data/hf_cache"` | Avoid re-downloading MedQuAD |
| `QUERY_CLUSTER_COSINE_THRESHOLD` | 0.92 | High threshold to avoid over-merging |
| `HIT_COUNT_THRESHOLD` | 10 | Min hits before synthesis trigger |
| `RESYNTH_DELTA` | 10 | **SU12**: additional hits since last synthesis before re-triggering |
| `POLL_INTERVAL_MINUTES` | 5 | Scheduler poll frequency |
| `SUPER_NODE_SCORE_MULTIPLIER` | 0.85 | L2 distance boost for super-nodes (exploitation mode) |
| `EXPLORATION_EPSILON` | 0.10 | Fraction of queries that skip score boost (SU4) |
| `VALIDATION_COVERAGE_THRESHOLD` | 0.90 | Minimum fact coverage to accept synthesis |
| `MAX_SYNTHESIS_RETRIES` | 3 | LLM synthesis retry limit |
| `DECAY_PRUNE_THRESHOLD` | 0.05 | Score below which node is pruned |
| `HALF_LIFE_DAYS` | 7 | Half-life for exponential decay (SU3) |
| `DEDUP_DISTANCE_THRESHOLD` | 0.15 | Cosine distance threshold for chunk deduplication (SU1) |
| `DEDUP_MIN_CHUNKS` | 5 | Minimum chunks before deduplication is applied (SU1) |
| `SYNTHESIS_MAX_TOKENS` | 4000 | Token budget for synthesis input after dedup (SU1) |
| `HIERARCHY_MERGE_THRESHOLD` | 0.88 | Cosine sim above which nodes merge |
| `MAX_LINEAGE_DEPTH` | 2 | **SU11**: Maximum merge nesting depth; beyond this, flag for manual review |
| `MAX_DRIFT_MARGIN` | 0.08 | **SU14**: Max allowed `drift_margin` before fidelity_flagged=True is set |
| `MIN_SOURCE_ANCHOR` | 0.55 | **SU14**: Minimum `sim_summary_to_source`; below this forces re-synthesis |
| `GENERATOR_LLM_MODEL` | `Meta-Llama-3.1-8B-Instruct` | User-facing answer generation model |
| `SUMMARIZER_LLM_MODEL` | `Qwen-2.5-32B-Instruct` | Phase 4 Summarizer and LLM-as-Judge validation model |
| `LLM_TEMPERATURE` | 0.1 | Low temp for factual consistency |
| `TOP_K_RAW` | 5 | Raw chunks retrieved per query |
| `TOP_K_SUPER` | 3 | Super-nodes retrieved per query |

---

## BUILD ORDER (File-by-File) — v2

```
Week 1 — Foundation (MedQuAD Ingestion Pipeline)
  1. config.py                     (add EXPLORATION_EPSILON, HALF_LIFE_DAYS,
                                     DEDUP_* params, SYNTHESIS_MAX_TOKENS)
  2. db/chroma_client.py
  3. db/sqlite_client.py           (add update_cluster_embedding, SU6)
                                   (add flag_cluster_for_re_synthesis, SU8)
                                   (add get_cluster_by_super_node_id, SU10)
                                   + migrations/001_initial.sql
  4. ingestion/embedder.py
  5. ingestion/chunker.py
  6. ingestion/web_fetcher.py
  7. ingestion/html_cleaner.py
  8. ingestion/dataset_loader.py
  9. ingestion/benchmark_loader.py
  10. ingestion/ingestor.py
  11. retrieval/retriever.py
  12. generation/generator.py
  13. api/routes/ingest.py
  14. api/routes/query.py
  15. main.py
  16. tests/test_phase1_ingestion.py

Week 2 — Query Logging (SU6 + SU9)
  17. logging_/normalizer.py
  18. logging_/query_logger.py     (+ update_cluster_centroid, SU6)
  19. middleware/query_interceptor.py  (+ doc_ prefix filter, SU9)
  20. tests/test_phase2_logging.py

Week 3 — Synthesis Engine (SU1 + SU2)
  21. synthesis/prompts.py
  22. synthesis/chunk_fetcher.py   (fetch_and_deduplicate, SU1)
  23. validation/entity_extractor.py  (retained for hallucination check)
  24. validation/validator.py      (LLM-as-judge, SU2)
  25. synthesis/synthesizer.py
  26. db/super_node_store.py
  27. tests/test_phase4_synthesis.py

Week 4 — Scheduler + Preferential Retrieval (SU4 + SU5 + SU10)
  28. scheduler/scheduler.py
  29. patterns/finder.py
  30. scheduler/jobs/pattern_finder.py
  31. retrieval/retriever.py       (ε-greedy + masking + SQLite counter, SU4+SU5+SU10)
  32. retrieval/score_booster.py
  33. api/routes/admin.py
  34. tests/test_phase3_scheduler.py
  35. tests/test_phase5_retrieval.py

Week 5 — Maintenance + Benchmarking (SU3 + SU7 + SU8 + SU11 + SU13)
  36. maintenance/staleness_checker.py  (soft-staleness, SU8; receives SU13 upward signals)
  37. maintenance/decay_scorer.py       (half-life + SQLite sync, SU3+SU10)
  38. maintenance/hierarchy_merger.py   (in_hierarchy lock, SU7; ground-truth re-synthesis, SU11)
  39. scheduler/jobs/maintenance_job.py
  40. scripts/benchmark.py
  41. tests/test_phase6_maintenance.py  (+ SU11 raw-ID tests + SU11 depth cap tests)
  42. Dockerfile + docker-compose.yml
```

---

## BENCHMARKING METHODOLOGY

The `benchmark_pairs` table (MedQuAD question/answer pairs) is the ground truth. Every metric below is computed against it — never against real-time user queries.

```python
# scripts/benchmark.py

METRICS = [
    "recall_at_k",          # Was the gold answer's source chunk in top-k?
    "query_latency_p50_ms",
    "query_latency_p95_ms",
    "context_token_count",
    "bleu_score",           # BLEU between generated answer and gold answer
    "rouge_l_score",        # ROUGE-L between generated and gold
    "fact_coverage_score",  # LLM-judge validation score on generated answer (SU2)
    "chroma_collection_size",
    "exploration_rate",     # % queries in ε-greedy exploration mode (SU4)
    "dedup_reduction_ratio", # avg chunks eliminated per synthesis run (SU1)
]

def run_benchmark(n_pairs: int = 200, use_super_nodes: bool = True):
    pairs = sqlite_client.sample_benchmark_pairs(n=n_pairs)
    baseline = measure_batch(pairs, use_super_nodes=False)
    evolved  = measure_batch(pairs, use_super_nodes=True)
    print_comparison_table(baseline, evolved)
    # Expected:
    #   latency:       ↓ ~6x
    #   token_count:   ↓ ~70% (further reduced by SU1 dedup)
    #   recall@5:      ↑ (super_node covers topic more completely)
    #   BLEU/ROUGE:    ↑ (denser context = better generation)
    #   fact_coverage: ↑ (LLM-judge is more accurate than regex, SU2)

def measure_batch(pairs: List[BenchmarkPair], use_super_nodes: bool) -> dict:
    results = []
    for p in pairs:
        t0 = time.perf_counter()
        ret = retriever.retrieve(p.question, use_super_nodes=use_super_nodes)
        latency = (time.perf_counter() - t0) * 1000

        generated = generator.generate(p.question, ret.documents)
        recall = compute_recall_at_k(
            retrieved_doc_ids=[m["document_id"] for m in ret.metadatas],
            gold_doc_id=p.document_id
        )
        bleu = sentence_bleu([p.answer.split()], generated.split())
        rouge = rouge_scorer.score(p.answer, generated)["rougeL"].fmeasure
        coverage = validator.validate(p.answer, generated).coverage_score  # LLM-judge (SU2)

        results.append({
            "latency_ms": latency,
            "token_count": count_tokens(ret.documents),
            "recall_at_k": recall,
            "bleu": bleu,
            "rouge_l": rouge,
            "fact_coverage": coverage,
        })
    return aggregate_stats(results)
```

---

## RECOMMENDED LIBRARIES (v2)

```
# requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.31.0
chromadb==0.5.20
sentence-transformers==3.3.1
openai==1.55.0
apscheduler==3.10.4
spacy==3.8.3                   # Retained for hallucination additions check
numpy==2.1.3
scikit-learn==1.5.2            # AgglomerativeClustering for SU1 dedup
datasets==3.2.0                # HuggingFace datasets (MedQuAD)
httpx==0.28.0
beautifulsoup4==4.12.3
lxml==5.3.0
tenacity==9.0.0
pydantic==2.10.0
python-multipart==0.0.12
prometheus-fastapi-instrumentator==7.0.0
structlog==24.4.0

# Dev
pytest==8.3.4
pytest-asyncio==0.24.0
pytest-mock==3.14.0
respx==0.21.1
```

---

## COMMON BUGS & PITFALLS (v2)

| Bug | Cause | Fix |
|-----|-------|-----|
| ChromaDB returns 0 results | Collection not created before query | Add `get_or_create_collection()` on startup |
| Embedding dimension mismatch | Switching models mid-run | Store `embedding_model` in collection metadata; assert on load |
| SQLite BLOB deserialization | numpy array not serialized correctly | Always `np.frombuffer(blob, dtype=np.float32)` |
| APScheduler job lost on restart | In-memory scheduler | Use `SQLAlchemyJobStore` for persistence |
| Super-node never retrieved | Score boost too small | Log both raw and boosted scores; verify multiplier effect |
| Validation always passes | LLM judge returns malformed JSON | Catch JSONDecodeError; fallback to entity_extractor; alert |
| Synthesis creates circular meta-nodes | No depth limit | Add `depth` field to metadata; skip merging if `depth >= 2` |
| Double synthesis on restart | `synthesizing=1` stuck | Add cron job: reset `synthesizing=1` rows older than 30min |
| HuggingFace dataset re-downloads on restart | No caching | Set `cache_dir=data/hf_cache` in `load_dataset()` |
| NLM/GHR page structure changes | BS4 selector breaks | Make selector list configurable; fallback to `soup.body` |
| Many MedQuAD rows → same URL crawled N times | No dedup before crawl | Deduplicate on `document_id` before web_fetcher call |
| Crawl banned by NLM | No User-Agent / too fast | Set realistic User-Agent + `CRAWL_RATE_PER_SEC=1` in config |
| `answer` field is None in some rows | MedQuAD has null answers | Filter `row.answer` not None before inserting benchmark_pair |
| Centroid BLOB grows corrupted | Float32 dtype drift | Always `astype(np.float32)` before `.tobytes()` on centroid write (SU6) |
| Super-node IDs leak into chunk_ids | Middleware not filtering | Assert all IDs in cluster.chunk_ids start with `doc_` on insert (SU9) |
| Access counter never increments | SQLite UPDATE targets wrong column | Verify `WHERE super_node_id IN (...)` matches ChromaDB IDs exactly (SU10) |
| All chunks deduped to 1 | distance_threshold too large | Reduce `DEDUP_DISTANCE_THRESHOLD`; minimum 2 chunks output enforced (SU1) |
| Meta-nodes paired with each other | No depth limit | Set `depth` metadata field; filter `depth >= 2` in merger WHERE clause (SU7) |
| Stale node served indefinitely | Re-synthesis queue not drained | Add scheduler job to re-synthesize all `pending_re_synthesis=1` clusters (SU8) |

---

## LOGGING & OBSERVABILITY (v2)

```python
# Structured logging (structlog)
import structlog
log = structlog.get_logger()

# Key log events (with context)
log.info("query_received", query=q, cluster_id=cluster_id)
log.info("centroid_updated", cluster_id=cluster_id, n=hit_count)           # SU6
log.info("ground_truth_filter", raw_ids=len(raw_ids), total_ids=len(all_ids))  # SU9
log.info("synthesis_triggered", cluster_id=cid, hit_count=hits)
log.info("dedup_applied", before=len(chunk_ids), after=len(unique_docs))   # SU1
log.info("super_node_created", sn_id=sn_id, coverage=cov, duration_ms=ms)
log.warning("validation_failed", attempt=n, coverage=cov, missing=missing[:3])
log.info("retrieval_strategy", mode="exploit"|"explore", epsilon=epsilon)  # SU4
log.info("hierarchy_mask_applied", masked_count=n)                         # SU5
log.info("access_logged_sqlite", super_node_ids=ids)                       # SU10
log.info("node_marked_stale", sn_id=sn_id, match_ratio=ratio)             # SU8
log.info("node_pruned", sn_id=sn_id, decay_score=score, days_since_access=d)  # SU3
log.info("meta_node_created", meta_id=mid, children=children)              # SU7

# Prometheus metrics
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)

# Custom metrics
from prometheus_client import Counter, Histogram, Gauge
synthesis_counter     = Counter("synthesis_total", "Synthesis runs", ["status"])
coverage_histogram    = Histogram("validation_coverage", "Fact coverage scores")
super_node_gauge      = Gauge("super_nodes_active", "Active super-nodes")
dedup_removed         = Counter("dedup_chunks_removed", "Chunks eliminated by SU1")
llm_judge_calls       = Counter("llm_judge_total", "LLM-as-judge validation calls")
exploration_counter   = Counter("retrieval_exploration_total", "ε-greedy explore queries")
stale_nodes_flagged   = Counter("stale_nodes_flagged_total", "Soft-stale flags set (SU8)")
merger_lock_counter   = Counter("hierarchy_locks_set_total", "in_hierarchy locks (SU7)")
```

---

## RESEARCH ABLATIONS (v2)

| Ablation | What to Remove | Expected Impact |
|----------|----------------|-----------------|
| No validation engine | Skip coverage check | ↑ synthesis speed, ↓ factual fidelity |
| Regex validator vs LLM-judge | Revert validator.py to entity_extractor | ↓ fact_coverage metric accuracy; paraphrased facts pass undetected |
| No decay scorer | Keep all super-nodes forever | ↑ DB bloat, ↑ stale retrievals over time |
| Linear vs half-life decay | Revert decay formula | Terminal trap: all old nodes pruned regardless of recency |
| No deduplication | Revert chunk_fetcher to direct concat | ↑ token count, ↓ synthesis quality at high query volume |
| No hierarchy merger | No meta-nodes | ↓ scaling, redundant nodes at high volume |
| No score boost | Treat super-nodes as raw chunks | ↓ super-node retrieval rate |
| No ε-greedy | Deterministic exploitation only | Filter bubble: new PDFs never surface; stale super-nodes dominate |
| No centroid update | Lock centroid to first query | Centroid stagnation; hit-rate degrades over long benchmarks |
| No ground-truth anchoring | Log all chunk IDs including sn_* | Generative artifacting after N synthesis cycles |
| No hierarchical masking | Return parent + children | Context window flood; token economy metrics invalidated |
| No SQLite counter | Write access_count to ChromaDB | Concurrency counter loss under load; pruning based on wrong counts |
| No soft staleness | Hard delete on stale super-nodes | Cascading amnesia; metric history wiped by minor upstream edits |
| No in_hierarchy lock | Revert merger to fetch all super_nodes | Exponential meta-node duplication on daily runs |
| Lower hit threshold (5 vs 10) | Trigger synthesis earlier | ↑ node count, ↓ synthesis quality (less signal) |
| Higher cosine threshold (0.95 vs 0.92) | Stricter cluster matching | ↑ cluster fragmentation, ↓ synthesis triggers |

---

## FAILURE RECOVERY FLOW (v2)

```
Query fails → check type:
  LLM timeout → tenacity retry (3x, exponential backoff)
  ChromaDB unavailable → health check → Docker restart policy
  SQLite locked → WAL mode + 500ms retry queue
  Synthesis validation loop → mark synthesis_failed=1, send alert
  Scheduler crash → FastAPI lifespan auto-restarts scheduler
  Embedding OOM → reduce batch_size, log warning
  LLM judge JSON parse error → fallback to entity_extractor; log alert (SU2)

Super-node stale (source chunks shifted) →  [SU8 — CHANGED]
  Mark is_stale=True in ChromaDB (DO NOT DELETE)
  flag_cluster_for_re_synthesis() in SQLite
  Stale node continues serving queries until re-synthesis completes
  Re-synthesis builds new super-node from updated raw chunks
  On success: replace old super-node, clear is_stale flag, reset pending_re_synthesis=0

Super-node hard-invalidated (all source chunks missing, match_ratio = 0.0) →
  Exceptional case: DELETE from ChromaDB (no ground truth remains)
  UPDATE query_clusters SET synthesized=0
  Cluster re-queues naturally on next Pattern Finder scan

Centroid BLOB corrupt (SU6) →
  Catch numpy frombuffer error
  Reset cluster embedding to latest query embedding
  Log alert; mark for manual review

SQLite access counter mismatch (SU10) →
  Detected during decay scorer maintenance pass
  ChromaDB access_count synced from SQLite hit_count
  Log delta for observability

Corrupt metadata →
  ChromaDB get() fails →
  Log + skip that node in maintenance pass
  Flag for manual review via /admin/corrupted-nodes endpoint
```

---

## CROSS-PHASE CONSISTENCY CHECK

### Dependency Validity
| Dependency | Status | Notes |
|------------|--------|-------|
| `chunk_fetcher` → ChromaDB `raw_chunks` (embeddings) | ✅ Valid | SU1 requires `include=["embeddings"]`; ensure collection created with `space="cosine"` |
| `validator` → OpenAI client | ✅ Valid | SU2 reuses existing `client` from generator.py config |
| `decay_scorer` → SQLite `query_clusters.hit_count` | ✅ Valid | SU10 adds `get_cluster_by_super_node_id()` to sqlite_client |
| `retriever` → SQLite `log_super_node_access()` | ✅ Valid | SU10 uses same sqlite_client singleton |
| `query_logger` → SQLite `update_cluster_embedding()` | ✅ Valid | SU6 adds new method to sqlite_client |
| `query_interceptor` → `response_ctx["chunk_ids"]` | ✅ Valid | SU9 filter applied before passing to query_logger |
| `hierarchy_merger` → ChromaDB `where` filter on `in_hierarchy` | ✅ Valid | SU7 requires `in_hierarchy` field present in all super_node metadata at creation (Phase 4) |
| `staleness_checker` → SQLite `flag_cluster_for_re_synthesis()` | ✅ Valid | SU8 adds new method; `pending_re_synthesis` column in schema |
| `hierarchy_merger` → `synthesizer.run()` | ✅ Valid | **SU11**: merger calls same Phase 4 synthesizer; no ad-hoc merge LLM call |
| `synthesizer.run()` → SQLite `flag_cluster_for_re_synthesis(parent_cluster_id)` | ✅ Valid | **SU13**: reuses SU8 queue; `flag_cluster_for_re_synthesis()` already exists |
| `pattern_finder` → SQLite `last_synthesized_hit_count` delta query | ✅ Valid | **SU12**: adds column; `get_ready_clusters()` updated with resynth_delta param |
| `synthesizer.run()` → ChromaDB `super_nodes.get(existing_sn_id)` | ✅ Valid | **SU12**: upsert path reads existing metadata before overwriting |
| `synthesizer.check_fidelity_bound()` → ChromaDB `raw_chunks.get(ids, include=["embeddings"])` | ✅ Valid | **SU14**: source chunk embeddings already in ChromaDB; no additional storage needed |
| `synthesizer.check_fidelity_bound()` → `embedder.encode([job.canonical_query])` | ✅ Valid | **SU14**: shared Embedder singleton; no extra model load |

### No Contradictory Algorithms
| Check | Status |
|-------|--------|
| Single validation function (`validate()` calls `validate_entailment()`) | ✅ No duplicate validation paths |
| Decay formula: only half-life version exists | ✅ Linear formula fully replaced |
| Access counter: SQLite is single source of truth; ChromaDB is read-only mirror | ✅ No conflicting writes |
| Chunk logging: only `doc_*` IDs enter `chunk_ids`; enforced in middleware | ✅ No super-node ID contamination |
| Centroid update: only `update_cluster_centroid()` modifies embeddings | ✅ No competing update paths |
| Merger synthesis: only `synthesizer.run()` is called; no ad-hoc `synthesis_merge()` | ✅ **SU11**: single synthesis code path; no summary-of-summaries fork |

### No Circular Lineage
| Check | Status |
|-------|--------|
| Synthesis always reads from `raw_chunks`, never from `super_nodes` | ✅ Guaranteed by SU9 ground-truth filter |
| Super-node synthesis prompt receives deduplicated raw text, never prior summaries | ✅ `chunk_fetcher.fetch_and_deduplicate()` queries `raw_chunks` collection only |
| Re-synthesis triggered by SU8 follows same path: reads `source_chunks` (raw IDs) | ✅ `source_chunks` metadata contains `doc_*` IDs set at creation time |
| **SU11**: Merger never reads `.documents`; always uses `source_chunks` IDs | ✅ `include=["documents"]` explicitly excluded from merger's ChromaDB get() |
| **SU11**: meta-node `source_chunks` is union of raw IDs, not sn_* IDs | ✅ Defensive `assert all(id_.startswith("doc_"))` in merger before synthesis call |
| **SU12**: Re-synthesis upserts same sn_id; never creates duplicate nodes | ✅ `existing_sn_id` passed from pattern_finder through SynthesisJob |
| **SU13**: Upward propagation reuses SU8's `flag_cluster_for_re_synthesis()`; no new queue | ✅ Same mechanism, additional trigger condition |
| **SU14**: Fidelity check only reads embeddings (already stored); never synthesizes new content | ✅ Pure cosine arithmetic; no LLM call; adds zero hallucination risk |

### Metrics Remain Measurable
| Metric | Status | Notes |
|--------|--------|-------|
| `fact_coverage_score` | ✅ More accurate | LLM-judge returns explicit float; semantic rather than string-match |
| `context_token_count` | ✅ Measurable + improved | SU1 dedup reduces tokens; `dedup_reduction_ratio` tracks savings |
| `recall_at_k` | ✅ Unaffected | Retrieval top-k still returned; hierarchical masking only removes redundant children |
| `super_node_hit_rate` | ✅ Measurable with caveat | Must separate exploitation vs exploration queries using `retrieval_type` label |
| `decay_score` | ✅ Comparable across runs | Half-life formula is deterministic given `last_accessed` and `HALF_LIFE_DAYS` |

### Failure Recovery Consistency
| Scenario | Status |
|----------|--------|
| Synthesis fails → cluster marked `synthesis_failed=1` | ✅ Unchanged from v1 |
| Staleness detected → soft flag, NOT hard delete | ✅ SU8 replaces hard delete; recovery path documented |
| Stale node re-synthesis path | ✅ New path via `pending_re_synthesis` flag + pattern finder scan |
| Corrupt ChromaDB metadata | ✅ Still skipped and flagged; unchanged from v1 |
| SQLite centroid BLOB corrupt | ✅ New recovery: reset to latest query embedding, log alert |

### Logging Compatibility
| Log Event | Status |
|-----------|--------|
| All new SU-specific log events use structlog with named fields | ✅ Consistent with existing structured logging |
| New Prometheus metrics follow existing naming convention (`_total`, `_ms`) | ✅ Compatible with existing Instrumentator setup |
| `query_log.retrieved_chunks` now stores only `doc_*` IDs | ⚠️ **Breaking change for existing log queries** — any dashboard filtering on `retrieved_chunks` that expects mixed IDs must be updated |
| `maintenance_log.event_type` adds new values: `'soft_stale'`, `'re_synthesis_queued'`, `'depth_cap_refused'`, `'manual_review_flagged'` | ✅ Additive; existing consumers of `'prune'` and `'merge'` are unaffected |
| SU12: `revision` counter in ChromaDB metadata is observable in Prometheus via custom gauge | ✅ New metric `super_node_revision_max` (gauge) tracks knowledge volatility |
| SU13: `stale_reason` field in ChromaDB metadata distinguishes source-chunk drift (SU8) from child revision (SU13) | ✅ Additive field; existing `is_stale` filter logic unaffected |
| SU14: `fidelity_flagged` and `drift_margin` in ChromaDB metadata are queryable via `/admin/super-nodes` for human spot-check queue | ✅ Additive fields; no existing query paths changed |

---

> **Blueprint version:** v3.1  
> **Improvements integrated:** SU1–SU14 (all 14)  
> **Breaking changes:** 1 (query_log.retrieved_chunks now raw-only; update any dashboards querying mixed chunk IDs)  
> **New hyperparameters:** EXPLORATION_EPSILON, HALF_LIFE_DAYS, DEDUP_DISTANCE_THRESHOLD, DEDUP_MIN_CHUNKS, SYNTHESIS_MAX_TOKENS, MAX_LINEAGE_DEPTH, RESYNTH_DELTA, MAX_DRIFT_MARGIN, MIN_SOURCE_ANCHOR  
> **New SQLite columns:** `last_synthesized_hit_count` (SU12)  
> **New SQLite tables:** `manual_review_queue` (SU11 depth-cap violations)  
> **New SQLite methods required:** update_cluster_embedding, flag_cluster_for_re_synthesis, get_cluster_by_super_node_id, flag_for_manual_review, get_manual_review_queue  
> **New ChromaDB metadata fields required:** in_hierarchy, is_stale, parent_meta_id, lineage_depth, revision, fidelity_flagged, drift_margin, sim_summary_to_source (all initialized in Phase 4 super-node upsert)  
> **Removed:** `synthesis_merge()` ad-hoc LLM call in hierarchy_merger.py (SU11); `synthesized` boolean as synthesis gate (SU12 replaces with delta counter)  
> **No new files added:** SU14 is entirely within `synthesis/synthesizer.py` — `check_fidelity_bound()` is a new function in that file
