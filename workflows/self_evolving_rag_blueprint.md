# Self-Evolving Write-Back RAG — Engineering Implementation Blueprint

> **Role:** Senior ML Systems Architect  
> **Stack:** Python · FastAPI · ChromaDB · SQLite/PostgreSQL · SentenceTransformers · OpenAI API · APScheduler · Docker  
> **Audience:** Engineer building from scratch, research-grade quality

---

## PHASE 1 — Standard RAG Baseline (Ingestion)

### Goal
Establish the foundational retrieval layer: ingest documents, chunk, embed, store in ChromaDB. Serve queries with top-k retrieval + LLM answer generation.

### Components
```
app/
  ingestion/
    chunker.py          # Semantic + recursive text splitter
    embedder.py         # SentenceTransformer wrapper
    ingestor.py         # Orchestrates chunk → embed → store
  retrieval/
    retriever.py        # Top-k ChromaDB retrieval
  generation/
    generator.py        # LLM answer generation (OpenAI)
  api/
    routes/ingest.py    # POST /ingest
    routes/query.py     # POST /query
  db/
    chroma_client.py    # ChromaDB singleton
  config.py             # All hyperparameters
```

### Inputs
- Raw PDF/DOC files (multipart upload)
- Query string (JSON POST body)

### Outputs
- ChromaDB collection: `raw_chunks` (vectors + metadata)
- JSON answer payload

### Database Schema (ChromaDB Collection: `raw_chunks`)
```
Collection: raw_chunks
  id:          str   → "doc_{source}_{chunk_idx}"
  embedding:   List[float] (384-dim, all-MiniLM-L6-v2)
  document:    str   → raw text of chunk
  metadata:
    source_file:  str
    chunk_id:     int
    type:         str  = "raw_chunk"
    created_at:   ISO8601 timestamp
    token_count:  int
```

### Data Structures
```python
@dataclass
class Chunk:
    chunk_id: str          # "doc_{source}_{idx}"
    text: str
    source_file: str
    token_count: int
    embedding: Optional[List[float]] = None

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
1. Receive PDF → extract text (pypdf2 / pdfplumber)
2. Split text:
   a. Try semantic split (split on sentence boundaries, group to ~500 tokens)
   b. Fallback: RecursiveCharacterTextSplitter(chunk_size=500, overlap=50)
3. For each chunk:
   a. Embed → all-MiniLM-L6-v2 (batch=64 for speed)
   b. Build metadata dict
   c. Upsert to ChromaDB raw_chunks collection
4. Query path:
   a. Embed query
   b. ChromaDB.query(top_k=5)
   c. Concatenate chunk texts as context
   d. Call OpenAI gpt-4o-mini with context + query
   e. Return answer + source_chunks + latency
```

### Pseudocode
```python
# ingestor.py
def ingest_document(file_path: str) -> int:
    text = extract_text(file_path)
    chunks: List[Chunk] = chunker.split(text, source=file_path)
    embeddings = embedder.encode([c.text for c in chunks], batch_size=64)
    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb
    chroma_client.raw_chunks.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=[c.embedding for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[build_metadata(c) for c in chunks]
    )
    return len(chunks)

# retriever.py
def retrieve(query: str, top_k: int = 5) -> List[dict]:
    q_emb = embedder.encode([query])[0]
    results = chroma_client.raw_chunks.query(
        query_embeddings=[q_emb], n_results=top_k
    )
    return results  # ids, documents, metadatas, distances
```

### API Endpoints
```
POST /api/v1/ingest
  Body: multipart/form-data { file: <PDF> }
  Response: { chunks_ingested: int, collection_size: int }

POST /api/v1/query
  Body: { query: str, top_k: int = 5 }
  Response: { answer: str, sources: List[str], latency_ms: float }

GET /api/v1/health
  Response: { status: "ok", chroma_size: int }
```

### Background Jobs
None in Phase 1.

### Error Handling
| Failure | Recovery |
|---------|----------|
| PDF parse error | Return 422, log file path |
| ChromaDB timeout | Retry 3x with exponential backoff |
| Embedding model OOM | Reduce batch size to 16, retry |
| OpenAI rate limit | Retry with jitter (tenacity) |
| Empty chunk after split | Skip, log warning |

### Metrics
- `ingest_latency_ms` (per file)
- `chunks_per_doc` (avg)
- `query_latency_ms` (p50, p95, p99)
- `chroma_collection_size`
- `openai_tokens_used` (per query)

### Unit Tests
```python
# test_phase1.py
def test_chunker_respects_token_limit():
    chunks = chunker.split(LONG_TEXT, source="test.pdf")
    assert all(c.token_count <= 550 for c in chunks)

def test_ingest_round_trip():
    count = ingest_document("fixtures/sample.pdf")
    assert count > 0
    results = retrieve("test query", top_k=3)
    assert len(results["ids"][0]) == 3

def test_query_returns_answer():
    response = client.post("/api/v1/query", json={"query": "What is X?"})
    assert response.status_code == 200
    assert len(response.json()["answer"]) > 10
```

### Future Scalability
- Replace SentenceTransformer CPU with GPU batch inference (torch.cuda)
- Swap ChromaDB for Weaviate/Qdrant when collection > 1M vectors
- Add Redis cache for repeated queries (exact match)

---

**NEXT FILES TO IMPLEMENT (Phase 1):**
1. `config.py` — env vars, hyperparams
2. `db/chroma_client.py` — singleton
3. `ingestion/chunker.py`
4. `ingestion/embedder.py`
5. `ingestion/ingestor.py`
6. `retrieval/retriever.py`
7. `generation/generator.py`
8. `api/routes/ingest.py`
9. `api/routes/query.py`
10. `main.py`

**EXACT CODE DEPENDENCIES:**
```
config.py → (none)
chroma_client.py → config.py
embedder.py → config.py
chunker.py → (none)
ingestor.py → chunker, embedder, chroma_client
retriever.py → embedder, chroma_client
generator.py → config.py (openai key)
routes/ingest.py → ingestor
routes/query.py → retriever, generator
main.py → all routes
```

---

## PHASE 2 — Query Logging and Hashing

### Goal
Intercept every query. Embed it. If cosine sim > 0.92 against an existing cluster in SQLite, increment that cluster's `hit_count` and record which chunks were retrieved. This is the data collection layer that makes self-evolution possible.

### Components
```
app/
  logging/
    query_logger.py     # Core log/upsert logic
    normalizer.py       # Strip → lowercase → embed
  db/
    sqlite_client.py    # SQLite connection + migrations
    models.py           # ORM models
  middleware/
    query_interceptor.py  # FastAPI middleware
```

### Inputs
- Raw query string
- Retrieved chunk IDs (from Phase 1 retriever)

### Outputs
- SQLite row: new query cluster OR incremented hit_count
- `retrieved_chunk_ids` list appended to cluster log

### Database Schema (SQLite)
```sql
CREATE TABLE query_clusters (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_query TEXT NOT NULL,
    query_embedding BLOB NOT NULL,      -- serialized numpy float32 array
    hit_count     INTEGER DEFAULT 1,
    chunk_ids     TEXT NOT NULL,        -- JSON array: ["chunk_1","chunk_48"]
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
    retrieved_chunks TEXT,              -- JSON array
    timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Data Structures
```python
@dataclass
class QueryCluster:
    id: Optional[int]
    canonical_query: str
    query_embedding: np.ndarray    # shape (384,)
    hit_count: int
    chunk_ids: List[str]           # union of all retrieved chunk IDs
    first_seen: datetime
    last_hit: datetime
    synthesized: bool
    super_node_id: Optional[str]

@dataclass
class QueryLogEntry:
    raw_query: str
    cluster_id: int
    retrieved_chunks: List[str]
    timestamp: datetime
```

### Algorithm
```
1. Query arrives at /query endpoint
2. Normalize: strip, lowercase, embed → q_emb (384-dim)
3. Fetch all query_clusters from SQLite (embedding BLOB)
4. Compute cosine similarity: q_emb vs each cluster.query_embedding
5. If max_sim >= 0.92:
   a. cluster = argmax cluster
   b. cluster.hit_count += 1
   c. cluster.chunk_ids = union(cluster.chunk_ids, new_retrieved_ids)
   d. cluster.last_hit = now()
   e. UPDATE query_clusters SET ...
6. Else (new cluster):
   a. INSERT new row: canonical_query=q_normalized, embedding=q_emb
7. INSERT query_log row
```

### Pseudocode
```python
# query_logger.py
def log_query(raw_query: str, retrieved_chunk_ids: List[str]) -> int:
    q_emb = normalizer.embed(raw_query)
    clusters = sqlite_client.fetch_all_clusters()  # List[QueryCluster]

    if clusters:
        sims = cosine_similarity([q_emb], [c.query_embedding for c in clusters])[0]
        best_idx, best_sim = np.argmax(sims), np.max(sims)

        if best_sim >= COSINE_THRESHOLD:  # 0.92
            cluster = clusters[best_idx]
            merged_ids = list(set(cluster.chunk_ids + retrieved_chunk_ids))
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
```

### API Endpoints
No new external endpoints. The logging is wired as FastAPI middleware injected into the existing `/query` route.

```python
# middleware/query_interceptor.py
@app.middleware("http")
async def log_query_middleware(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/api/v1/query":
        body = await request.json()
        # extract retrieved_chunks from response context
        query_logger.log_query(body["query"], response_ctx["chunk_ids"])
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

### ⚠️ Hidden Engineering Problem
**Scalability of in-memory cosine search:** Loading all cluster embeddings into RAM to compute cosine similarity works at <10k clusters. Beyond that, you need a FAISS or Annoy index over cluster embeddings. Build the `sqlite_client.fetch_all_clusters()` interface now so you can swap the backend later without touching `query_logger.py`.

### Metrics
- `clusters_total` (Gauge)
- `cluster_hit_rate` = queries that matched existing cluster / total queries
- `avg_hit_count` across clusters
- `new_clusters_per_hour`
- `sqlite_write_latency_ms`

### Unit Tests
```python
def test_new_query_creates_cluster():
    log_query("what is the penalty for late work", ["c1","c2"])
    clusters = sqlite_client.fetch_all_clusters()
    assert len(clusters) == 1

def test_similar_query_increments_hit():
    log_query("what is the penalty for late work", ["c1","c2"])
    log_query("what happens if I submit tomorrow", ["c1","c3"])
    clusters = sqlite_client.fetch_all_clusters()
    assert len(clusters) == 1
    assert clusters[0].hit_count == 2
    assert "c3" in clusters[0].chunk_ids

def test_dissimilar_query_creates_new_cluster():
    log_query("penalty for late work", ["c1"])
    log_query("what is the refund policy", ["c5"])
    assert len(sqlite_client.fetch_all_clusters()) == 2
```

### Future Scalability
- Swap full-table scan for FAISS IVF index on cluster embeddings (>10k clusters)
- Migrate SQLite → PostgreSQL with pgvector extension
- Add Bloom filter for exact-match deduplication before embedding

---

**NEXT FILES TO IMPLEMENT (Phase 2):**
1. `db/sqlite_client.py`
2. `logging/normalizer.py`
3. `logging/query_logger.py`
4. `middleware/query_interceptor.py`

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
Background job polls SQLite every N minutes. When a cluster's `hit_count >= threshold` AND `synthesized == 0`, extract the canonical query + union chunk IDs and push to Phase 4 synthesis pipeline.

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
    chunk_ids: List[str]        # Union of all co-occurring chunk IDs
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
2. Query: SELECT * FROM query_clusters WHERE hit_count >= 10 AND synthesized = 0
3. For each qualifying cluster:
   a. chunk_ids = JSON.parse(cluster.chunk_ids)  # already a union
   b. Build SynthesisJob
   c. Call synthesizer.run(job)  [Phase 4]
   d. On success: UPDATE query_clusters SET synthesized=1, super_node_id=<id>
4. Log: clusters_triggered, synthesis_duration
```

### Pseudocode
```python
# jobs/pattern_finder.py
def scan_and_trigger():
    ready = sqlite_client.get_ready_clusters(
        min_hit_count=config.HIT_COUNT_THRESHOLD
    )
    for cluster in ready:
        job = SynthesisJob(
            cluster_id=cluster.id,
            canonical_query=cluster.canonical_query,
            chunk_ids=cluster.chunk_ids,
            hit_count=cluster.hit_count,
            triggered_at=datetime.utcnow()
        )
        try:
            super_node_id = synthesizer.run(job)
            sqlite_client.mark_synthesized(cluster.id, super_node_id)
            metrics.increment("synthesis_success")
        except SynthesisError as e:
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
**Double synthesis:** Without `max_instances=1`, two scheduler ticks can both read the same un-synthesized cluster and launch duplicate synthesis jobs. Always set `max_instances=1` and mark cluster as `synthesizing=1` (add column) before kicking off Phase 4, not after.

Add a `synthesizing` column to `query_clusters` as a lock flag:
```sql
ALTER TABLE query_clusters ADD COLUMN synthesizing BOOLEAN DEFAULT 0;
-- Pattern finder sets synthesizing=1 atomically BEFORE calling synthesizer
-- Sets synthesized=1 on success, synthesizing=0 on failure
```

### Metrics
- `pattern_finder_run_duration_ms`
- `clusters_eligible` (per run)
- `synthesis_triggered` (counter)
- `synthesis_success_rate`

### Unit Tests
```python
def test_pattern_finder_triggers_above_threshold():
    insert_cluster(hit_count=10, synthesized=0)
    jobs = pattern_finder.get_ready_clusters(min_hit_count=10)
    assert len(jobs) == 1

def test_pattern_finder_skips_below_threshold():
    insert_cluster(hit_count=5, synthesized=0)
    jobs = pattern_finder.get_ready_clusters(min_hit_count=10)
    assert len(jobs) == 0

def test_pattern_finder_skips_already_synthesized():
    insert_cluster(hit_count=15, synthesized=1)
    jobs = pattern_finder.get_ready_clusters(min_hit_count=10)
    assert len(jobs) == 0

def test_no_double_synthesis(mocker):
    mocker.patch("synthesizer.run", side_effect=lambda j: "sn_1")
    insert_cluster(hit_count=10, synthesized=0)
    scan_and_trigger()
    scan_and_trigger()
    assert synthesizer.run.call_count == 1
```

---

**NEXT FILES TO IMPLEMENT (Phase 3):**
1. `scheduler/scheduler.py`
2. `patterns/finder.py`
3. `scheduler/jobs/pattern_finder.py`

**EXACT CODE DEPENDENCIES:**
```
scheduler.py → config.py, apscheduler
finder.py → sqlite_client.py
pattern_finder.py (job) → finder.py, synthesizer.py (Phase 4)
main.py → scheduler.py (start on lifespan)
```

---

## PHASE 4 — Summarizer Agent & Validation Engine

### Goal
Fetch raw chunks, run LLM synthesis with strict prompt, validate factual fidelity (≥90% coverage), upsert Super-Node to ChromaDB. The core intelligence of the system.

### Components
```
app/
  synthesis/
    chunk_fetcher.py     # Fetch chunk text by IDs from ChromaDB
    synthesizer.py       # LLM synthesis orchestrator
    prompts.py           # System + user prompt templates
  validation/
    validator.py         # Fact coverage engine
    entity_extractor.py  # Regex + spaCy NER
  db/
    super_node_store.py  # ChromaDB upsert for super-nodes
```

### Inputs
- `SynthesisJob`: canonical_query, chunk_ids, hit_count

### Outputs
- ChromaDB upsert: Super-Node document with full metadata
- Returns: `super_node_id: str` OR raises `SynthesisError`

### Database Schema (ChromaDB Collection: `super_nodes`)
```
Collection: super_nodes
  id:          "sn_{cluster_id}_{timestamp}"
  embedding:   List[float] (384-dim)
  document:    str  → compressed synthesis text
  metadata:
    type:           "super_node"
    source_query:   str   → canonical query
    source_chunks:  str   → JSON list of chunk IDs
    hit_count:      int
    created_at:     ISO8601
    last_accessed:  ISO8601
    access_count:   int = 0
    decay_score:    float = 1.0
    cluster_id:     int
    fact_coverage:  float  → validation score
```

### Data Structures
```python
@dataclass
class SuperNode:
    id: str
    text: str
    embedding: List[float]
    source_query: str
    source_chunks: List[str]
    hit_count: int
    created_at: datetime
    last_accessed: datetime
    access_count: int
    decay_score: float
    cluster_id: int
    fact_coverage: float

@dataclass
class ValidationResult:
    passed: bool
    coverage_score: float          # facts_in_summary / facts_in_source
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
1. Fetch raw texts for all chunk_ids from ChromaDB
2. Concatenate into source_material string
3. Build synthesis prompt (canonical_query + source_material)
4. Call LLM (gpt-4o-mini, temperature=0.1, max_tokens=1000)
5. Extract candidate summary from response
6. VALIDATION:
   a. Extract facts from source_material → Set[source_facts]
      - Regex: numbers, dates, percentages, currency
      - spaCy NER: PERSON, ORG, DATE, CARDINAL, LAW
   b. Extract facts from summary → Set[summary_facts]
   c. coverage = |summary_facts ∩ source_facts| / |source_facts|
   d. IF coverage < 0.90:
      - Retry with re-synthesis prompt (max 3 retries)
      - If still failing: ABORT, mark cluster as synthesis_failed
7. Embed summary → 384-dim vector
8. Upsert to ChromaDB super_nodes collection
9. Return super_node_id
```

### Synthesis Prompt (Exact)
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
    # Step 1: Fetch chunks
    chunks = chunk_fetcher.fetch(job.chunk_ids)
    if not chunks:
        raise SynthesisError(f"No chunks found for IDs: {job.chunk_ids}")
    source_text = "\n\n---\n\n".join(c["document"] for c in chunks)

    # Step 2: Synthesize with retry
    summary, attempt = None, 0
    missing_facts = []

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

        # Step 3: Validate
        result = validator.validate(source_text, summary)
        if result.passed:
            break

        missing_facts = result.missing_facts
        attempt += 1
        log.warning(f"Validation failed (attempt {attempt}): coverage={result.coverage_score:.2f}")

    if not result.passed:
        raise ValidationError(f"Max retries exceeded. Final coverage: {result.coverage_score:.2f}")

    # Step 4: Embed + Upsert
    embedding = embedder.encode([summary])[0]
    sn_id = f"sn_{job.cluster_id}_{int(datetime.utcnow().timestamp())}"

    chroma_client.super_nodes.upsert(
        ids=[sn_id],
        embeddings=[embedding.tolist()],
        documents=[summary],
        metadatas=[{
            "type": "super_node",
            "source_query": job.canonical_query,
            "source_chunks": json.dumps(job.chunk_ids),
            "hit_count": job.hit_count,
            "created_at": datetime.utcnow().isoformat(),
            "last_accessed": datetime.utcnow().isoformat(),
            "access_count": 0,
            "decay_score": 1.0,
            "cluster_id": job.cluster_id,
            "fact_coverage": result.coverage_score
        }]
    )
    return sn_id

# validator.py
def validate(source_text: str, summary: str) -> ValidationResult:
    source_facts = extract_facts(source_text)
    summary_facts = extract_facts(summary)

    if not source_facts:
        return ValidationResult(passed=True, coverage_score=1.0, ...)

    matched = source_facts & summary_facts
    coverage = len(matched) / len(source_facts)
    missing = list(source_facts - summary_facts)

    return ValidationResult(
        passed=(coverage >= 0.90),
        coverage_score=coverage,
        missing_facts=missing[:10],  # top 10 for re-synthesis prompt
        source_fact_count=len(source_facts),
        summary_fact_count=len(summary_facts)
    )

# entity_extractor.py
def extract_facts(text: str) -> Set[str]:
    facts = set()
    # Regex: numbers, percentages, dates, currencies
    facts.update(re.findall(r'\b\d+\.?\d*%?\b', text))
    facts.update(re.findall(r'\$[\d,]+', text))
    facts.update(re.findall(r'\b\d{4}-\d{2}-\d{2}\b', text))
    # spaCy NER
    doc = nlp(text)
    facts.update(ent.text.lower() for ent in doc.ents
                 if ent.label_ in ("PERSON","ORG","DATE","CARDINAL","LAW","GPE"))
    return facts
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
| spaCy model not loaded | Fallback to regex-only extraction (log warning) |
| Empty source chunks | Abort synthesis, log error |

### ⚠️ Hidden Engineering Problems
1. **spaCy model download:** `en_core_web_sm` must be downloaded in Dockerfile. Add: `RUN python -m spacy download en_core_web_sm`
2. **Regex over-matching:** Number extraction will match page numbers, footnotes. Normalize by stripping numbers under 3 in common positions. 
3. **LLM hallucination in summary:** The validator catches drops but not additions. Consider adding a "no new facts" check: any entity in summary NOT in source_facts is an injected hallucination.
4. **Token budget:** If source_chunks > 3000 tokens, the prompt + completion exceeds gpt-4o-mini context. Truncate intelligently — prioritize chunks with highest co-occurrence frequency.

### Metrics
- `synthesis_duration_ms` (histogram)
- `synthesis_retries` (histogram)
- `validation_coverage_score` (histogram)
- `synthesis_abort_rate`
- `super_node_count` (gauge)
- `avg_fact_coverage`

### Unit Tests
```python
def test_synthesizer_creates_super_node():
    job = SynthesisJob(cluster_id=1, canonical_query="...", chunk_ids=["c1","c2"], hit_count=10)
    sn_id = synthesizer.run(job)
    assert sn_id.startswith("sn_1_")
    result = chroma_client.super_nodes.get(ids=[sn_id])
    assert result["metadatas"][0]["type"] == "super_node"

def test_validator_rejects_low_coverage():
    result = validator.validate(
        source_text="The fee is $500. Deadline is 2024-01-15.",
        summary="The deadline is approaching."   # missing $500, date
    )
    assert not result.passed
    assert result.coverage_score < 0.90

def test_validator_accepts_high_coverage():
    source = "The fee is $500. Deadline is 2024-01-15. Contact John Smith."
    summary = "Fee: $500, due 2024-01-15, contact John Smith."
    result = validator.validate(source, summary)
    assert result.passed

def test_synthesis_aborts_after_max_retries(mocker):
    mocker.patch("validator.validate", return_value=ValidationResult(passed=False, ...))
    with pytest.raises(ValidationError):
        synthesizer.run(job)
```

---

**NEXT FILES TO IMPLEMENT (Phase 4):**
1. `validation/entity_extractor.py`
2. `validation/validator.py`
3. `synthesis/prompts.py`
4. `synthesis/chunk_fetcher.py`
5. `synthesis/synthesizer.py`
6. `db/super_node_store.py`

---

## PHASE 5 — Preferential Retrieval

### Goal
Modify the retrieval router to query BOTH `raw_chunks` and `super_nodes` collections. Apply score boost to super_nodes. Return the best result. Fall back gracefully if no super_node exists.

### Components
```
app/
  retrieval/
    retriever.py          # Updated: dual-collection search
    score_booster.py      # Score boost logic
    router.py             # Route to super_node or raw_chunk
```

### Algorithm
```
1. Embed query → q_emb
2. Query super_nodes collection (top_k=3)
3. Query raw_chunks collection (top_k=5)
4. For each super_node result:
   a. Apply score boost: boosted_distance = L2_distance * 0.85
   b. This artificially raises super_node priority
5. Merge + re-rank by boosted distance (ascending)
6. Return top_k documents from merged list
7. Update super_node metadata: access_count++, last_accessed=now()
```

### Pseudocode
```python
# retriever.py (updated)
def retrieve(query: str, top_k: int = 5) -> RetrievalResult:
    q_emb = embedder.encode([query])[0]

    # Dual collection search
    sn_results = chroma_client.super_nodes.query(
        query_embeddings=[q_emb], n_results=3,
        include=["documents","metadatas","distances"]
    )
    raw_results = chroma_client.raw_chunks.query(
        query_embeddings=[q_emb], n_results=top_k,
        include=["documents","metadatas","distances"]
    )

    candidates = []

    for i, (doc, meta, dist) in enumerate(zip(
        sn_results["documents"][0],
        sn_results["metadatas"][0],
        sn_results["distances"][0]
    )):
        boosted = dist * config.SUPER_NODE_SCORE_MULTIPLIER  # 0.85
        candidates.append({"doc": doc, "meta": meta, "score": boosted, "type": "super_node"})
        # Update access metadata async
        asyncio.create_task(update_access(sn_results["ids"][0][i]))

    for doc, meta, dist in zip(
        raw_results["documents"][0],
        raw_results["metadatas"][0],
        raw_results["distances"][0]
    ):
        candidates.append({"doc": doc, "meta": meta, "score": dist, "type": "raw_chunk"})

    # Sort ascending (lower L2 = better)
    candidates.sort(key=lambda x: x["score"])
    top = candidates[:top_k]

    return RetrievalResult(
        documents=[c["doc"] for c in top],
        types=[c["type"] for c in top],
        scores=[c["score"] for c in top]
    )
```

### ⚠️ Hidden Engineering Problem
**Score boost interaction with L2 vs cosine distance:** ChromaDB returns L2 distance by default. Multiplying by 0.85 works for L2 but breaks for cosine (where distance is already 0–2). Explicitly set `space="cosine"` when creating collections and verify distance semantics before applying multiplier.

### Metrics
- `retrieval_type_distribution` — super_node vs raw_chunk (per query)
- `super_node_hit_rate` — % of queries where a super_node was top result
- `query_latency_ms` — compare Phase 1 vs Phase 5 (target: <200ms)
- `context_token_reduction` — avg tokens Phase 1 vs Phase 5

---

## PHASE 6 — Maintenance, Decay & Hierarchy

### Goal
Prevent database bloat and semantic drift. Three sub-systems: (1) Staleness Check, (2) Decay Scorer, (3) Hierarchy Merger.

### Components
```
app/
  maintenance/
    staleness_checker.py   # Invalidate super-nodes if source chunks updated
    decay_scorer.py        # Score = access_count / (days_since_created + 1)
    hierarchy_merger.py    # Merge similar super-nodes → meta-nodes
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
    access_count: int
    days_since_created: int
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

#### Staleness Check
```
1. For each super_node in ChromaDB:
   a. source_chunks = JSON.parse(meta["source_chunks"])
   b. For each chunk_id in source_chunks:
      - Query raw_chunks for chunk_id
      - If not found → source was deleted → INVALIDATE super_node
      - If found but updated_at > super_node.created_at → INVALIDATE
2. Invalidation: DELETE from super_nodes collection
3. Reset cluster: UPDATE query_clusters SET synthesized=0 WHERE super_node_id=...
   (allows re-synthesis with updated source chunks)
```

#### Decay Scorer
```
Formula: score = access_count / (days_since_created + 1)
         Penalize if source_chunk updated: score *= 0.5

1. For each super_node:
   a. Compute decay_score
   b. UPDATE metadata decay_score field
   c. If decay_score < PRUNE_THRESHOLD (0.05):
      - DELETE from ChromaDB
      - Log pruning event
   d. If decay_score > HIGH_SCORE_THRESHOLD (5.0):
      - Flag for re-synthesis (may need fresh content)
```

#### Hierarchy Merger
```
1. Fetch all super_node embeddings
2. Compute pairwise cosine similarity matrix
3. Find pairs with sim > 0.88
4. For each qualifying pair (sn_A, sn_B):
   a. Merge texts → run synthesis (compact prompt)
   b. Embed merged text → meta_node embedding
   c. Upsert meta_node to ChromaDB (type="meta_node")
   d. Keep sn_A, sn_B as children (metadata: parent_meta_node_id)
5. Retrieval order: meta_node → super_node → raw_chunk
```

### Pseudocode
```python
# decay_scorer.py
def compute_and_apply_decay():
    all_sn = chroma_client.super_nodes.get(
        where={"type": {"$eq": "super_node"}},
        include=["metadatas", "ids"]
    )
    for sn_id, meta in zip(all_sn["ids"], all_sn["metadatas"]):
        created = datetime.fromisoformat(meta["created_at"])
        days = (datetime.utcnow() - created).days
        score = meta["access_count"] / (days + 1)
        chroma_client.super_nodes.update(ids=[sn_id], metadatas=[{**meta, "decay_score": score}])
        if score < config.PRUNE_THRESHOLD:
            chroma_client.super_nodes.delete(ids=[sn_id])
            log.info(f"Pruned super_node {sn_id} (score={score:.3f})")

# hierarchy_merger.py
def merge_similar_nodes():
    all_sn = chroma_client.super_nodes.get(
        where={"type": {"$eq": "super_node"}},
        include=["metadatas","embeddings","documents","ids"]
    )
    if len(all_sn["ids"]) < 2:
        return

    embs = np.array(all_sn["embeddings"])
    sim_matrix = cosine_similarity(embs)
    np.fill_diagonal(sim_matrix, 0)
    pairs = np.argwhere(sim_matrix > config.MERGE_THRESHOLD)  # 0.88
    pairs = [(a,b) for a,b in pairs if a < b]   # deduplicate

    for i, j in pairs:
        merged_text = synthesis_merge(
            all_sn["documents"][i], all_sn["documents"][j]
        )
        meta_id = f"mn_{int(datetime.utcnow().timestamp())}"
        emb = embedder.encode([merged_text])[0]
        chroma_client.super_nodes.upsert(
            ids=[meta_id], embeddings=[emb.tolist()],
            documents=[merged_text],
            metadatas=[{"type": "meta_node",
                        "children": json.dumps([all_sn["ids"][i], all_sn["ids"][j]]),
                        "created_at": datetime.utcnow().isoformat()}]
        )
```

### Background Jobs
| Job | Trigger | Interval |
|-----|---------|----------|
| `staleness_check` | Interval | 1 hour |
| `decay_scorer` | Interval | 6 hours |
| `hierarchy_merger` | Interval | 24 hours |

### ⚠️ Hidden Engineering Problems
1. **Pairwise cosine on 10k super-nodes = 10^8 operations.** Cap hierarchy merger at 500 most-accessed super-nodes per run, or use FAISS to find approximate neighbors.
2. **Cascade delete:** When raw_chunk is updated and super_node is invalidated, the cluster's `synthesized=0` must be reset atomically. Use a SQLite transaction.
3. **Meta-node creation loop:** If two meta-nodes are themselves similar, you'll create meta-meta-nodes. Cap hierarchy at 2 levels deep.

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
│   │       ├── ingest.py              # POST /ingest
│   │       ├── query.py               # POST /query
│   │       └── admin.py              # Admin endpoints
│   │
│   ├── db/
│   │   ├── chroma_client.py           # ChromaDB singleton
│   │   ├── sqlite_client.py           # SQLite connection + queries
│   │   └── migrations/
│   │       └── 001_initial.sql
│   │
│   ├── ingestion/
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   └── ingestor.py
│   │
│   ├── retrieval/
│   │   ├── retriever.py
│   │   └── score_booster.py
│   │
│   ├── generation/
│   │   └── generator.py
│   │
│   ├── logging_/
│   │   ├── query_logger.py
│   │   └── normalizer.py
│   │
│   ├── middleware/
│   │   └── query_interceptor.py
│   │
│   ├── patterns/
│   │   └── finder.py
│   │
│   ├── synthesis/
│   │   ├── chunk_fetcher.py
│   │   ├── synthesizer.py
│   │   └── prompts.py
│   │
│   ├── validation/
│   │   ├── validator.py
│   │   └── entity_extractor.py
│   │
│   ├── maintenance/
│   │   ├── staleness_checker.py
│   │   ├── decay_scorer.py
│   │   └── hierarchy_merger.py
│   │
│   └── scheduler/
│       ├── scheduler.py
│       └── jobs/
│           ├── pattern_finder.py
│           └── maintenance_job.py
│
├── tests/
│   ├── conftest.py
│   ├── test_phase1_ingestion.py
│   ├── test_phase2_logging.py
│   ├── test_phase3_scheduler.py
│   ├── test_phase4_synthesis.py
│   ├── test_phase5_retrieval.py
│   └── test_phase6_maintenance.py
│
├── scripts/
│   ├── benchmark.py                   # Benchmarking suite
│   └── seed_data.py                   # Load test documents
│
└── data/
    ├── chroma/                        # ChromaDB persistence
    └── sqlite/
        └── query_log.db
```

---

## DATABASE SCHEMAS (Complete SQL)

```sql
-- query_clusters: core pattern tracking table
CREATE TABLE IF NOT EXISTS query_clusters (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_query  TEXT NOT NULL,
    query_embedding  BLOB NOT NULL,
    hit_count        INTEGER NOT NULL DEFAULT 1,
    chunk_ids        TEXT NOT NULL DEFAULT '[]',
    first_seen       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_hit         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    synthesized      BOOLEAN NOT NULL DEFAULT 0,
    synthesizing     BOOLEAN NOT NULL DEFAULT 0,
    synthesis_failed BOOLEAN NOT NULL DEFAULT 0,
    super_node_id    TEXT,
    retry_count      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_hit_count_synth
    ON query_clusters(hit_count DESC, synthesized, synthesizing);

-- query_log: raw query audit trail
CREATE TABLE IF NOT EXISTS query_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_query        TEXT NOT NULL,
    normalized_query TEXT,
    cluster_id       INTEGER REFERENCES query_clusters(id),
    retrieved_chunks TEXT DEFAULT '[]',
    retrieval_type   TEXT,  -- 'super_node' | 'raw_chunk' | 'mixed'
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
    event_type       TEXT NOT NULL,  -- 'prune'|'invalidate'|'merge'
    super_node_id    TEXT,
    reason           TEXT,
    decay_score      REAL,
    timestamp        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

---

## HYPERPARAMETERS

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `CHUNK_SIZE_TOKENS` | 500 | Balances context density vs retrieval precision |
| `CHUNK_OVERLAP_TOKENS` | 50 | Prevents boundary information loss |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | 384-dim, fast, good semantic quality |
| `EMBEDDING_BATCH_SIZE` | 64 | GPU-efficient batch size |
| `QUERY_CLUSTER_COSINE_THRESHOLD` | 0.92 | High threshold to avoid over-merging |
| `HIT_COUNT_THRESHOLD` | 10 | Min hits before synthesis trigger |
| `POLL_INTERVAL_MINUTES` | 5 | Scheduler poll frequency |
| `SUPER_NODE_SCORE_MULTIPLIER` | 0.85 | L2 distance boost for super-nodes |
| `VALIDATION_COVERAGE_THRESHOLD` | 0.90 | Minimum fact coverage to accept synthesis |
| `MAX_SYNTHESIS_RETRIES` | 3 | LLM synthesis retry limit |
| `DECAY_PRUNE_THRESHOLD` | 0.05 | Score below which node is pruned |
| `HIERARCHY_MERGE_THRESHOLD` | 0.88 | Cosine sim above which nodes merge |
| `LLM_MODEL` | `gpt-4o-mini` | Cost-efficient, sufficient quality |
| `LLM_TEMPERATURE` | 0.1 | Low temp for factual consistency |
| `TOP_K_RAW` | 5 | Raw chunks retrieved per query |
| `TOP_K_SUPER` | 3 | Super-nodes retrieved per query |

---

## BUILD ORDER (File-by-File)

```
Week 1 — Foundation
  1. config.py
  2. db/chroma_client.py
  3. ingestion/embedder.py
  4. ingestion/chunker.py
  5. ingestion/ingestor.py
  6. retrieval/retriever.py
  7. generation/generator.py
  8. api/routes/ingest.py
  9. api/routes/query.py
  10. main.py
  11. tests/test_phase1_ingestion.py

Week 2 — Query Logging
  12. db/sqlite_client.py + migrations/001_initial.sql
  13. logging_/normalizer.py
  14. logging_/query_logger.py
  15. middleware/query_interceptor.py
  16. tests/test_phase2_logging.py

Week 3 — Synthesis Engine
  17. synthesis/prompts.py
  18. synthesis/chunk_fetcher.py
  19. validation/entity_extractor.py
  20. validation/validator.py
  21. synthesis/synthesizer.py
  22. db/super_node_store.py
  23. tests/test_phase4_synthesis.py

Week 4 — Scheduler + Preferential Retrieval
  24. scheduler/scheduler.py
  25. patterns/finder.py
  26. scheduler/jobs/pattern_finder.py
  27. retrieval/score_booster.py (update retriever.py)
  28. api/routes/admin.py
  29. tests/test_phase3_scheduler.py
  30. tests/test_phase5_retrieval.py

Week 5 — Maintenance + Benchmarking
  31. maintenance/staleness_checker.py
  32. maintenance/decay_scorer.py
  33. maintenance/hierarchy_merger.py
  34. scheduler/jobs/maintenance_job.py
  35. scripts/benchmark.py
  36. tests/test_phase6_maintenance.py
  37. Dockerfile + docker-compose.yml
```

---

## BENCHMARKING METHODOLOGY

```python
# scripts/benchmark.py

METRICS = [
    "query_latency_p50_ms",
    "query_latency_p95_ms",
    "context_token_count",
    "fact_coverage_score",
    "chroma_collection_size",
]

def run_benchmark(n_queries: int = 100):
    """
    Phase A: Baseline (Phase 1 only, no super-nodes)
    Phase B: Self-evolving (Phase 5, with super-nodes)

    Compare same queries across both phases.
    Track token usage: count context window tokens per query.
    """
    test_queries = load_test_queries("data/benchmark_queries.json")
    baseline = measure_batch(test_queries, use_super_nodes=False)
    evolved = measure_batch(test_queries, use_super_nodes=True)

    print_comparison_table(baseline, evolved)
    # Expected: latency ↓ 6x, token_count ↓ 70%

def measure_batch(queries, use_super_nodes: bool) -> dict:
    results = []
    for q in queries:
        t0 = time.perf_counter()
        result = retriever.retrieve(q, use_super_nodes=use_super_nodes)
        latency = (time.perf_counter() - t0) * 1000
        results.append({
            "latency_ms": latency,
            "token_count": count_tokens(result.documents),
            "type": result.types[0]
        })
    return aggregate_stats(results)
```

---

## RECOMMENDED LIBRARIES

```
# requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.31.0
chromadb==0.5.20
sentence-transformers==3.3.1
openai==1.55.0
apscheduler==3.10.4
spacy==3.8.3
numpy==2.1.3
scikit-learn==1.5.2
pdfplumber==0.11.4
pypdf==5.1.0
tenacity==9.0.0
pydantic==2.10.0
python-multipart==0.0.12
prometheus-fastapi-instrumentator==7.0.0
structlog==24.4.0

# Dev
pytest==8.3.4
pytest-asyncio==0.24.0
pytest-mock==3.14.0
httpx==0.28.0
```

---

## COMMON BUGS & PITFALLS

| Bug | Cause | Fix |
|-----|-------|-----|
| ChromaDB returns 0 results | Collection not created before query | Add `get_or_create_collection()` on startup |
| Embedding dimension mismatch | Switching models mid-run | Store `embedding_model` in collection metadata; assert on load |
| SQLite BLOB deserialization | numpy array not serialized correctly | Always `np.frombuffer(blob, dtype=np.float32)` |
| APScheduler job lost on restart | In-memory scheduler | Use `SQLAlchemyJobStore` for persistence |
| Super-node never retrieved | Score boost too small | Log both raw and boosted scores; verify multiplier effect |
| Validation always fails | NER model not matching | Test `entity_extractor` on known text first |
| Synthesis creates circular meta-nodes | No depth limit | Add `depth` field to metadata; skip merging if `depth >= 2` |
| Double synthesis on restart | `synthesizing=1` stuck | Add cron job: reset `synthesizing=1` rows older than 30min |

---

## LOGGING & OBSERVABILITY

```python
# Structured logging (structlog)
import structlog
log = structlog.get_logger()

# Key log events (with context)
log.info("query_received", query=q, cluster_id=cluster_id)
log.info("synthesis_triggered", cluster_id=cid, hit_count=hits)
log.info("super_node_created", sn_id=sn_id, coverage=cov, duration_ms=ms)
log.warning("validation_failed", attempt=n, coverage=cov, missing=missing[:3])
log.info("node_pruned", sn_id=sn_id, decay_score=score)

# Prometheus metrics
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)

# Custom metrics
from prometheus_client import Counter, Histogram, Gauge
synthesis_counter = Counter("synthesis_total", "Synthesis runs", ["status"])
coverage_histogram = Histogram("validation_coverage", "Fact coverage scores")
super_node_gauge = Gauge("super_nodes_active", "Active super-nodes")
```

---

## RESEARCH ABLATIONS

| Ablation | What to Remove | Expected Impact |
|----------|----------------|-----------------|
| No validation engine | Skip coverage check | ↑ synthesis speed, ↓ factual fidelity |
| No decay scorer | Keep all super-nodes forever | ↑ DB bloat, ↑ stale retrievals over time |
| No hierarchy merger | No meta-nodes | ↓ scaling, redundant nodes at high volume |
| No score boost | Treat super-nodes as raw chunks | ↓ super-node retrieval rate |
| Lower hit threshold (5 vs 10) | Trigger synthesis earlier | ↑ node count, ↓ synthesis quality (less signal) |
| Higher cosine threshold (0.95 vs 0.92) | Stricter cluster matching | ↑ cluster fragmentation, ↓ synthesis triggers |

---

## FAILURE RECOVERY FLOW

```
Query fails → check type:
  LLM timeout → tenacity retry (3x, exponential backoff)
  ChromaDB unavailable → health check → Docker restart policy
  SQLite locked → WAL mode + 500ms retry queue
  Synthesis validation loop → mark synthesis_failed=1, send alert
  Scheduler crash → FastAPI lifespan auto-restarts scheduler
  Embedding OOM → reduce batch_size, log warning

Super-node invalidated (staleness) →
  DELETE from ChromaDB
  UPDATE query_clusters SET synthesized=0
  Cluster re-queues naturally on next Pattern Finder scan

Corrupt metadata →
  ChromaDB get() fails →
  Log + skip that node in maintenance pass
  Flag for manual review via /admin/corrupted-nodes endpoint
```

---

## TODO LIST (Ordered by Dependencies)

```
[ ] 1. config.py — env var loading (no deps)
[ ] 2. db/chroma_client.py — singleton (dep: config)
[ ] 3. ingestion/embedder.py (dep: config)
[ ] 4. ingestion/chunker.py (dep: none)
[ ] 5. ingestion/ingestor.py (dep: 2,3,4)
[ ] 6. retrieval/retriever.py (dep: 2,3)
[ ] 7. generation/generator.py (dep: config)
[ ] 8. api/routes/ingest.py (dep: 5)
[ ] 9. api/routes/query.py (dep: 6,7)
[ ] 10. main.py (dep: 8,9)
[ ] 11. db/sqlite_client.py (dep: config)
[ ] 12. logging_/normalizer.py (dep: 3)
[ ] 13. logging_/query_logger.py (dep: 11,12)
[ ] 14. middleware/query_interceptor.py (dep: 13)
[ ] 15. synthesis/prompts.py (dep: none)
[ ] 16. synthesis/chunk_fetcher.py (dep: 2)
[ ] 17. validation/entity_extractor.py (dep: spacy)
[ ] 18. validation/validator.py (dep: 17)
[ ] 19. synthesis/synthesizer.py (dep: 3,15,16,18)
[ ] 20. scheduler/scheduler.py (dep: apscheduler)
[ ] 21. patterns/finder.py (dep: 11)
[ ] 22. scheduler/jobs/pattern_finder.py (dep: 19,21)
[ ] 23. retrieval/score_booster.py (dep: 6)
[ ] 24. maintenance/staleness_checker.py (dep: 2,11)
[ ] 25. maintenance/decay_scorer.py (dep: 2)
[ ] 26. maintenance/hierarchy_merger.py (dep: 2,19)
[ ] 27. scheduler/jobs/maintenance_job.py (dep: 24,25,26)
[ ] 28. api/routes/admin.py (dep: 20,21,22)
[ ] 29. scripts/benchmark.py (dep: all)
[ ] 30. Dockerfile + docker-compose.yml
[ ] 31. All tests (dep: each module they test)
```

##IMPROVEMENTS
If your primary goal is to extract empirical results and validate the architecture for a research paper, we can entirely eliminate the infrastructure headaches. You won't need to worry about Async Event Loop Blocking or Multi-Pod Race Conditions because you will likely run controlled benchmark scripts locally or on a single dedicated machine.

However, the remaining two flaws will actively corrupt your research data and invalidate your findings. Here is the filtered blueprint for the two research-critical fixes you must implement.

---

## 1. Fixing the Token Window Collapse (Data Bloat)

**The Research Risk:** If you run an automated benchmark simulating 50 queries of the same intent to test the Pattern Finder, your `chunk_ids` union will grow massively. Feeding 40 redundant chunks into `gpt-4o-mini` will trigger the "lost in the middle" effect, degrading synthesis quality and artificially skewing your token-economy benchmarks.

**The Fix:** Implement Semantic Deduplication in Phase 4 before sending the prompt to the LLM.

### Updated Component: `synthesis/chunk_fetcher.py`

Instead of blindly returning all texts, embed the fetched chunks, cluster them to find unique semantic concepts, and return only the top representative chunk from each cluster.

```python
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from typing import List, Dict

def fetch_and_deduplicate(chunk_ids: List[str], max_tokens: int = 4000) -> str:
    # 1. Fetch raw chunks from ChromaDB
    raw_results = chroma_client.raw_chunks.get(
        ids=chunk_ids,
        include=["documents", "embeddings"]
    )
    
    docs = raw_results["documents"]
    embs = np.array(raw_results["embeddings"])
    
    # If small enough, skip clustering
    if len(docs) <= 5:
        return "\n\n---\n\n".join(docs)
        
    # 2. Cluster chunks to find redundancies (distance threshold = 0.15)
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=0.15, 
        metric="cosine", 
        linkage="average"
    )
    labels = clustering.fit_predict(embs)
    
    # 3. Pick one representative document per cluster
    unique_docs = []
    seen_labels = set()
    for i, label in enumerate(labels):
        if label not in seen_labels:
            unique_docs.append(docs[i])
            seen_labels.add(label)
            
    # 4. Enforce token limits (truncate if still too large)
    # Assume ~4 chars per token roughly for truncation
    final_text = "\n\n---\n\n".join(unique_docs)
    max_chars = max_tokens * 4
    return final_text[:max_chars]

```

---

## 2. Fixing Rigid Fact Validation (Accuracy Flaw)

**The Research Risk:** Your paper claims that the Validation Engine guarantees $\ge90\%$ factual fidelity. If you use simple Regex and spaCy NER, the system will pass a summary that hallucinates connections, as long as the numbers and dates match. This invalidates your "Fact Coverage" metric in your benchmark table.

**The Fix:** Upgrade `validator.py` to use a fast, low-cost Cross-Encoder (NLI model) or a structured LLM-as-a-judge call to verify semantic entailment, rather than raw string extraction.

### Updated Component: `validation/validator.py`

Using an LLM to evaluate the coverage mathematically ensures nuanced facts are captured.

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
{"passed": true/false (if >= 0.90), "coverage_score": float, "missing_facts": [list of strings]}

Source Material:
{source_text}

Summary:
{summary_text}
"""

def validate_entailment(source_text: str, summary: str) -> ValidationOutput:
    # Use gpt-4o-mini in strict JSON mode for validation
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": LLM_JUDGE_PROMPT.format(
                source_text=source_text, summary_text=summary
            )}
        ],
        temperature=0.0
    )
    
    result_dict = json.loads(response.choices[0].message.content)
    return ValidationOutput(**result_dict)

```

##3. The Linear Time Decay Flaw (Phase 6)
The Research Risk: The blueprint implements the decay formula directly from your architecture specification: Score = access_count / (days_since_created + 1). In a long-term research simulation, this mathematical structure introduces a terminal trap. Because days_since_created marches forward linearly forever, the denominator grows unbounded. Older nodes are mathematically guaranteed to drop below your PRUNE_THRESHOLD purely because they are old, not because they are useless.

The Fix: Replace linear decay with an Exponential Moving Window / Half-Life Decay formula (similar to radioactive decay or human memory models), which judges relevancy based on recent utility rather than total historical age.

Updated Component: maintenance/decay_scorer.py
Python
import math
from datetime import datetime

def compute_exponential_decay(meta: dict, half_life_days: int = 7) -> float:
    \"\"\"
    Computes a decay score using an exponential half-life based on the LAST time it was accessed,
    ensuring that a node accessed recently remains alive regardless of its total creation age.
    \"\"\"
    last_accessed = datetime.fromisoformat(meta["last_accessed"])
    days_since_last_access = (datetime.utcnow() - last_accessed).days
    
    # Half-life decay math
    decay_factor = math.exp(-days_since_last_access / half_life_days)
    
    # Initial energy provided by its cumulative access density
    base_utility = min(meta["access_count"], 10) 
    
    return base_utility * decay_factor


##4. The Self-Fulfilling Feedback Loop (Phase 5)
The Research Risk: In Phase 5, the retrieval router applies an artificial 0.85 multiplier to lower a Super-Node's L2 distance, prioritizing it over raw chunks. This creates a dangerous algorithmic bias known as a Filter Bubble. Because it is always at the top, it gets accessed constantly, its score stays flawless, and it effectively "shadowbans" underlying raw chunks. New, highly accurate PDFs dropped into the database will never be surfaced.

The Fix: Implement an ε-Greedy (Exploration vs. Exploitation) Retrieval Router. For a small percentage of queries (e.g., 10%), programmatically strip the score boost to check if raw chunks have naturally overtaken the Super-Node in semantic relevance.

Updated Component: retrieval/retriever.py
Python
import random

def retrieve_with_exploration(query: str, epsilon: float = 0.10, top_k: int = 5):
    q_emb = embedder.encode([query])[0]
    
    # Fetch results normally
    sn_results = chroma_client.super_nodes.query(query_embeddings=[q_emb], n_results=3)
    raw_results = chroma_client.raw_chunks.query(query_embeddings=[q_emb], n_results=top_k)
    
    # Determine strategy: Exploitation (Apply Boost) vs Exploration (No Boost)
    apply_boost = random.random() > epsilon
    multiplier = 0.85 if apply_boost else 1.0  # Force exploration 10% of the time
    
    candidates = []
    for doc, meta, dist in zip(sn_results["documents"][0], sn_results["metadatas"][0], sn_results["distances"][0]):
        boosted_score = dist * multiplier
        candidates.append({"doc": doc, "meta": meta, "score": boosted_score, "type": "super_node"})
        
    for doc, meta, dist in zip(raw_results["documents"][0], raw_results["metadatas"][0], raw_results["distances"][0]):
        candidates.append({"doc": doc, "meta": meta, "score": dist, "type": "raw_chunk"})
        
    candidates.sort(key=lambda x: x["score"])
    return candidates[:top_k]


##5. The Parent-Child Vector Collision Flaw (Phase 6 Hierarchy)
The Research Risk: When two Super-Nodes merge into a Meta-Node, they reside in the exact same semantic neighborhood. A single query will retrieve both the parent Meta-Node and the child Super-Nodes simultaneously. This floods your context window with massive redundancy, completely invalidating your "Context Token Economy" benchmarking metrics.

The Fix: Update the retrieval routing logic so that if a meta_node is retrieved, its child super_node IDs are automatically masked/filtered out of the candidate list.

Updated Component: retrieval/retriever.py
Python
def retrieve_with_hierarchical_masking(query: str, top_k: int = 5):
    q_emb = embedder.encode([query])[0]
    
    # Fetch from your collections
    sn_results = chroma_client.super_nodes.query(query_embeddings=[q_emb], n_results=top_k)
    
    candidates = []
    masked_node_ids = set()
    
    # First pass: Identify any Meta-Nodes and log their children for exclusion
    for doc, meta, id_ in zip(sn_results["documents"][0], sn_results["metadatas"][0], sn_results["ids"][0]):
        if meta.get("type") == "meta_node":
            children_ids = json.loads(meta.get("children", "[]"))
            masked_node_ids.update(children_ids)
            
    # Second pass: Compile clean candidates, skipping children shadowed by their parents
    for doc, meta, id_ in zip(sn_results["documents"][0], sn_results["metadatas"][0], sn_results["ids"][0]):
        if id_ in masked_node_ids:
            continue  # Skip child node to prevent redundant token usage
            
        candidates.append({"doc": doc, "meta": meta, "id": id_})
        
    return candidates[:top_k]


##6. The "First-Query" Centroid Drift Flaw (Phase 2 Query Logging)
The Research Risk: The vector representing a query cluster is permanently locked to the very first user query that created it. If the initial query is poorly phrased, the cluster center is locked there. As hundreds of users ask clearer variations, the semantic center of what users actually mean shifts away. Your matches will begin to miss valid organic patterns (Centroid Stagnation).

The Fix: Every time a query matches an existing cluster, update that cluster's stored embedding vector using a running weighted average.

Updated Component: logging_/query_logger.py
Python
def update_cluster_centroid(cluster_id: int, new_query_emb: np.ndarray):
    cluster = sqlite_client.get_cluster(cluster_id)
    current_emb = np.frombuffer(cluster.query_embedding, dtype=np.float32)
    n = cluster.hit_count  # The current size before incrementing
    
    # Running weighted average formula
    # New Centroid = ((Current Centroid * n) + New Embedding) / (n + 1)
    updated_emb = ((current_emb * n) + new_query_emb) / (n + 1)
    
    sqlite_client.update_cluster_embedding(
        cluster_id=cluster_id,
        query_embedding=updated_emb.astype(np.float32).tobytes()
    )

    
##7. The Infinite Merger Loop Flaw (Phase 6 Hierarchy)
The Research Risk: The hierarchy_merger job pulls all elements where type == "super_node". Because original sibling Super-Nodes are kept as children after a merge, their similarity remains >0.88. Every single day, your system will multiply duplicate Meta-Nodes for the exact same static pairs, causing exponential database bloat.

The Fix: Modify the metadata of a Super-Node to track its assignment status (in_hierarchy). Once assigned to a Meta-Node, exclude it from future pairing passes.

Updated Component: maintenance/hierarchy_merger.py
Python
def merge_similar_nodes():
    # Only pull super-nodes that do NOT already belong to a hierarchy branch
    all_sn = chroma_client.super_nodes.get(
        where={
            "$and": [
                {"type": {"$eq": "super_node"}},
                {"in_hierarchy": {"$eq": False}} # Explicitly filter out unassigned nodes
            ]
        },
        include=["metadatas", "embeddings", "documents", "ids"]
    )
    
    # ... [Perform clustering matrix logic matching your threshold] ...
    
    # When a pair (i, j) is merged:
    meta_id = f"mn_{int(datetime.utcnow().timestamp())}"
    
    # Update the children to lock them out of future iterations
    for idx in [i, j]:
        child_id = all_sn["ids"][idx]
        child_meta = all_sn["metadatas"][idx]
        child_meta["in_hierarchy"] = True
        child_meta["parent_meta_id"] = meta_id
        chroma_client.super_nodes.update(ids=[child_id], metadatas=[child_meta])


##8. The Cascading Amnesia Flaw (Phase 6 Staleness Check)
The Research Risk: If a researcher fixes a minor typo in an upstream document, text layout boundaries shift, causing the recursive chunker to generate new hashes and split points. Old chunk IDs disappear. The staleness check panics and drops the Super-Node entirely, wiping clean months of organic user intent metrics and high hit_count data over minor upstream edits.

The Fix: Implement a soft-validation flag. If a source chunk shifts, mark the Super-Node as is_stale. Serve the summary but trigger an immediate high-priority asynchronous patch to rebuild it without dropping historical metrics.

Updated Component: maintenance/staleness_checker.py
Python
def run_soft_staleness_check():
    all_sn = chroma_client.super_nodes.get(where={"type": {"$eq": "super_node"}}, include=["metadatas", "ids"])
    
    for sn_id, meta in zip(all_sn["ids"], all_sn["metadatas"]):
        source_chunk_ids = json.loads(meta["source_chunks"])
        
        # Verify how many original IDs are actually missing from the raw collection
        existing = chroma_client.raw_chunks.get(ids=source_chunk_ids, include=["ids"])
        match_ratio = len(existing["ids"]) / len(source_chunk_ids)
        
        if match_ratio < 1.0: # Some structural mapping split has changed
            # Update metadata to stale state without deleting historical access counts
            meta["is_stale"] = True
            chroma_client.super_nodes.update(ids=[sn_id], metadatas=[meta])
            
            # Request background asynchronous re-synthesis to update text lineage
            sqlite_client.flag_cluster_for_re_synthesis(meta["cluster_id"])


##9. The "Knowledge JPEG" Artifacting Flaw (Phase 2 & Phase 4)
The Research Risk: If a Super-Node is fed into the prompt to generate an updated Super-Node, you are forcing the LLM to summarize a summary. Over weeks of continuous operation, this causes Generative Artifacting (like saving a JPEG repeatedly). Nuances are stripped away, and hallucinations multiply.

The Fix: Ground-Truth Anchoring. Only log raw_chunks in the cluster lineage. The LLM must always build syntheses from the ground truth.

Updated Component: middleware/query_interceptor.py
Python
@app.middleware("http")
async def log_query_middleware(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/api/v1/query":
        body = await request.json()
        
        # Ground-Truth Filter: Only log IDs that belong to the original documents
        # Assuming raw chunks start with "doc_" and super_nodes start with "sn_"
        raw_grounding_ids = [
            chunk_id for chunk_id in response_ctx["chunk_ids"] 
            if chunk_id.startswith("doc_")
        ]
        
        query_logger.log_query(body["query"], raw_grounding_ids)
    return response


##10. The ChromaDB Concurrency Illusion (Phase 5)
The Research Risk: ChromaDB is a vector store, not a transactional database. If a high-throughput benchmarking script simulates 50 concurrent users, 50 threads will read the Super-Node's access count simultaneously, increment it to the same value in memory, and write the exact same value back. You lose 49 hits, meaning highly valuable nodes will be pruned purely because the database couldn't count fast enough.

The Fix: Never use a vector database for rapidly changing counters. Move the access_count tracking into SQLite, and merge it with ChromaDB during maintenance.

Updated Component: retrieval/retriever.py & decay_scorer.py
Python
# Updated snippet for retrieval/retriever.py (Phase 5)
def log_super_node_access(super_node_ids: list[str]):
    # Do not write to ChromaDB. Execute an atomic SQL update instead.
    sqlite_client.execute(
        \"\"\"
        UPDATE query_clusters 
        SET hit_count = hit_count + 1, last_hit = CURRENT_TIMESTAMP
        WHERE super_node_id IN (?, ?, ?)
        \"\"\", 
        super_node_ids
    )

# Then, in Phase 6 (decay_scorer.py), pull the true hit_count from SQLite 
# to calculate the exponential decay score.
"""