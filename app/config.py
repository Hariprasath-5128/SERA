import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# override=True ensures stale OS-level env vars are always replaced by .env values
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv(override=True)

# System Directories
DATA_DIR = BASE_DIR / "data"
SQLITE_DIR = DATA_DIR / "sqlite"
CHROMA_DIR = DATA_DIR / "chroma"
CACHE_DIR = DATA_DIR / "cache"

# Ensure directories exist
SQLITE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# API Keys & Tokens
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY_FALLBACK = os.getenv("OPENAI_API_KEY_FALLBACK")

# Ingestion & Crawling Settings
USER_AGENT = os.getenv("USER_AGENT", "SERABot/2.0 (+https://github.com/user/SERA)")
CRAWL_RATE_PER_SEC = float(os.getenv("CRAWL_RATE_PER_SEC", "1.0"))  # 1 req/sec max to respect NLM/NIH rate-limits
CRAWL_TIMEOUT = int(os.getenv("CRAWL_TIMEOUT", "15"))
CRAWL_MAX_RETRIES = int(os.getenv("CRAWL_MAX_RETRIES", "3"))

# Chunker Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # Target token limit per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Database Configurations
SQLITE_DB_PATH = SQLITE_DIR / "rag.db"
CHROMA_PERSIST_PATH = str(CHROMA_DIR)
CHROMA_RAW_COLLECTION = "raw_chunks"
CHROMA_SUPER_COLLECTION = "super_nodes"

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

# Retrieval & LLM Generation Settings
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

# Phase 5: Preferential Retrieval
# SU4: ε-Greedy exploration rate — fraction of queries that skip the super-node boost
EXPLORATION_EPSILON = float(os.getenv("EXPLORATION_EPSILON", "0.10"))
# SU4: L2 distance multiplier applied to super-nodes during exploitation queries
SUPER_NODE_SCORE_MULTIPLIER = float(os.getenv("SUPER_NODE_SCORE_MULTIPLIER", "0.85"))

# We support OpenAI-compatible endpoints (like Ollama, vLLM, Groq, Together) via base_url
LLM_BASE_URL = os.getenv("LLM_BASE_URL", None) 
GENERATOR_LLM_MODEL = os.getenv("GENERATOR_LLM_MODEL", "Llama-3.1-8B-Instruct")
SUMMARIZER_LLM_MODEL = os.getenv("SUMMARIZER_LLM_MODEL", "Qwen-2.5-32B-Instruct")

# Phase 2: Query Clustering thresholds
COSINE_THRESHOLD = float(os.getenv("COSINE_THRESHOLD", "0.92"))
# Minimum number of cluster hits before Phase 3 triggers synthesis
HIT_COUNT_THRESHOLD = int(os.getenv("HIT_COUNT_THRESHOLD", "10"))
# SU12: number of *additional* hits after last synthesis before re-triggering
RESYNTH_DELTA = int(os.getenv("RESYNTH_DELTA", "10"))

# Phase 3: Synthesis Trigger thresholds
POLL_INTERVAL_MINUTES = int(os.getenv("POLL_INTERVAL_MINUTES", "5"))
CENTROID_SHIFT_THRESHOLD = float(os.getenv("CENTROID_SHIFT_THRESHOLD", "0.05"))

# Phase 4: Synthesis quality thresholds
# SU2: minimum fact-coverage score to accept an LLM synthesis
VALIDATION_COVERAGE_THRESHOLD = float(os.getenv("VALIDATION_COVERAGE_THRESHOLD", "0.90"))
MAX_SYNTHESIS_RETRIES = int(os.getenv("MAX_SYNTHESIS_RETRIES", "3"))
# SU14: Synthesis Fidelity Bound
# drift_margin > this → soft flag fidelity_flagged=True (node still served)
MAX_DRIFT_MARGIN = float(os.getenv("MAX_DRIFT_MARGIN", "0.08"))
# sim_summary_to_source < this → hard reject, force re-synthesis
MIN_SOURCE_ANCHOR = float(os.getenv("MIN_SOURCE_ANCHOR", "0.55"))

# Phase 6: Maintenance thresholds
# SU3: half-life for exponential decay scoring (days)
HALF_LIFE_DAYS = float(os.getenv("HALF_LIFE_DAYS", "7.0"))
# Decay score below this → node is pruned
DECAY_PRUNE_THRESHOLD = float(os.getenv("DECAY_PRUNE_THRESHOLD", "0.05"))
# SU11: maximum merge nesting depth; deeper pairs go to manual_review_queue
# Depth 3 allows: Disease → Aspect (symptoms/treatment/causes) → Detail
MAX_LINEAGE_DEPTH = int(os.getenv("MAX_LINEAGE_DEPTH", "3"))
# Cosine similarity threshold above which two super-nodes are merged into a meta-node.
# 0.75 is chosen to group thematically related nodes (e.g. different aspects of the
# same disease: symptoms, causes, treatments) without merging unrelated diseases.
HIERARCHY_MERGE_THRESHOLD = float(os.getenv("HIERARCHY_MERGE_THRESHOLD", "0.75"))
