import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

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

# We support OpenAI-compatible endpoints (like Ollama, vLLM, Groq, Together) via base_url
LLM_BASE_URL = os.getenv("LLM_BASE_URL", None) 
GENERATOR_LLM_MODEL = os.getenv("GENERATOR_LLM_MODEL", "Llama-3.1-8B-Instruct")
SUMMARIZER_LLM_MODEL = os.getenv("SUMMARIZER_LLM_MODEL", "Qwen-2.5-32B-Instruct")

# Phase 2: Query Clustering thresholds
COSINE_THRESHOLD = float(os.getenv("COSINE_THRESHOLD", "0.92"))
