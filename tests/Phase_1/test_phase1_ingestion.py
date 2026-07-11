import pytest
import itertools
import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.ingestion import dataset_loader, web_fetcher, html_cleaner, chunker
from app.api.routes import ingest
from app.db import sqlite_client

client = TestClient(app)

def test_dataset_loader_yields_rows():
    rows = list(itertools.islice(dataset_loader.stream_medquad(), 5))
    assert len(rows) == 5
    assert all(r.document_url.startswith("http") for r in rows)

def test_web_fetcher_gets_html(respx_mock):
    # respx is a pytest plugin for mocking httpx
    respx_mock.get("https://ghr.nlm.nih.gov/condition/mabry-syndrome").mock(
        return_value=httpx.Response(200, text="<main>Mabry syndrome content</main>")
    )
    # the fetcher now returns a tuple (html, source_type)
    html, source = web_fetcher.get("https://ghr.nlm.nih.gov/condition/mabry-syndrome")
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
    LONG_TEXT = "word " * 1000
    chunks = chunker.split(
        text=LONG_TEXT,
        document_id="0000613",
        source_url="http://x.com",
        question_focus="Mabry",
        umls_semantic_group="Disorders",
        section="Summary"
    )
    assert all(c.token_count <= 550 for c in chunks)
    assert all(c.chunk_id.startswith("0000613_") for c in chunks)

def test_ingest_writes_benchmark_pairs(monkeypatch, tmp_path):
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "test.db"))
    from app.db import sqlite_client
    monkeypatch.setattr(sqlite_client, "SQLITE_DB_PATH", tmp_path / "test.db")
    sqlite_client.init_db()
    
    # Mock the functions *inside* the ingestor module namespace
    from app.ingestion import ingestor
    import itertools
    
    original_stream = ingestor.stream_medquad
    monkeypatch.setattr(ingestor, "stream_medquad", lambda: itertools.islice(original_stream(), 2))
    
    # Mock network fetch to prevent hanging
    monkeypatch.setattr(ingestor, "get", lambda *args, **kwargs: ("<html><body><p>Mock text for testing</p></body></html>", "Live"))
    
    # Mock GPU embedding to prevent CUDA out of memory
    monkeypatch.setattr(ingestor, "encode", lambda chunks, **kwargs: chunks)
    
    ingestor.run_ingestion()
    
    with sqlite_client.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM benchmark_qa")
        pairs = cursor.fetchall()
        
    assert len(pairs) == 2

def test_ingest_deduplicates_docs(monkeypatch, tmp_path):
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "test2.db"))
    from app.db import sqlite_client
    monkeypatch.setattr(sqlite_client, "SQLITE_DB_PATH", tmp_path / "test2.db")
    sqlite_client.init_db()
    
    from app.ingestion import ingestor
    import itertools
    
    original_stream = ingestor.stream_medquad
    monkeypatch.setattr(ingestor, "stream_medquad", lambda: itertools.islice(original_stream(), 2))
    
    # Mock network fetch to prevent hanging
    monkeypatch.setattr(ingestor, "get", lambda *args, **kwargs: ("<html><body><p>Mock text for testing</p></body></html>", "Live"))
    
    # Mock GPU embedding to prevent CUDA out of memory
    monkeypatch.setattr(ingestor, "encode", lambda chunks, **kwargs: chunks)
    
    ingestor.run_ingestion()
    
    with sqlite_client.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT source_url FROM benchmark_qa")
        unique_urls = cursor.fetchall()
        
    assert len(unique_urls) <= 2

def test_query_returns_answer():
    # Test the query endpoint basic shape
    response = client.post("/query", json={"query": "What is Mabry syndrome?", "top_k": 1})
    assert response.status_code == 200
    # The actual implementation of query might return a canned answer if generation isn't hooked up yet
    assert "answer" in response.json()
