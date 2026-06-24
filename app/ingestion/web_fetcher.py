import time
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import CRAWL_RATE_PER_SEC, USER_AGENT, CRAWL_TIMEOUT, CRAWL_MAX_RETRIES

class FetchError(Exception):
    """Custom exception for HTTP fetch failures"""
    pass

@retry(
    stop=stop_after_attempt(CRAWL_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)
def get(url: str) -> str:
    """
    HTTP GET an article's HTML.
    Includes rate-limiting (sleeps according to CRAWL_RATE_PER_SEC) 
    and exponential backoff retries via Tenacity.
    """
    if CRAWL_RATE_PER_SEC > 0:
        time.sleep(1.0 / CRAWL_RATE_PER_SEC)
        
    try:
        resp = httpx.get(
            url, 
            timeout=CRAWL_TIMEOUT, 
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True
        )
        resp.raise_for_status()
        return resp.text
    except httpx.RequestError as e:
        raise FetchError(f"Request error while fetching {url}: {str(e)}") from e
    except httpx.HTTPStatusError as e:
        raise FetchError(f"HTTP error {e.response.status_code} while fetching {url}") from e
