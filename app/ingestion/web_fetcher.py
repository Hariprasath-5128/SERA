import time
import httpx
import requests
from urllib.parse import urlparse
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import CRAWL_RATE_PER_SEC, USER_AGENT, CRAWL_TIMEOUT, CRAWL_MAX_RETRIES

class FetchError(Exception):
    """Custom exception for HTTP fetch failures"""
    pass

def get_wayback_url(url, timestamp=None):
    api_url = f'http://archive.org/wayback/available?url={url}'
    if timestamp:
        api_url += f'&timestamp={timestamp}'
    try:
        data = requests.get(api_url, timeout=10).json()
        if data.get('archived_snapshots') and 'closest' in data['archived_snapshots']:
            closest = data['archived_snapshots']['closest']
            if timestamp and timestamp.startswith('2023'):
                if closest['timestamp'][:4] < '2023':
                    return None
            return closest['url']
    except:
        pass
    if timestamp:
        return f"http://web.archive.org/web/{timestamp}/{url}"
    return None

@retry(
    stop=stop_after_attempt(CRAWL_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)
def get(url: str, use_wayback: bool = False, wb_timestamp: str = None, use_playwright: bool = False) -> tuple[str, str]:
    if CRAWL_RATE_PER_SEC > 0:
        time.sleep(1.0 / CRAWL_RATE_PER_SEC)
        
    try:
        if use_playwright:
            from playwright.sync_api import sync_playwright
            print(f"   [INFO] Launching headless browser to extract {url}...")
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(user_agent=USER_AGENT)
                # Wait for DOM content to load, then wait 2 seconds for JS rendering to populate UI
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=15000)
                    page.wait_for_timeout(2000)
                except Exception as e:
                    print(f"   [WARN] Playwright timeout/error while waiting, proceeding to extract content: {e}")
                
                html = page.content()
                browser.close()
                if html:
                    return html, "Live (Playwright)"
                raise FetchError(f"Playwright failed to retrieve HTML for {url}")

        elif use_wayback:
            snap_url = get_wayback_url(url, timestamp=wb_timestamp) if wb_timestamp else get_wayback_url(url)
            if snap_url:
                resp = httpx.get(
                    snap_url, 
                    timeout=15, 
                    headers={"User-Agent": USER_AGENT},
                    follow_redirects=True
                )
                resp.raise_for_status()
                html = resp.text
                if html and "This Page Has Moved" not in html and "Redirect to first topic" not in html:
                    return html, "Wayback"
            
            if 'cdc.gov' in urlparse(url).netloc:
                snap_url = get_wayback_url(url, timestamp='20140101')
                if snap_url:
                    resp = httpx.get(
                        snap_url, 
                        timeout=15, 
                        headers={"User-Agent": USER_AGENT},
                        follow_redirects=True
                    )
                    resp.raise_for_status()
                    html = resp.text
                    if html and "This Page Has Moved" not in html:
                        return html, "Wayback (2014)"
            
            raise FetchError(f"Wayback machine failed to retrieve valid HTML for {url}")
            
        else:
            try:
                resp = httpx.get(
                    url, 
                    timeout=CRAWL_TIMEOUT, 
                    headers={"User-Agent": USER_AGENT},
                    follow_redirects=True
                )
                resp.raise_for_status()
                return resp.text, "Live"
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    print(f"   [WARN] 404 on live site for {url}, falling back to Wayback Machine...")
                    snap_url = get_wayback_url(url)
                    if snap_url:
                        resp = httpx.get(
                            snap_url, 
                            timeout=15, 
                            headers={"User-Agent": USER_AGENT},
                            follow_redirects=True
                        )
                        resp.raise_for_status()
                        html = resp.text
                        if html and "This Page Has Moved" not in html and "Redirect to first topic" not in html:
                            return html, "Wayback (Fallback)"
                raise FetchError(f"HTTP error {e.response.status_code} while fetching {url}") from e
            
    except httpx.RequestError as e:
        raise FetchError(f"Request error while fetching {url}: {str(e)}") from e
