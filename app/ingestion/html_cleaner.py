import re
from bs4 import BeautifulSoup

def clean(html: str) -> str:
    """
    Strips navigation, footers, ads, and boilerplate code from HTML 
    to extract just the clean article text.
    """
    soup = BeautifulSoup(html, "lxml")
    
    # Remove unwanted tags entirely
    for tag in soup(["nav", "footer", "script", "style", "aside", "header"]):
        tag.decompose()
        
    # Unwrap inline tags so get_text() doesn't insert newlines around them
    for tag in soup(["a", "span", "i", "b", "strong", "em", "u"]):
        tag.unwrap()
        
    # Crucial: Merge the newly unwrapped adjacent strings into single text blocks
    soup.smooth()
    
    # NLM/NIH pages: main content lives in <div id="section-body"> or <main>
    main = soup.find("main") or soup.find("div", {"id": "section-body"}) or soup.body
    
    if not main:
        return ""
        
    text = main.get_text(separator="\n", strip=True)
    
    # Collapse blank lines (3 or more newlines into 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

if __name__ == "__main__":
    import os
    import sys
    
    # Add project root to sys.path so absolute imports (app.x) work when run directly
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.insert(0, project_root)
    
    # Hack to load .env token for standalone testing
    env_path = os.path.join(project_root, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    os.environ["HF_TOKEN"] = line.strip().split("=")[1]
                    
    from app.ingestion.dataset_loader import stream_medquad
    from app.ingestion.web_fetcher import get
    from app.ingestion.url_extractor import UrlExtractor
    
    extractor = UrlExtractor()
    print("Fetching and cleaning 5 unique articles from the dataset stream...\n")
    
    count = 0
    for row in stream_medquad():
        if count >= 1:  # Just do 1 to keep the output readable
            break
            
        if extractor.is_new_document(row.document_id):
            print(f"[{count+1}/1] Fetching doc_id: {row.document_id} from {row.document_url}...")
            try:
                raw_html = get(row.document_url)
                
                print(f"\n---> RAW HTML ({len(raw_html)} chars) <---")
                print(raw_html[:300].replace('\n', ' ') + " ...[TRUNCATED]")
                
                clean_text = clean(raw_html)
                print(f"\n---> CLEANED TEXT ({len(clean_text)} chars) <---")
                print(clean_text[:300].replace('\n', ' ') + " ...[TRUNCATED]\n")
                
                count += 1
            except Exception as e:
                print(f"Failed to fetch or clean: {e}\n")

