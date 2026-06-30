import os
import json
from collections import defaultdict
from datasets import load_dataset
from urllib.parse import urlparse
import sys
import concurrent.futures

sys.path.insert(0, 'C:/Projects/SERA')
from app.ingestion.dataset_loader import stream_medquad
from app.ingestion.chunker import split
from app.ingestion.web_fetcher import get
from app.ingestion.custom_extractor import extract_structured_content

def process_url(row, domain):
    if 'nihseniorhealth.gov' in domain:
        return {'url': row.document_url, 'domain': domain, 'status': 'live', 'source': 'Live'}
        
    url = row.document_url
    all_chunks = []
    source = "Unknown"
    
    try:
        raw, src = get(url, use_wayback=False, use_playwright=True)
        secs = extract_structured_content(raw, domain, url)
        for sec_name, sec_text in secs.items():
            chunks = split(sec_text, row.document_id, url, row.question_focus, "Disorder", sec_name)
            all_chunks.extend(chunks)
        if all_chunks:
            source = src
    except Exception:
        pass
        
    if not all_chunks:
        for year in range(2026, 2012, -1):
            wb = f"{year}0101"
            try:
                raw, src = get(url, use_wayback=True, wb_timestamp=wb, use_playwright=False)
                secs = extract_structured_content(raw, domain, url)
                for sec_name, sec_text in secs.items():
                    chunks = split(sec_text, row.document_id, url, row.question_focus, "Disorder", sec_name)
                    all_chunks.extend(chunks)
                if all_chunks:
                    source = src
                    break
            except Exception:
                pass
                
    if all_chunks:
        if source == 'Live' or 'Playwright' in source:
            return {'url': url, 'domain': domain, 'status': 'live', 'source': source}
        else:
            return {'url': url, 'domain': domain, 'status': 'wayback', 'source': source}
    else:
        return {'url': url, 'domain': domain, 'status': 'skipped', 'source': 'None'}

def main():
    print("Starting full dataset analysis dry-run (MULTI-THREADED)...")
    stats = defaultdict(lambda: {
        'total': 0,
        'live': 0,
        'wayback': 0,
        'skipped': 0,
        'skipped_urls': []
    })
    
    seen_urls = set()
    overall_total = 0
    overall_live = 0
    overall_wayback = 0
    overall_skipped = 0
    
    tasks = []
    
    # Pre-collect tasks
    for row in stream_medquad():
        domain = urlparse(row.document_url).netloc
        if row.document_url not in seen_urls:
            stats[domain]['total'] += 1
            seen_urls.add(row.document_url)
            overall_total += 1
            tasks.append((row, domain))
            
    print(f"Loaded {len(tasks)} unique URLs. Processing...")
    
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(process_url, row, domain): (row, domain) for row, domain in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            completed += 1
            if completed % 100 == 0:
                print(f"Processed {completed}/{len(tasks)}...")
                
            d = res['domain']
            if res['status'] == 'live':
                stats[d]['live'] += 1
                overall_live += 1
            elif res['status'] == 'wayback':
                stats[d]['wayback'] += 1
                overall_wayback += 1
            else:
                stats[d]['skipped'] += 1
                overall_skipped += 1
                if len(stats[d]['skipped_urls']) < 5:
                    stats[d]['skipped_urls'].append(res['url'])
                    
    print("\n" + "="*50)
    print(" FINAL ANALYSIS REPORT ")
    print("="*50)
    for d, s in stats.items():
        print(f"\nDOMAIN: {d}")
        print(f"  - Total Links in Dataset: {s['total']}")
        print(f"  - Accessible (Live/JSW):  {s['live']}")
        print(f"  - Accessible (Wayback):   {s['wayback']}")
        print(f"  - Links Skipped:          {s['skipped']}")
        if s['skipped_urls']:
            print("  - Sample of skipped links:")
            for u in s['skipped_urls']:
                print(f"      * {u}")
                
    print("\n=== OVERALL DATASET STATS ===")
    print(f"  - Total Links in Dataset: {overall_total}")
    print(f"  - Total Accessible (Live/JSW): {overall_live}")
    print(f"  - Total Accessible (WB):   {overall_wayback}")
    print(f"  - Total Links Skipped:     {overall_skipped}")

if __name__ == '__main__':
    main()
