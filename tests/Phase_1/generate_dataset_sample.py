import os
import sys
import json
from collections import defaultdict
from urllib.parse import urlparse

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

os.environ["CRAWL_MAX_RETRIES"] = "0"
os.environ["CRAWL_TIMEOUT"] = "10"

from datasets import load_dataset
from app.ingestion.ingestor import process_document, map_nihseniorhealth_question

def main():
    print("Loading MedQuAD Dataset...")
    ds = load_dataset("lavita/MedQuAD", split="train")
    
    target_domains = [
        "ghr.nlm.nih.gov",
        "cancer.gov",
        "nhlbi.nih.gov",
        "niddk.nih.gov",
        "cdc.gov",
        "ninds.nih.gov",
        "nlm.nih.gov",
        "nihseniorhealth.gov"
    ]
    
    # We want 2 entirely NEW samples per domain.
    # To ensure they are new, we will skip the first 5 unique URLs for each domain.
    skip_count = 5
    target_count = 2
    
    seen_urls = defaultdict(set)
    collected_samples = defaultdict(list)
    nih_senior_health_data = defaultdict(dict)
    
    stats = defaultdict(lambda: {"accessed": 0, "not_accessed": 0, "wayback_used": 0, "live_used": 0})
    
    print("Iterating over dataset...")
    for row in ds:
        url = row['document_url']
        domain = urlparse(url).netloc
        
        # Determine actual domain category
        matched_domain = None
        for d in target_domains:
            if d in domain:
                matched_domain = d
                break
                
        if not matched_domain:
            continue
            
        if matched_domain == "nihseniorhealth.gov":
            # For nihseniorhealth, we group rows by URL natively
            if len(seen_urls[matched_domain]) < skip_count:
                seen_urls[matched_domain].add(url)
                continue
                
            if url not in seen_urls[matched_domain] and len(collected_samples[matched_domain]) >= target_count:
                continue
                
            seen_urls[matched_domain].add(url)
            mapped_header = map_nihseniorhealth_question(row.get('question_type', 'information'))
            if mapped_header:
                nih_senior_health_data[url][mapped_header] = row['answer']
                
            # If we've gathered at least 3 sections for this URL, consider it a full sample (just for reporting)
            if len(nih_senior_health_data[url]) >= 3 and url not in [s['url'] for s in collected_samples[matched_domain]]:
                collected_samples[matched_domain].append({"url": url, "sections": nih_senior_health_data[url]})
                stats[matched_domain]["accessed"] += 1
                stats[matched_domain]["live_used"] += 1 # Native counts as Live
        else:
            if url in seen_urls[matched_domain]:
                continue
                
            seen_urls[matched_domain].add(url)
            
            if len(seen_urls[matched_domain]) <= skip_count:
                continue
                
            if len(collected_samples[matched_domain]) >= target_count:
                continue
                
            if stats[matched_domain]["not_accessed"] > 10:
                print(f"Skipping {matched_domain} due to too many failures")
                continue
                
            print(f"[{matched_domain}] Fetching new sample: {url}")
            try:
                sections, source = process_document(url, domain)
                if sections:
                    collected_samples[matched_domain].append({"url": url, "sections": sections})
                    stats[matched_domain]["accessed"] += 1
                    if "Wayback" in source:
                        stats[matched_domain]["wayback_used"] += 1
                    else:
                        stats[matched_domain]["live_used"] += 1
                else:
                    stats[matched_domain]["not_accessed"] += 1
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                stats[matched_domain]["not_accessed"] += 1
                
        # Check if we are done with all domains
        done = True
        for d in target_domains:
            if len(collected_samples[d]) < target_count:
                done = False
                break
        if done:
            break
            
    print("Writing output files...")
    
    # 1. Write sample_dataset.txt
    sample_path = os.path.join(project_root, "data", "sample_dataset.txt")
    with open(sample_path, "w", encoding="utf-8") as f:
        for domain in target_domains:
            f.write(f"==================================================\n")
            f.write(f"DOMAIN: {domain}\n")
            f.write(f"==================================================\n\n")
            for idx, sample in enumerate(collected_samples[domain]):
                f.write(f"--- Sample {idx+1} ---\n")
                f.write(f"URL: {sample['url']}\n\n")
                for sec_name, sec_text in sample['sections'].items():
                    f.write(f"[{sec_name.strip('[]')}]\n")
                    f.write(f"{sec_text}\n\n")
                f.write("\n")
                
    # 2. Write report.txt
    report_path = os.path.join(project_root, "data", "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DOMAIN EXTRACTION REPORT\n")
        f.write("========================\n\n")
        
        total_accessed = 0
        total_not_accessed = 0
        total_wayback = 0
        total_live = 0
        
        for domain in target_domains:
            s = stats[domain]
            f.write(f"{domain}:\n")
            f.write(f"  - Accessed successfully: {s['accessed']}\n")
            f.write(f"  - Failed to access:      {s['not_accessed']}\n")
            f.write(f"  - Live/Native Used:      {s['live_used']}\n")
            f.write(f"  - Wayback Used:          {s['wayback_used']}\n\n")
            
            total_accessed += s['accessed']
            total_not_accessed += s['not_accessed']
            total_wayback += s['wayback_used']
            total_live += s['live_used']
            
        f.write("========================\n")
        f.write(f"TOTAL ACCESSED:     {total_accessed}\n")
        f.write(f"TOTAL FAILED:       {total_not_accessed}\n")
        f.write(f"TOTAL LIVE/NATIVE:  {total_live}\n")
        f.write(f"TOTAL WAYBACK:      {total_wayback}\n")

    print(f"Done! Created {sample_path} and {report_path}")

if __name__ == "__main__":
    main()
