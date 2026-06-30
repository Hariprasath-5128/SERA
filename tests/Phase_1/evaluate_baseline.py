import os
import argparse
import concurrent.futures
import json
import logging
from collections import defaultdict
from urllib.parse import urlparse
from datasets import load_dataset
import traceback
import test_parser
import custom_extractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_overlap(extracted_text, native_text):
    if not extracted_text or not native_text:
        return 0.0
    
    # Tokenize simply by words
    ext_tokens = set(extracted_text.lower().split())
    native_tokens = set(native_text.lower().split())
    
    if not native_tokens:
        return 0.0
        
    overlap = len(ext_tokens.intersection(native_tokens))
    return overlap / len(native_tokens)

def process_url(url, rows):
    try:
        htmls = test_parser.fetch_htmls_for_url(url)
        if not htmls or not htmls[0][0]:
            return url, None, "Fetch Failed"
            
        domain = urlparse(url).netloc
        # Process the first HTML (or combine them if multiple like NHLBI)
        combined_extracted_text = ""
        sections = {}
        for html, source in htmls:
            if not html: continue
            extracted = custom_extractor.extract_structured_content(html, domain, url=url)
            sections.update(extracted)
            
        combined_extracted_text = " ".join(sections.values())
        
        results = []
        for row in rows:
            native_ans = row.get('answer', '')
            overlap = calculate_overlap(combined_extracted_text, native_ans)
            results.append({
                'question': row.get('question', ''),
                'native_answer': native_ans[:200] + "..." if len(native_ans) > 200 else native_ans,
                'overlap_score': overlap
            })
            
        return url, results, "Success", sections
        
    except Exception as e:
        return url, None, str(e), {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Limit number of unique URLs to process')
    parser.add_argument('--workers', type=int, default=10, help='Number of thread workers')
    parser.add_argument('--domain', type=str, default=None, help='Filter by specific domain')
    parser.add_argument('--output', type=str, default='baseline_results.jsonl', help='Output JSONL file')
    args = parser.parse_args()

    logging.info("Loading MedQuAD dataset...")
    dataset = load_dataset("lavita/MedQuAD", split="train")
    
    url_groups = defaultdict(list)
    for row in dataset:
        url = row.get('document_url')
        if not url: continue
        
        domain = urlparse(url).netloc
        if args.domain and args.domain not in domain:
            continue
            
        url_groups[url].append(row)
        
    unique_urls = list(url_groups.keys())
    if args.limit:
        unique_urls = unique_urls[:args.limit]
        
    logging.info(f"Found {len(unique_urls)} unique URLs to process.")
    
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    
    total_processed = 0
    success_count = 0
    total_overlap = 0.0
    questions_evaluated = 0
    
    ingested_path = os.path.join(os.path.dirname(__file__), 'ingested_dataset.jsonl')
    
    with open(output_path, 'w', encoding='utf-8') as out_f, open(ingested_path, 'w', encoding='utf-8') as ingested_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_url = {executor.submit(process_url, url, url_groups[url]): url for url in unique_urls}
            
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                total_processed += 1
                
                try:
                    url, results, status, sections = future.result()
                    if results:
                        success_count += 1
                        for r in results:
                            questions_evaluated += 1
                            total_overlap += r['overlap_score']
                            
                        # Write baseline metrics
                        out_f.write(json.dumps({
                            'url': url,
                            'status': status,
                            'questions_count': len(results),
                            'avg_overlap': sum(r['overlap_score'] for r in results) / len(results)
                        }) + '\n')
                        
                        # Write fully ingested, cleaned text
                        ingested_f.write(json.dumps({
                            'url': url,
                            'extracted_sections': sections
                        }) + '\n')
                        
                except Exception as exc:
                    logging.error(f"{url} generated an exception: {exc}")
                    
                if total_processed % 50 == 0:
                    logging.info(f"Processed {total_processed}/{len(unique_urls)} URLs...")
                    
    avg_score = (total_overlap / questions_evaluated) if questions_evaluated > 0 else 0
    success_rate = (success_count / len(unique_urls)) * 100 if unique_urls else 0
    
    logging.info("="*50)
    logging.info(f"EVALUATION COMPLETE")
    logging.info(f"URLs Processed successfully: {success_count}/{len(unique_urls)} ({success_rate:.1f}%)")
    logging.info(f"Total Questions Evaluated: {questions_evaluated}")
    logging.info(f"Average Token Overlap Score: {avg_score:.3f}")
    logging.info("="*50)

if __name__ == "__main__":
    main()
