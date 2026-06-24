import os
import sys

# Add project root to sys.path so absolute imports (app.x) work
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
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
from app.ingestion.html_cleaner import clean
from app.ingestion.cleaner import clean_text
from app.ingestion.section_parser import parse_sections

def run_test():
    print("Fetching 1 article from MedQuAD to test the new cleaning and section parsing pipeline...\n")
    
    for row in stream_medquad():
        print(f"Testing with: {row.question_focus} (Doc ID: {row.document_id})")
        print(f"URL: {row.document_url}\n")
        
        try:
            # 1. Fetch
            raw_html = get(row.document_url)
            
            # 2. HTML to Text
            text = clean(raw_html)
            print(f"-> Text after HTML stripping: {len(text)} characters")
            
            # 3. Clean Boilerplate
            cleaned_text = clean_text(text)
            print(f"-> Text after removing noise (cleaner.py): {len(cleaned_text)} characters\n")
            
            # 4. Parse Sections
            sections = parse_sections(cleaned_text)
            
            print("="*60)
            print(" EXTRACTED SECTIONS ")
            print("="*60)
            
            for sec_name, sec_text in sections:
                print(f"\n[{sec_name.upper()}] ({len(sec_text)} chars)")
                # Print a preview of the section content safely for Windows terminals
                preview = sec_text[:150].replace('\n', ' ')
                safe_preview = preview.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
                print(f"Preview: {safe_preview}...")
                
            # Save the full result to a text file for manual review
            output_file = os.path.join(script_dir, "cleaned_sample.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"URL: {row.document_url}\n")
                f.write(f"Doc ID: {row.document_id}\n\n")
                f.write("="*60 + "\n")
                f.write(" FULL CLEANED TEXT (Before Section Parsing)\n")
                f.write("="*60 + "\n\n")
                f.write(cleaned_text + "\n\n\n")
                f.write("="*60 + "\n")
                f.write(" PARSED SECTIONS\n")
                f.write("="*60 + "\n\n")
                for sec_name, sec_text in sections:
                    f.write(f"[{sec_name.upper()}]\n")
                    f.write(f"{sec_text}\n\n")
                    
            print(f"\n=> I have saved the full cleaned text to: {output_file}")
            
            break # Just test one document
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_test()
