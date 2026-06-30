import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from app.ingestion.ingestor import process_document

url = "https://rarediseases.info.nih.gov/diseases/1082/2-methylbutyryl-coa-dehydrogenase-deficiency"
domain = "rarediseases.info.nih.gov"

print(f"Testing process_document with {url}")
sections, source = process_document(url, domain)

print(f"Source: {source}")
for sec, content in sections.items():
    print(f"\n--- {sec} ---")
    print(content[:500] + ("..." if len(content) > 500 else ""))
