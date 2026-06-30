import sys
import os
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

env_path = os.path.join(script_dir, ".env")
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)

from app.retrieval.retriever import Retriever
from app.generation.generator import Generator

query = "What is Niemann-Pick disease?"
print(f"Question: {query}")
print("Retrieving chunks...")
results = Retriever.search(query, top_k=3)
for i, res in enumerate(results):
    print(f"Chunk {i+1} Distance: {res['distance']:.4f}, Source: {res['metadata'].get('source_url')}")

print("\nGenerating Answer...")
answer = Generator.generate_answer(query, results)
print("\n=== SERA ANSWER ===")
print(answer)
