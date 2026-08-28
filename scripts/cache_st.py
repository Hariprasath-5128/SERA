import time
import os
from sentence_transformers import SentenceTransformer

def cache_model(model_name, max_retries=1000):
    for i in range(max_retries):
        try:
            print(f"Attempt {i+1} to load {model_name}...")
            # This will force SentenceTransformers to fetch its required files
            SentenceTransformer(model_name)
            print(f"Successfully loaded and cached {model_name}!")
            return True
        except Exception as e:
            print(f"Failed with {e}. Retrying in 5 seconds...")
            time.sleep(5)
    return False

if __name__ == "__main__":
    cache_model("BAAI/bge-m3")
