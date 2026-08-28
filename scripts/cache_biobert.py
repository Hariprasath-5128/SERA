import time
from transformers import AutoModel, AutoTokenizer

def cache_model(model_name, max_retries=1000):
    for i in range(max_retries):
        try:
            print(f"Attempt {i+1} to load {model_name}...")
            AutoTokenizer.from_pretrained(model_name)
            AutoModel.from_pretrained(model_name)
            print("Successfully loaded model and tokenizer!")
            return True
        except Exception as e:
            print(f"Failed with {e}. Retrying in 5 seconds...")
            time.sleep(5)
    return False

if __name__ == "__main__":
    cache_model("dmis-lab/biobert-base-cased-v1.1")
