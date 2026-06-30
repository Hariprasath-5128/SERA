import requests
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="C:/Projects/SERA/.env")
hf_token = os.getenv("HF_TOKEN")

url = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
headers = {"Authorization": f"Bearer {hf_token}"}
payload = {"inputs": "What is Niemann-Pick disease?"}

res = requests.post(url, headers=headers, json=payload)
print(res.status_code)
print(res.text)
