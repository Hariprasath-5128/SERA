import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="C:/Projects/SERA/.env")
hf_token = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/",
    api_key=hf_token
)

try:
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "Say hello!"}],
        max_tokens=10
    )
    print("Success:", response.choices[0].message.content)
except Exception as e:
    print("Error:", str(e))
