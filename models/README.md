# SERA Model Weights

All active AI/ML model weights in SERA are managed dynamically through local system service caches:

- **LLM Generators & Synthesis Models (LLaMA3 / Qwen2.5):** 
  Managed locally via Ollama (`C:\Users\<User>\.ollama\models`).
  
- **Embedding & Evaluation Models (BGE-M3 / BioBERT / ROUGE):** 
  Managed locally via HuggingFace Hub / SentenceTransformers cache (`C:\Users\<User>\.cache\huggingface\hub`).

---

### Optional Offline Storage
If you wish to package offline model files directly within this repository (e.g., GGUF, ONNX, or PyTorch checkpoints), store them in the following subdirectories:

- `models/embedding/` — Custom embedding models
- `models/reranker/` — Custom reranker models
- `models/summarizer/` — Offline GGUF/bin LLM summarizer checkpoints
