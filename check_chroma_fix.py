import chromadb
from app.config import CHROMA_PERSIST_PATH, CHROMA_RAW_COLLECTION

client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
try:
    collection = client.get_collection(name=CHROMA_RAW_COLLECTION)
    res = collection.get(limit=1, include=["embeddings"])
    if res is not None:
        embs = res.get("embeddings")
        if embs is not None and len(embs) > 0:
            print(f"Successfully retrieved embedding. Dimension: {len(embs[0])}")
        else:
            print("No embeddings found in the dictionary.")
    else:
        print("No results returned.")
except Exception as e:
    print(f"Error: {e}")
