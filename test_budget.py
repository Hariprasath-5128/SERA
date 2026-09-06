import sys, os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
from app.db.chroma_client import ChromaClient
r_col = ChromaClient.get_raw_chunks_collection()
s_col = ChromaClient.get_super_nodes_collection()

r_docs = r_col.get(limit=100)['documents']
s_docs = s_col.get(limit=100)['documents']

r_len = sum(len(d.split()) for d in r_docs) / len(r_docs)
s_len = sum(len(d.split()) for d in s_docs) / len(s_docs)

print(f"Raw Chunk Avg Words: {r_len:.1f}")
print(f"Super Node Avg Words: {s_len:.1f}")
print(f"Static k=5 Budget: {r_len * 5:.1f}")
print(f"Dynamic k=2 Budget: {s_len * 2:.1f}")
