import argparse
import sys
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import app.config as config
from app.retrieval.retriever import Retriever
from app.generation.generator import Generator
from app.db.chroma_client import ChromaClient
from app.db import super_node_store

def run_query(query, static_k=5, dynamic_k=2):
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}\n")

    # 1. RUN STATIC PIPELINE
    print(f"--- RUNNING STATIC PIPELINE (k={static_k}) ---")
    config.CHROMA_PERSIST_PATH = os.path.join(project_root, 'data', 'backups', 'v1', 'chroma')
    config.SQLITE_DB_PATH = os.path.join(project_root, 'data', 'backups', 'v1', 'rag.db')
    ChromaClient._client = None
    ChromaClient._collection = None
    ChromaClient._super_nodes_collection = None
    super_node_store._collection = None
    
    static_chunks = Retriever.search(query, top_k=static_k)
    static_ans = Generator.generate_answer(query, static_chunks)
    
    print("\n[STATIC PIPELINE ANSWER]:")
    print(static_ans)
    print(f"\n{'-'*80}\n")

    # 2. RUN DYNAMIC PIPELINE
    print(f"--- RUNNING DYNAMIC PIPELINE (Super Nodes | k={dynamic_k}) ---")
    config.CHROMA_PERSIST_PATH = os.path.join(project_root, 'data', 'chroma')
    config.SQLITE_DB_PATH = os.path.join(project_root, 'data', 'sqlite', 'rag.db')
    ChromaClient._client = None
    ChromaClient._collection = None
    ChromaClient._super_nodes_collection = None
    super_node_store._collection = None
    
    raw_dynamic = Retriever.search(query, top_k=20)
    dynamic_chunks = [c for c in raw_dynamic if c.get('type') == 'super_node'][:dynamic_k]
    if not dynamic_chunks:
        dynamic_chunks = raw_dynamic[:dynamic_k]
        
    dynamic_ans = Generator.generate_answer(query, dynamic_chunks)
    
    print("\n[DYNAMIC PIPELINE ANSWER]:")
    print(dynamic_ans)
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('query', type=str, help='The medical question to ask')
    parser.add_argument('--static_k', type=int, default=5)
    parser.add_argument('--dynamic_k', type=int, default=2)
    args = parser.parse_args()
    
    run_query(args.query, args.static_k, args.dynamic_k)
