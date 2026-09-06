import sys, os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
from app.db.chroma_client import ChromaClient
c = ChromaClient.get_super_nodes_collection()
meta = c.get(where={"type": {"$eq": "meta_node"}})
print(f"Meta Nodes: {len(meta['ids'])}")
in_hierarchy = c.get(where={"in_hierarchy": {"$eq": True}})
print(f"Super Nodes in hierarchy: {len(in_hierarchy['ids'])}")
