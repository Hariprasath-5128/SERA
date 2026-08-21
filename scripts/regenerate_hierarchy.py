import os
import sys
import logging

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Force environment config
from dotenv import load_dotenv
env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)

from app.db.chroma_client import ChromaClient
from app.maintenance.hierarchy_merger import merge_similar_nodes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reset_and_merge():
    logger.info("Connecting to ChromaDB super_nodes collection...")
    super_coll = ChromaClient.get_super_nodes_collection()
    if not super_coll:
        logger.error("super_nodes collection not available!")
        return

    # Get all nodes
    all_nodes = super_coll.get()
    ids = all_nodes.get('ids', [])
    metadatas = all_nodes.get('metadatas', [])
    
    meta_nodes_to_delete = []
    super_nodes_to_reset_ids = []
    super_nodes_to_reset_meta = []
    
    for i, meta in zip(ids, metadatas):
        if not meta:
            continue
        if meta.get('type') == 'meta_node':
            meta_nodes_to_delete.append(i)
        elif meta.get('type') == 'super_node':
            # Reset in_hierarchy and parent_meta_id
            if meta.get('in_hierarchy') or 'parent_meta_id' in meta:
                meta['in_hierarchy'] = False
                if 'parent_meta_id' in meta:
                    del meta['parent_meta_id']
                super_nodes_to_reset_ids.append(i)
                super_nodes_to_reset_meta.append(meta)
                
    if meta_nodes_to_delete:
        logger.info(f"Deleting {len(meta_nodes_to_delete)} existing meta-nodes...")
        super_coll.delete(ids=meta_nodes_to_delete)
        
    if super_nodes_to_reset_ids:
        logger.info(f"Resetting {len(super_nodes_to_reset_ids)} super-nodes to 'in_hierarchy=False'...")
        super_coll.update(ids=super_nodes_to_reset_ids, metadatas=super_nodes_to_reset_meta)
        
    logger.info("Successfully reset the hierarchy! Starting the merge process...")
    
    # Run the merger job to rebuild meta-nodes with the new prompts
    merge_similar_nodes()
    logger.info("Merge process completed.")

if __name__ == "__main__":
    reset_and_merge()
