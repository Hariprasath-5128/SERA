import sys
import os
import json
import logging
import sqlite3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath('.'))

from app.db.chroma_client import ChromaClient
from app.validation import entity_extractor

def reset_failed_nodes():
    logger.info('Loading ChromaDB...')
    sn_collection = ChromaClient.get_super_nodes_collection()
    sn_data = sn_collection.get(include=['documents', 'metadatas'])
    ids = sn_data['ids']
    docs = sn_data['documents']
    metas = sn_data['metadatas']

    raw_collection = ChromaClient.get_collection()
    
    nodes_to_delete = []
    cluster_ids_to_reset = []
    
    logger.info(f'Scanning {len(ids)} Super Nodes against 85% rule...')
    
    for i in range(len(ids)):
        sn_id = ids[i]
        summary = docs[i]
        meta = metas[i]
        
        if meta.get('type') == 'meta_node':
            nodes_to_delete.append(sn_id)
            continue
            
        chunk_ids_str = meta.get('source_chunks', '[]')
        try:
            chunk_ids = json.loads(chunk_ids_str)
        except:
            chunk_ids = []
            
        if not chunk_ids:
            nodes_to_delete.append(sn_id)
            continue
            
        raw_data = raw_collection.get(ids=chunk_ids, include=['documents'])
        source_text = ' '.join(raw_data['documents']) if raw_data['documents'] else ''
        
        result = entity_extractor.fallback_validate(source_text, summary)
        
        if not result['passed']:
            nodes_to_delete.append(sn_id)
            if 'cluster_id' in meta and meta['cluster_id'] != -1:
                cluster_ids_to_reset.append(int(meta['cluster_id']))

    logger.info(f'Found {len(nodes_to_delete)} FAILED nodes to purge.')
    
    if nodes_to_delete:
        sn_collection.delete(ids=nodes_to_delete)
        logger.info('Deleted from ChromaDB.')
        
    if cluster_ids_to_reset:
        db_path = os.path.join(os.path.abspath('.'), 'data', 'sqlite', 'rag.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        placeholders = ','.join('?' for _ in cluster_ids_to_reset)
        c.execute(f"UPDATE query_clusters SET synthesized=0, synthesis_failed=0 WHERE id IN ({placeholders})", cluster_ids_to_reset)
        
        placeholders_sn = ','.join('?' for _ in nodes_to_delete)
        c.execute(f"DELETE FROM super_node_history WHERE super_node_id IN ({placeholders_sn})", nodes_to_delete)
        
        conn.commit()
        conn.close()
        logger.info(f'Reset {len(cluster_ids_to_reset)} clusters in SQLite.')
        
    logger.info('Done! Ready to trigger re-synthesis.')

if __name__ == '__main__':
    reset_failed_nodes()
