"""
Manual Synthesis Trigger
========================
Forces the synthesis of the next N highest-traffic clusters that haven't
crossed the HIT_COUNT_THRESHOLD yet, using the manual admin override endpoint.
"""

import sqlite3
import requests
import time
import sys
import io

# Force UTF-8 for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

BASE_URL = "http://localhost:8000"
DB_PATH = "data/sqlite/rag.db"

# Number of additional super-nodes to force generate
TARGET_COUNT = 50 

def main():
    print(f"\n🚀 Forcing manual synthesis for the next {TARGET_COUNT} clusters...\n")
    
    # 1. Connect to DB and find clusters that are NOT synthesized yet
    # meaning hit_count < 10 and not currently synthesizing
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, hit_count, canonical_query 
        FROM query_clusters 
        WHERE hit_count < 10 
          AND synthesizing = 0 
          AND synthesis_failed = 0
        ORDER BY hit_count DESC
        LIMIT ?
    """, (TARGET_COUNT,))
    
    clusters = cursor.fetchall()
    conn.close()
    
    if not clusters:
        print("No eligible clusters found to synthesize.")
        return
        
    print(f"Found {len(clusters)} clusters to synthesize.\n")
    
    success_count = 0
    
    for idx, (cluster_id, hits, query) in enumerate(clusters, 1):
        print(f"[{idx}/{len(clusters)}] Forcing Cluster {cluster_id} (hits: {hits}) | {query[:45]}...")
        
        try:
            url = f"{BASE_URL}/admin/synthesize/{cluster_id}"
            response = requests.post(url, timeout=300)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    print(f"   ✅ Success! Created Super-Node: {data.get('super_node_id')}")
                    success_count += 1
                else:
                    print(f"   ⚠️  Failed: {data.get('message')}")
            else:
                print(f"   ❌ HTTP Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            
        time.sleep(1) # Small delay to be safe
        
    print(f"\n✅ Done! Successfully force-synthesized {success_count}/{len(clusters)} new Super-Nodes.")

if __name__ == "__main__":
    main()
