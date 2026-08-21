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
import json

# Force UTF-8 for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

BASE_URL = "http://localhost:8000"
DB_PATH = "data/sqlite/rag.db"

TARGET_COUNT = 50 

def synthesize_with_retry(cluster_id, max_retries=5):
    url = f"{BASE_URL}/admin/synthesize/{cluster_id}"
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, timeout=300)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    return True, f"✅ Success! Created Super-Node: {data.get('super_node_id')}"
                else:
                    return False, f"⚠️ Failed: {data.get('message')}"
                    
            elif response.status_code == 422:
                # Validation error, no need to retry
                return False, f"❌ Validation Error: {response.text}"
                
            elif response.status_code == 500 and "Rate limit" in response.text:
                print(f"   ⏳ Rate limited! Waiting 20 seconds before retry {attempt+1}/{max_retries}...")
                time.sleep(20)
                continue
                
            else:
                return False, f"❌ HTTP Error {response.status_code}: {response.text}"
                
        except requests.exceptions.ReadTimeout:
            print(f"   ⏳ Read timeout! The server is taking a long time. Waiting 10s and retrying...")
            time.sleep(10)
            continue
        except Exception as e:
            return False, f"❌ Exception: {e}"
            
    return False, f"❌ Max retries reached for cluster {cluster_id}."

def main():
    print(f"\n🚀 Forcing manual synthesis for the next {TARGET_COUNT} clusters...\n")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Exclude the ones that already synthesized successfully.
    cursor.execute("""
        SELECT id, hit_count, canonical_query 
        FROM query_clusters 
        WHERE hit_count < 10 
          AND super_node_id IS NULL
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
        
        success, msg = synthesize_with_retry(cluster_id)
        print(f"   {msg}")
        
        if success:
            success_count += 1
            
        time.sleep(2) # Small delay to be safe
        
    print(f"\n✅ Done! Successfully force-synthesized {success_count}/{len(clusters)} new Super-Nodes.")

if __name__ == "__main__":
    main()
