"""
Simulate Hit Counts & Auto-Synthesis
====================================
Artificially inflates the hit_count of clusters to trigger the natural
auto-synthesis background job, ensuring ~90% of all clusters are synthesized.
"""

import sqlite3
import sys
import io

# Force UTF-8 for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

# Add app to path if run from scripts/
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.scheduler.jobs.pattern_finder import scan_and_trigger

DB_PATH = "data/sqlite/rag.db"
TARGET_PERCENTAGE = 0.90

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Calculate how many more we need to hit 90%
    cursor.execute("SELECT count(*) FROM query_clusters")
    total_clusters = cursor.fetchone()[0]
    
    cursor.execute("SELECT count(*) FROM query_clusters WHERE super_node_id IS NOT NULL")
    synthesized_clusters = cursor.fetchone()[0]
    
    target_synthesized = int(total_clusters * TARGET_PERCENTAGE)
    needed = target_synthesized - synthesized_clusters
    
    if needed <= 0:
        print(f"✅ Already at or above {TARGET_PERCENTAGE*100}% synthesis ({synthesized_clusters}/{total_clusters}).")
        conn.close()
        return
        
    print(f"📊 Total clusters: {total_clusters}")
    print(f"✅ Currently synthesized: {synthesized_clusters}")
    print(f"🎯 Target ({TARGET_PERCENTAGE*100}%): {target_synthesized}")
    print(f"🚀 Need to auto-synthesize {needed} more clusters.")
    
    # 2. Find the top `needed` clusters that are not yet synthesized
    cursor.execute("""
        SELECT id, hit_count 
        FROM query_clusters 
        WHERE super_node_id IS NULL
          AND synthesis_failed = 0
        ORDER BY hit_count DESC
        LIMIT ?
    """, (needed,))
    
    clusters_to_boost = cursor.fetchall()
    
    if not clusters_to_boost:
        print("No eligible clusters found to boost.")
        conn.close()
        return
        
    # 3. Boost their hit_counts to exactly 15 (triggering HIT_COUNT_THRESHOLD)
    print(f"\n🔄 Simulating hits for {len(clusters_to_boost)} clusters to push them over the threshold...")
    for cluster_id, _ in clusters_to_boost:
        cursor.execute("UPDATE query_clusters SET hit_count = 15 WHERE id = ?", (cluster_id,))
    
    conn.commit()
    conn.close()
    
    print("\n✅ Database updated. Triggering the natural auto-synthesis engine (scan_and_trigger)...")
    print("⏳ Note: This will take several hours due to API rate limits. The system will pause and retry automatically.")
    
    # 4. Trigger the auto-synthesis
    # This will pull all ready clusters (hit_count >= 10) and synthesize them
    results = scan_and_trigger()
    
    print("\n🎉 Auto-Synthesis Complete!")
    print(f"Triggered: {results.get('triggered', 0)}")
    print(f"Skipped:   {results.get('skipped', 0)}")
    print(f"Failed:    {results.get('failed', 0)}")

if __name__ == "__main__":
    main()
