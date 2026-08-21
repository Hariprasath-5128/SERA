import os
import sys
import sqlite3
import uuid

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

db_path = os.path.join(project_root, "data", "sqlite", "rag.db")

multihop_questions = [
    "What are the combined treatment pathways and dietary guidelines for a patient managing both Type 2 Diabetes and Hypertension?",
    "How do the side effects of Lisinopril compare with other ACE inhibitors, and what are the long-term renal impacts?",
    "Can chronic stress trigger asthma attacks, and how do cortisol levels link these two conditions?",
    "What is the relationship between insulin resistance, high cholesterol, and cardiovascular disease risk?",
    "How does a mild concussion affect cognitive function over time, and what are the recommended rehabilitation stages?",
    "What are the early warning signs of a heart attack versus a panic attack, and how do their physiological causes differ?",
    "How do dietary choices for managing cholesterol affect blood pressure regulation and overall heart health?",
    "What are the long-term cardiovascular risks of untreated sleep apnea compared to chronic hypertension?",
    "How do beta-blockers and calcium channel blockers differ in their mechanisms for treating high blood pressure and arrhythmia?",
    "What are the interactions between alcohol consumption, liver damage, and the efficacy of common hypertension medications?",
    "How do seasonal allergies and chronic asthma interact, and what is the combined pharmacological management plan?",
    "What are the early symptoms of Type 2 Diabetes, and how do they relate to the physiological mechanisms of insulin resistance?",
    "Compare the treatment strategies for mild concussions in children versus adults, highlighting recovery timelines.",
    "What are the common side effects of Lisinopril, and how do they compare to the side effects of Amlodipine?",
    "How do stress-induced hormone changes affect gut health and overall immune system function?"
]

def inject():
    print(f"Connecting to database at: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Backup existing benchmark questions in a backup table just in case
    print("Backing up existing benchmark_qa table...")
    cursor.execute("CREATE TABLE IF NOT EXISTS benchmark_qa_backup AS SELECT * FROM benchmark_qa")
    
    # 2. Delete all existing benchmark questions
    print("Clearing existing questions from benchmark_qa...")
    cursor.execute("DELETE FROM benchmark_qa")
    
    # 3. Insert the new multi-hop queries
    print(f"Inserting {len(multihop_questions)} multi-hop queries...")
    for q in multihop_questions:
        qa_id = str(uuid.uuid4())
        # We leave document_id and answer as None, as the evaluator generates answers dynamically
        cursor.execute(
            "INSERT INTO benchmark_qa (id, document_id, question, answer, source_url) VALUES (?, ?, ?, ?, ?)",
            (qa_id, None, q, None, None)
        )
        
    conn.commit()
    conn.close()
    print("Multi-hop queries successfully injected!")

if __name__ == "__main__":
    inject()
