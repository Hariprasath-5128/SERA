"""
SERA Demo Population Script
============================
Fires 200+ diverse real medical queries at the live SERA API to build:
  - 200+ query log entries across 18 medical specialties
  - 80+ synthesized super-nodes
  - 15+ meta-nodes (hierarchy levels)

Run:  python -X utf8 scripts/populate_demo.py
Server must be running at http://localhost:8000
"""

import io
import requests
import time
import sys
import random

# Force UTF-8 for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_URL = "http://localhost:8000"
QUERY_URL  = f"{BASE_URL}/query/"
TRIGGER_URL = f"{BASE_URL}/simulation/trigger-synthesis"
MAINTENANCE_URL = f"{BASE_URL}/admin/trigger-maintenance"
HEALTH_URL  = f"{BASE_URL}/health"

# =============================================================================
# 200+ DIVERSE MEDICAL QUERIES across 18 specialties
# Within each specialty, queries are intentionally semantically similar so
# the clustering engine groups them and hit counts build quickly.
# =============================================================================
QUERIES_BY_SPECIALTY = {

    # ─── 1. CARDIOLOGY (12 queries) ───────────────────────────────────────────
    "Cardiology": [
        "What are the symptoms of heart failure?",
        "How is congestive heart failure treated?",
        "What causes left ventricular dysfunction in heart failure?",
        "What is the difference between systolic and diastolic heart failure?",
        "What medications are used for heart failure management?",
        "What are the risk factors for coronary artery disease?",
        "How is myocardial infarction diagnosed and treated?",
        "What is atrial fibrillation and how is it managed?",
        "What causes hypertrophic cardiomyopathy?",
        "What are the symptoms of aortic stenosis?",
        "How is pericarditis diagnosed?",
        "What is the treatment for ventricular tachycardia?",
    ],

    # ─── 2. NEUROLOGY (14 queries) ────────────────────────────────────────────
    "Neurology": [
        "What are the early signs of Alzheimer's disease?",
        "How does Alzheimer's disease progress over time?",
        "What treatments are available for Alzheimer's disease?",
        "What causes Parkinson's disease?",
        "What are the motor symptoms of Parkinson's disease?",
        "How is Parkinson's disease managed with medication?",
        "What is multiple sclerosis and what causes it?",
        "How are multiple sclerosis flare-ups treated?",
        "What are the types of seizures in epilepsy?",
        "How is epilepsy managed long-term?",
        "What causes migraine headaches?",
        "How is migraine prevented and treated?",
        "What is amyotrophic lateral sclerosis (ALS)?",
        "What are the symptoms of a transient ischemic attack?",
    ],

    # ─── 3. ENDOCRINOLOGY (12 queries) ───────────────────────────────────────
    "Endocrinology": [
        "What is the difference between Type 1 and Type 2 diabetes?",
        "What are the symptoms of Type 2 diabetes?",
        "How is Type 2 diabetes managed with medication?",
        "What causes insulin resistance in diabetes?",
        "What is HbA1c and what is a normal range?",
        "What are the symptoms of diabetic ketoacidosis?",
        "What causes hypothyroidism?",
        "What are the symptoms of an underactive thyroid?",
        "How is Hashimoto's thyroiditis diagnosed and treated?",
        "What are the symptoms of Cushing's syndrome?",
        "What causes adrenal insufficiency?",
        "What is the treatment for hyperthyroidism?",
    ],

    # ─── 4. ONCOLOGY (14 queries) ─────────────────────────────────────────────
    "Oncology": [
        "What are the risk factors for breast cancer?",
        "How is breast cancer staged and treated?",
        "What are the symptoms of lung cancer?",
        "How is non-small cell lung cancer treated?",
        "What are the early warning signs of colorectal cancer?",
        "How is colorectal cancer screened and diagnosed?",
        "What causes prostate cancer?",
        "What is the Gleason score in prostate cancer?",
        "What are the symptoms of ovarian cancer?",
        "How is pancreatic cancer diagnosed?",
        "What is the treatment for acute myeloid leukemia?",
        "What causes non-Hodgkin lymphoma?",
        "What are the symptoms of melanoma?",
        "What is the difference between benign and malignant tumors?",
    ],

    # ─── 5. PULMONOLOGY (11 queries) ──────────────────────────────────────────
    "Pulmonology": [
        "What is chronic obstructive pulmonary disease (COPD)?",
        "What causes COPD and how is it diagnosed?",
        "What are the treatments for COPD?",
        "What is the difference between COPD and asthma?",
        "What triggers an asthma attack?",
        "How is asthma managed long-term with inhalers?",
        "What is idiopathic pulmonary fibrosis?",
        "What are the symptoms and causes of pulmonary fibrosis?",
        "What are the symptoms of pulmonary embolism?",
        "How is pulmonary hypertension treated?",
        "What causes sleep apnea and how is it treated?",
    ],

    # ─── 6. GASTROENTEROLOGY (12 queries) ────────────────────────────────────
    "Gastroenterology": [
        "What causes Crohn's disease?",
        "What are the symptoms and treatment of Crohn's disease?",
        "What is the difference between Crohn's disease and ulcerative colitis?",
        "What causes irritable bowel syndrome?",
        "How is irritable bowel syndrome managed?",
        "What are the symptoms of celiac disease?",
        "How is celiac disease diagnosed?",
        "What causes gastroesophageal reflux disease?",
        "What are the long-term complications of untreated GERD?",
        "What causes liver cirrhosis?",
        "What are the symptoms of hepatic encephalopathy?",
        "How is nonalcoholic fatty liver disease treated?",
    ],

    # ─── 7. IMMUNOLOGY / AUTOIMMUNE (11 queries) ─────────────────────────────
    "Immunology": [
        "What is rheumatoid arthritis and what causes it?",
        "How is rheumatoid arthritis treated with biologics?",
        "What is systemic lupus erythematosus?",
        "What are the symptoms and diagnosis of lupus?",
        "What is ankylosing spondylitis?",
        "What is psoriatic arthritis and how is it treated?",
        "How does the immune system cause autoimmune disease?",
        "What is the role of TNF inhibitors in autoimmune diseases?",
        "What is Sjogren's syndrome?",
        "What causes vasculitis?",
        "How is primary immunodeficiency diagnosed?",
    ],

    # ─── 8. NEPHROLOGY (10 queries) ───────────────────────────────────────────
    "Nephrology": [
        "What causes chronic kidney disease?",
        "How is chronic kidney disease staged?",
        "What are the symptoms of kidney failure?",
        "What is the difference between hemodialysis and peritoneal dialysis?",
        "What causes polycystic kidney disease?",
        "What are the symptoms of nephrotic syndrome?",
        "How is IgA nephropathy diagnosed and treated?",
        "What causes acute kidney injury?",
        "What is renal tubular acidosis?",
        "How is kidney transplant rejection managed?",
    ],

    # ─── 9. HEMATOLOGY (10 queries) ───────────────────────────────────────────
    "Hematology": [
        "What causes aplastic anemia?",
        "How is iron deficiency anemia diagnosed and treated?",
        "What is chronic lymphocytic leukemia?",
        "What is the treatment for non-Hodgkin's lymphoma?",
        "What are the symptoms of hemophilia?",
        "What causes sickle cell disease?",
        "How is sickle cell disease treated?",
        "What is polycythemia vera?",
        "How is myelodysplastic syndrome treated?",
        "What causes thrombocytopenia?",
    ],

    # ─── 10. INFECTIOUS DISEASES (12 queries) ────────────────────────────────
    "Infectious Diseases": [
        "What are the symptoms of tuberculosis?",
        "How is tuberculosis spread and treated?",
        "What is the difference between latent and active tuberculosis?",
        "What causes HIV and how is it transmitted?",
        "How does HIV progress to AIDS?",
        "What antiretroviral drugs are used for HIV treatment?",
        "What are the symptoms of hepatitis B?",
        "How is hepatitis C diagnosed and treated?",
        "What is sepsis and how is it managed?",
        "What causes bacterial meningitis?",
        "What are the symptoms of Lyme disease?",
        "How is Clostridium difficile infection treated?",
    ],

    # ─── 11. GENETIC / RARE DISEASES (12 queries) ────────────────────────────
    "Genetics": [
        "What is cystic fibrosis and how is it inherited?",
        "What are the symptoms and treatment of cystic fibrosis?",
        "What is Huntington's disease and how does it progress?",
        "What is Marfan syndrome?",
        "What are the cardiovascular complications of Marfan syndrome?",
        "What is Duchenne muscular dystrophy?",
        "What causes Pompe disease?",
        "What are the symptoms of Wilson's disease?",
        "What is Fragile X syndrome?",
        "What causes phenylketonuria and how is it treated?",
        "What is neurofibromatosis type 1?",
        "What is the genetic basis of Down syndrome?",
    ],

    # ─── 12. DERMATOLOGY (10 queries) ─────────────────────────────────────────
    "Dermatology": [
        "What causes psoriasis?",
        "How is plaque psoriasis treated?",
        "What is atopic dermatitis and what triggers it?",
        "How is eczema managed in adults?",
        "What are the ABCDE criteria for melanoma?",
        "What causes acne vulgaris?",
        "How is severe acne treated with isotretinoin?",
        "What is urticaria and what causes it?",
        "What is pemphigus vulgaris?",
        "How is rosacea treated?",
    ],

    # ─── 13. PSYCHIATRY (10 queries) ──────────────────────────────────────────
    "Psychiatry": [
        "What are the diagnostic criteria for major depressive disorder?",
        "How is major depression treated with antidepressants?",
        "What is bipolar disorder and what are its types?",
        "How is bipolar disorder managed with mood stabilizers?",
        "What are the symptoms of schizophrenia?",
        "What antipsychotic medications treat schizophrenia?",
        "What causes generalized anxiety disorder?",
        "How is post-traumatic stress disorder treated?",
        "What are the symptoms of obsessive-compulsive disorder?",
        "How is attention deficit hyperactivity disorder diagnosed in adults?",
    ],

    # ─── 14. ORTHOPEDICS (9 queries) ──────────────────────────────────────────
    "Orthopedics": [
        "What causes osteoarthritis of the knee?",
        "What are the treatment options for knee osteoarthritis?",
        "What is osteoporosis and what causes bone loss?",
        "How is osteoporosis diagnosed and treated?",
        "What causes rotator cuff tears?",
        "How is a herniated disc in the lumbar spine treated?",
        "What is spinal stenosis and what are its symptoms?",
        "What causes gout and how is it treated?",
        "What is the treatment for avascular necrosis of the hip?",
    ],

    # ─── 15. OPHTHALMOLOGY (8 queries) ────────────────────────────────────────
    "Ophthalmology": [
        "What causes glaucoma and how is it diagnosed?",
        "How is open-angle glaucoma treated?",
        "What is age-related macular degeneration?",
        "How is wet macular degeneration treated?",
        "What causes cataracts?",
        "What is diabetic retinopathy?",
        "How is retinal detachment treated?",
        "What causes uveitis?",
    ],

    # ─── 16. PEDIATRICS (9 queries) ───────────────────────────────────────────
    "Pediatrics": [
        "What are the symptoms of childhood asthma?",
        "What causes febrile seizures in children?",
        "What is the treatment for pediatric type 1 diabetes?",
        "What are the developmental milestones for a 2-year-old?",
        "What causes failure to thrive in infants?",
        "What is the treatment for kawasaki disease in children?",
        "What causes autism spectrum disorder?",
        "What are the symptoms of ADHD in children?",
        "How is RSV bronchiolitis managed in infants?",
    ],

    # ─── 17. OBSTETRICS / GYNECOLOGY (9 queries) ──────────────────────────────
    "OB/GYN": [
        "What are the symptoms of preeclampsia in pregnancy?",
        "How is gestational diabetes managed?",
        "What causes polycystic ovarian syndrome?",
        "What are the symptoms of endometriosis?",
        "How is endometriosis treated?",
        "What causes uterine fibroids?",
        "What is the treatment for premature ovarian insufficiency?",
        "What are the risk factors for ectopic pregnancy?",
        "What is the management of postpartum hemorrhage?",
    ],

    # ─── 18. PHARMACOLOGY / GENERAL MEDICINE (10 queries) ────────────────────
    "Pharmacology": [
        "What are the side effects of long-term corticosteroid use?",
        "How do ACE inhibitors work in hypertension?",
        "What is the mechanism of action of statins?",
        "What are the contraindications of metformin?",
        "How do proton pump inhibitors work?",
        "What are the side effects of NSAIDs?",
        "How do beta-blockers reduce blood pressure?",
        "What drugs interact with warfarin?",
        "What is the mechanism of action of monoclonal antibodies?",
        "What causes drug-induced liver injury?",
    ],
}

# Flatten all queries into a single list
ALL_QUERIES = []
for specialty, qs in QUERIES_BY_SPECIALTY.items():
    ALL_QUERIES.extend(qs)

print(f"Total unique queries prepared: {len(ALL_QUERIES)} across {len(QUERIES_BY_SPECIALTY)} specialties")


# =============================================================================
# Helper functions
# =============================================================================

def check_health() -> bool:
    try:
        r = requests.get(HEALTH_URL, timeout=8)
        r.raise_for_status()
        print("[OK] Server is healthy and ready.")
        return True
    except Exception as e:
        print(f"[ERR] Server health check failed: {e}")
        print("      Make sure uvicorn is running: uvicorn app.main:app --reload --port 8000")
        return False


def fire_query(query: str) -> bool:
    try:
        r = requests.post(QUERY_URL, json={"query": query, "top_k": 3}, timeout=60)
        return r.status_code == 200
    except Exception:
        return False


def trigger_synthesis() -> dict:
    try:
        r = requests.post(TRIGGER_URL, timeout=300)
        r.raise_for_status()
        return r.json().get("synthesis", {})
    except Exception as e:
        print(f"  [WARN] Synthesis trigger error: {e}")
        return {}


def trigger_maintenance():
    try:
        r = requests.post(MAINTENANCE_URL, timeout=30)
        r.raise_for_status()
        print("  [OK] Maintenance pipeline triggered in background.")
    except Exception as e:
        print(f"  [WARN] Maintenance trigger error: {e}")


def run_query_round(label: str, queries: list, delay: float = 1.2):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  {len(queries)} queries | {delay}s delay each | est. {len(queries)*delay/60:.1f} min")
    print(f"{'='*65}")
    success = 0
    for i, q in enumerate(queries, 1):
        sys.stdout.write(f"\r  [{i:>3}/{len(queries)}] {q[:58]:<58}")
        sys.stdout.flush()
        if fire_query(q):
            success += 1
        time.sleep(delay)
    print(f"\n  Done: {success}/{len(queries)} queries succeeded.")
    return success


def synthesis_pass(label: str, wait_s: int = 20):
    print(f"\n[SYNTHESIS] {label}")
    result = trigger_synthesis()
    t = result.get("triggered", 0)
    s = result.get("skipped", 0)
    f = result.get("failed", 0)
    print(f"  Triggered:{t}  Skipped:{s}  Failed:{f}")
    if t > 0:
        print(f"  Waiting {wait_s}s for synthesis jobs to finish...")
        time.sleep(wait_s)
    else:
        time.sleep(3)


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n=== SERA DEMO POPULATION SCRIPT ===")
    print(f"Queries: {len(ALL_QUERIES)} | Specialties: {len(QUERIES_BY_SPECIALTY)}")

    if not check_health():
        sys.exit(1)

    # ------------------------------------------------------------------
    # ROUND 1 — Fire ALL queries once (full specialty coverage)
    # This builds the initial clustering from diverse real medical topics.
    # At 1.2s delay: ~4 min
    # ------------------------------------------------------------------
    run_query_round(
        "ROUND 1 — Full specialty sweep (all queries, all specialties)",
        ALL_QUERIES,
        delay=1.2,
    )

    # First synthesis pass — many topic clusters should already exceed threshold=10
    synthesis_pass("Pass 1 — Build first wave of super-nodes", wait_s=25)

    # ------------------------------------------------------------------
    # ROUND 2 — Shuffle and repeat a large subset
    # This pushes existing clusters past the re-synthesis delta (RESYNTH_DELTA=10)
    # creating Revision 2 super-nodes, AND finishes off any stragglers.
    # At 1.0s delay: ~3 min
    # ------------------------------------------------------------------
    round2 = random.sample(ALL_QUERIES, min(120, len(ALL_QUERIES)))
    run_query_round(
        "ROUND 2 — Second pass to build revisions and remaining super-nodes",
        round2,
        delay=1.0,
    )

    synthesis_pass("Pass 2 — Revisions and remaining clusters", wait_s=25)

    # ------------------------------------------------------------------
    # ROUND 3 — One more targeted shuffle with extra weight on each specialty
    # This makes sure every specialty has multiple synthesized super-nodes.
    # At 0.9s delay: ~2 min
    # ------------------------------------------------------------------
    # Build round 3 by taking 3 queries from every specialty for full coverage
    round3 = []
    for qs in QUERIES_BY_SPECIALTY.values():
        round3.extend(random.sample(qs, min(4, len(qs))))
    random.shuffle(round3)

    run_query_round(
        "ROUND 3 — Targeted specialty sweep (4 queries per specialty)",
        round3,
        delay=0.9,
    )

    synthesis_pass("Pass 3 — Finalizing super-nodes", wait_s=20)

    # ------------------------------------------------------------------
    # ROUND 4 — Extra weight on highest-yield clusters to push Rev 3
    # Target the 3 biggest specialties with the most semantic overlap.
    # ------------------------------------------------------------------
    high_yield = (
        QUERIES_BY_SPECIALTY["Cardiology"] +
        QUERIES_BY_SPECIALTY["Neurology"] +
        QUERIES_BY_SPECIALTY["Oncology"] +
        QUERIES_BY_SPECIALTY["Endocrinology"] +
        QUERIES_BY_SPECIALTY["Infectious Diseases"]
    )
    random.shuffle(high_yield)
    run_query_round(
        "ROUND 4 — High-yield clusters (Rev 3 push: Cardiology, Neurology, Oncology, Endo, Infectious)",
        high_yield,
        delay=0.8,
    )

    synthesis_pass("Pass 4 — Rev 3 super-nodes", wait_s=20)

    # ------------------------------------------------------------------
    # HIERARCHY MERGER — Build Meta-Nodes from similar super-nodes
    # The maintenance job runs staleness check + decay + hierarchy merger.
    # After this, the Hierarchy Graph tab should show grouped meta-nodes.
    # ------------------------------------------------------------------
    print("\n[MAINTENANCE] Running hierarchy merger to build meta-nodes...")
    trigger_maintenance()
    print("  Waiting 60s for hierarchy merger to complete in background...")
    time.sleep(60)

    # Final synthesis pass — the hierarchy merger may have flagged some
    # clusters for re-synthesis (bidirectional staleness propagation)
    synthesis_pass("Pass 5 — Post-merger cleanup", wait_s=15)

    # ------------------------------------------------------------------
    print("\n" + "="*65)
    print("  POPULATION COMPLETE!")
    print("="*65)
    print("  Open: http://localhost:8000/dashboard.html")
    print("  You should see:")
    print(f"   - {len(ALL_QUERIES)*3}+ entries in the Query Log (3 rounds x {len(ALL_QUERIES)})")
    print("   - 80+  super-nodes (Super-Nodes tab)")
    print("   - Rev 2 / Rev 3 revisions on high-traffic clusters")
    print("   - 10+  meta-nodes in the Hierarchy Graph")
    print("   - Coverage across 18 medical specialties")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
