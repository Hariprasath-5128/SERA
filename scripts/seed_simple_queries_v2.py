"""
seed_simple_queries_v2.py
-------------------------
Replace the contents of `benchmark_qa_simple` with a fresh set of 150
genuine SIMPLE questions (single concept, direct phrasing).

These are NOT engineered to favour either pipeline. They are ordinary
single-topic medical questions drawn from the domains SERA has ingested,
so retrieval is never empty and the benchmark reports whatever actually
happens (static or dynamic can win on any given metric).

Old rows are copied to `benchmark_qa_simple_backup_20260831` before deletion.
"""
import os
import sqlite3
import uuid

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "sqlite", "rag.db")

QUESTIONS = [
    # --- Diabetes / endocrine (13) ---
    "What are the early symptoms of type 2 diabetes?",
    "How is type 2 diabetes diagnosed?",
    "How does insulin resistance lead to diabetes?",
    "What is gestational diabetes?",
    "What are the symptoms of diabetic ketoacidosis?",
    "How is type 1 diabetes different from type 2 diabetes?",
    "What causes hypothyroidism?",
    "What are the symptoms of hyperthyroidism?",
    "How is Hashimoto's thyroiditis diagnosed?",
    "What are the symptoms of thyroid disorders?",
    "What is Cushing syndrome?",
    "What causes adrenal insufficiency?",
    "How is hyperthyroidism treated?",

    # --- Cardiovascular (19) ---
    "How is hypertension diagnosed in adults?",
    "What are the long-term effects of high blood pressure?",
    "What are the common side effects of Lisinopril?",
    "How do ACE inhibitors work in hypertension?",
    "How do beta-blockers work?",
    "What are the early warning signs of a heart attack?",
    "What are the symptoms of heart failure?",
    "What is the difference between systolic and diastolic heart failure?",
    "How is high cholesterol diagnosed?",
    "How do statins work?",
    "What are the risk factors for coronary artery disease?",
    "What are the risk factors for stroke?",
    "What is a transient ischemic attack?",
    "How is congestive heart failure treated?",
    "What causes atrial fibrillation?",
    "How does sodium intake affect blood pressure?",
    "What are the benefits of cardiovascular exercise?",
    "How can the risk of heart disease be reduced?",
    "What is the purpose of an EKG?",

    # --- Respiratory (13) ---
    "How is asthma managed in young children?",
    "What are the symptoms of childhood asthma?",
    "Can asthma be triggered by seasonal allergies?",
    "How is COPD treated?",
    "What is the difference between COPD and asthma?",
    "How does smoking affect lung capacity?",
    "What are the symptoms of lung cancer?",
    "What is idiopathic pulmonary fibrosis?",
    "What is a pulmonary embolism?",
    "How is pulmonary hypertension treated?",
    "What is sleep apnea and why is it dangerous?",
    "What is the difference between viral and bacterial pneumonia?",
    "What are the symptoms of tuberculosis?",

    # --- Renal (8) ---
    "What are the symptoms of kidney failure?",
    "What causes chronic kidney disease?",
    "What is acute kidney injury?",
    "What is nephrotic syndrome?",
    "What is the difference between hemodialysis and peritoneal dialysis?",
    "What is polycystic kidney disease?",
    "What are the symptoms of a urinary tract infection?",
    "How is a urinary tract infection treated?",

    # --- GI / hepatic (13) ---
    "What is the difference between Crohn's disease and ulcerative colitis?",
    "What causes Crohn's disease?",
    "What are the symptoms of irritable bowel syndrome?",
    "How is irritable bowel syndrome managed?",
    "What is GERD?",
    "How is celiac disease diagnosed?",
    "What is cirrhosis?",
    "How does alcohol consumption affect the liver?",
    "What is nonalcoholic fatty liver disease?",
    "How is peptic ulcer disease treated?",
    "How is hepatitis C treated?",
    "How do antibiotics affect the gut microbiome?",
    "What is lactose intolerance?",

    # --- Neurology / psychiatry (20) ---
    "What are the signs of a concussion?",
    "What is the recommended treatment for mild concussions?",
    "What are the early signs of Alzheimer's disease?",
    "How does Alzheimer's disease progress?",
    "What are the motor symptoms of Parkinson's disease?",
    "How is Parkinson's disease managed with medication?",
    "What is multiple sclerosis?",
    "What are the types of seizures in epilepsy?",
    "How is epilepsy managed long term?",
    "How is a migraine prevented and treated?",
    "What is the difference between anxiety and panic attacks?",
    "Can chronic stress cause physical illness?",
    "What is generalized anxiety disorder?",
    "How is major depression treated with antidepressants?",
    "How is post-traumatic stress disorder treated?",
    "What is bipolar disorder?",
    "How is bipolar disorder treated?",
    "What are the symptoms of schizophrenia?",
    "What are the symptoms of a stroke?",
    "How does regular exercise impact mental health?",

    # --- Autoimmune / rheumatology / dermatology (12) ---
    "What is the difference between rheumatoid arthritis and osteoarthritis?",
    "How is rheumatoid arthritis treated with biologics?",
    "What is psoriatic arthritis?",
    "What is ankylosing spondylitis?",
    "What are the symptoms of lupus?",
    "How is lupus diagnosed?",
    "What is Sjogren's syndrome?",
    "What causes psoriasis?",
    "How is psoriasis treated?",
    "How is eczema managed?",
    "What are the signs of an allergic reaction?",
    "What are the side effects of long-term corticosteroid use?",

    # --- Hematology / oncology (15) ---
    "How is iron deficiency anemia treated?",
    "What is sickle cell disease?",
    "How is sickle cell disease treated?",
    "What are the side effects of chemotherapy?",
    "What is targeted therapy in cancer?",
    "What causes non-Hodgkin lymphoma?",
    "How is acute myeloid leukemia treated?",
    "What are the risk factors for breast cancer?",
    "How is breast cancer treated?",
    "What are the symptoms of ovarian cancer?",
    "What causes prostate cancer?",
    "What is the purpose of a mammogram?",
    "What is the purpose of a colonoscopy?",
    "What are the early warning signs of colorectal cancer?",
    "How does warfarin interact with other drugs and foods?",

    # --- Infectious disease / immunity (9) ---
    "How does HIV progress to AIDS?",
    "What causes HIV and how is it transmitted?",
    "What is antiretroviral therapy?",
    "What is the difference between latent and active tuberculosis?",
    "How do vaccines work to build immunity?",
    "How does the body fight off viral infections?",
    "What are the symptoms and stages of Lyme disease?",
    "What is the difference between a cold and the flu?",
    "How does aging affect the immune system?",

    # --- Musculoskeletal / bone (8) ---
    "How is osteoporosis diagnosed and treated?",
    "What is the role of vitamin D in bone health?",
    "What is the recovery time for a hip replacement?",
    "What is the standard treatment for a torn ACL?",
    "What are the common causes of chronic back pain?",
    "How is a herniated disc in the lumbar spine treated?",
    "How is gout diagnosed and treated?",
    "How is a bone fracture treated?",

    # --- Women's health / other (5) ---
    "What is the purpose of a Pap smear?",
    "What causes uterine fibroids?",
    "What are the risk factors for an ectopic pregnancy?",
    "What is preeclampsia?",
    "What is endometriosis?",

    # --- Diagnostics / general health (9) ---
    "What is the purpose of an ultrasound?",
    "What is the difference between an MRI and a CT scan?",
    "What is the difference between a cardiologist and a neurologist?",
    "How can sleep hygiene be improved?",
    "What are the effects of sleep deprivation?",
    "How much water should a person drink each day?",
    "What are the signs of skin cancer?",
    "What are the symptoms of melanoma?",
    "How is glaucoma diagnosed?",

    # --- Genetic / rare (6) ---
    "What is Down syndrome?",
    "What is Fragile X syndrome?",
    "What is cystic fibrosis?",
    "What is Duchenne muscular dystrophy?",
    "What is Marfan syndrome?",
    "What is the treatment for Kawasaki disease in children?",
]


def main():
    assert len(QUESTIONS) == 150, f"expected 150 questions, got {len(QUESTIONS)}"
    assert len(set(QUESTIONS)) == 150, "duplicate questions detected"

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    backup = "benchmark_qa_simple_backup_20260831"
    c.execute(f"DROP TABLE IF EXISTS {backup}")
    c.execute(f"CREATE TABLE {backup} AS SELECT * FROM benchmark_qa_simple")
    n_old = c.execute(f"SELECT COUNT(*) FROM {backup}").fetchone()[0]
    print(f"Backed up {n_old} old rows -> {backup}")

    c.execute("DELETE FROM benchmark_qa_simple")
    rows = [(str(uuid.uuid4()), None, q, None, None) for q in QUESTIONS]
    c.executemany(
        "INSERT INTO benchmark_qa_simple (id, document_id, question, answer, source_url) VALUES (?,?,?,?,?)",
        rows,
    )
    conn.commit()

    n_new = c.execute("SELECT COUNT(*) FROM benchmark_qa_simple").fetchone()[0]
    print(f"Inserted {n_new} new simple questions.")
    print("First 3:")
    for r in c.execute("SELECT question FROM benchmark_qa_simple LIMIT 3"):
        print("  -", r[0])
    conn.close()


if __name__ == "__main__":
    main()
