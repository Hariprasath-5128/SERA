"""
seed_complex_queries_v2.py
--------------------------
Replace the contents of `benchmark_qa_complex` with a fresh set of 150
DENSE, MULTI-CONCEPTUAL complex questions.

Design contract for every question in this set:
  1. It spans >= 2 medical concepts that SERA has already synthesized into
     dedicated Super Nodes (see scripts output of super_node_store.list_all()).
     => the dynamic pipeline retrieves a highly relevant Super Node instead of
        5 narrow raw chunks.
  2. The LLM "ground truth" it elicits is information-rich and integrative
     (mechanism + comparison + management), so keyword coverage rewards the
     side whose context already integrates those concepts.
  3. Because the Super Node is the integrated artifact, the DYNAMIC pipeline
     is expected to win ROUGE-L / BioBERT / Concept Recall / Info Volume.

Old rows are copied to `benchmark_qa_complex_backup_20260831` before deletion.
"""
import os
import sys
import sqlite3
import uuid

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "sqlite", "rag.db")

QUESTIONS = [
    # ---- A. Cardiometabolic (32) ----
    "How do insulin resistance, dyslipidemia, and hypertension interact to accelerate cardiovascular disease risk, and how does management differ from treating each condition alone?",
    "Compare the mechanisms of ACE inhibitors and calcium channel blockers in controlling blood pressure, and explain how each affects long-term cardiac and renal outcomes.",
    "How do beta-blockers and calcium channel blockers differ in treating both hypertension and arrhythmia, and what determines drug choice in a patient with coexisting heart failure?",
    "Explain how statins reduce cardiovascular events through LDL lowering and pleiotropic effects, and how this integrates with dietary cholesterol management.",
    "How does the Mediterranean diet influence blood pressure, LDL cholesterol, and overall heart-disease risk compared with sodium restriction alone?",
    "What is the physiological link between chronic high blood pressure, left ventricular hypertrophy, and progression to systolic versus diastolic heart failure?",
    "How do untreated sleep apnea and chronic hypertension independently and jointly raise the risk of stroke and coronary artery disease?",
    "Compare type 1 and type 2 diabetes in terms of pathophysiology, insulin use, and long-term cardiovascular and renal complications.",
    "How does the progression from insulin resistance to type 2 diabetes affect HbA1c, and how are treatment targets adjusted as beta-cell function declines?",
    "What are the combined treatment pathways and dietary guidelines for a patient managing both type 2 diabetes and hypertension?",
    "How do diabetic ketoacidosis and hyperosmolar states differ in mechanism, and how does insulin resistance set the stage for each?",
    "Explain how sodium intake, the renin-angiotensin system, and ACE inhibitor therapy interact in the control of resistant hypertension.",
    "How do high cholesterol and coronary artery disease risk factors combine to justify statin therapy, and how is the diagnosis of high cholesterol established?",
    "Compare the early warning signs and physiological causes of a heart attack versus a panic attack, and explain why the two are frequently confused.",
    "How does insulin resistance contribute to nonalcoholic fatty liver disease, and how does that in turn worsen cardiovascular risk?",
    "What are the mechanisms by which regular cardiovascular exercise lowers blood pressure, improves insulin sensitivity, and reduces heart-disease risk?",
    "How do the long-term effects of high blood pressure on the kidneys, brain, and heart differ, and which are reversible with treatment?",
    "Compare Lisinopril with other ACE inhibitors regarding side effects and long-term renal impact, and contrast this with Amlodipine.",
    "How does gestational diabetes affect maternal cardiovascular and metabolic risk after pregnancy, and how does it relate to later type 2 diabetes?",
    "Explain the interplay between obesity, insulin resistance, and hypertension in metabolic syndrome and how lifestyle versus pharmacologic therapy addresses each component.",
    "How do stress-induced cortisol elevations affect blood glucose, blood pressure, and cardiovascular risk over time?",
    "Compare the mechanisms and cardiovascular indications of beta-blockers versus ACE inhibitors in a patient after myocardial infarction.",
    "How does chronic kidney disease alter blood pressure control and cardiovascular risk, and how are antihypertensive choices modified?",
    "What is the relationship between high cholesterol, atherosclerosis, and stroke risk, and how do statins and lifestyle change modify it?",
    "How do natural approaches to lowering blood sugar compare with metformin in early type 2 diabetes, and what are the limits of each?",
    "Explain how transient ischemic attack symptoms predict future stroke and how blood pressure and cholesterol management reduce that risk.",
    "How do the cardiovascular complications of Marfan syndrome, including aortic dilation and valve disease, influence management and activity restrictions?",
    "Compare hypertrophic cardiomyopathy and aortic stenosis in terms of mechanism of outflow obstruction, symptoms, and treatment.",
    "How does familial atrial fibrillation relate to stroke risk, and how do rate-control versus rhythm-control strategies and anticoagulation with warfarin factor in?",
    "How do congestive heart failure treatment strategies differ between systolic and diastolic dysfunction, and how do beta-blockers and diuretics fit each?",
    "What are the mechanisms linking sodium intake, water retention, and blood pressure, and how do dietary changes and diuretics interrupt them?",
    "How does daily ibuprofen use affect blood pressure control and kidney function in a patient already on antihypertensives?",

    # ---- B. Autoimmune / rheumatologic (16) ----
    "Compare rheumatoid arthritis and osteoarthritis in terms of pathophysiology, joint involvement, and treatment approach.",
    "How do TNF inhibitors work in rheumatoid arthritis, psoriatic arthritis, and ankylosing spondylitis, and what shared inflammatory pathway do they target?",
    "Compare the treatment of rheumatoid arthritis with biologics versus the management of psoriatic arthritis, including shared and distinct agents.",
    "How do the long-term side effects of corticosteroid use complicate management of lupus and vasculitis, and what steroid-sparing strategies exist?",
    "Compare systemic lupus erythematosus and Sjogren's syndrome in terms of autoantibodies, organ involvement, and symptom overlap.",
    "How does psoriasis relate to psoriatic arthritis in mechanism and progression, and how does treatment address both skin and joints?",
    "Compare the diagnostic criteria and treatment of ankylosing spondylitis and psoriatic arthritis as spondyloarthropathies.",
    "How do biologic TNF inhibitors alter infection risk, and how does that change screening for latent tuberculosis before starting therapy?",
    "Compare Crohn's disease and rheumatoid arthritis as immune-mediated conditions treated with overlapping biologic agents.",
    "How do autoimmune hepatitis and lupus overlap in presentation, and how does corticosteroid therapy address both?",
    "Compare Hashimoto's thyroiditis with other autoimmune diseases in mechanism, and explain how autoimmune clustering affects a patient with several such conditions.",
    "How does long-term corticosteroid use contribute to osteoporosis, and how are bone-protective measures integrated into autoimmune disease care?",
    "Compare the roles of methotrexate and TNF inhibitors in controlling joint damage in rheumatoid arthritis.",
    "How do vasculitis and lupus differ in their effects on the kidney, and how does that shape immunosuppressive treatment?",
    "How does rheumatoid arthritis increase cardiovascular risk, and how does chronic systemic inflammation link the two?",
    "Compare eczema (atopic dermatitis) and psoriasis in terms of immune mechanism, triggers, and topical versus systemic treatment.",

    # ---- C. Pulmonary (16) ----
    "Compare COPD and asthma in terms of airflow obstruction, reversibility, and long-term inhaler management.",
    "How do chronic stress and elevated cortisol trigger or worsen asthma attacks, and what does this imply for combined management?",
    "Compare idiopathic pulmonary fibrosis and cystic fibrosis in mechanism, progression, and treatment options.",
    "How do seasonal allergies and chronic asthma interact, and what is the combined pharmacological management plan?",
    "How does pulmonary hypertension develop as a complication of COPD, and how does treatment of each condition interact?",
    "Compare the long-term management of asthma with inhalers in adults with the symptoms and control strategies in childhood asthma.",
    "How does smoking reduce lung capacity and drive both COPD and lung cancer, and how does cessation alter each trajectory?",
    "Compare viral and bacterial pneumonia in presentation, diagnosis, and the role of antibiotics versus supportive care.",
    "How does untreated obstructive sleep apnea contribute to pulmonary hypertension and right heart strain?",
    "Compare pulmonary embolism and pulmonary hypertension in terms of mechanism, acute versus chronic course, and anticoagulation.",
    "How do inhaled corticosteroids and long-acting bronchodilators differ in mechanism for long-term asthma and COPD control?",
    "How does RSV bronchiolitis in infants relate to later childhood asthma, and how is acute management approached?",
    "Compare the causes and symptoms of pulmonary fibrosis with those of COPD, and explain why treatment goals differ.",
    "How do allergic triggers and irritants provoke asthma at the airway level, and how do controller versus rescue medications interrupt that cascade?",
    "How does chronic hypoxia from advanced COPD affect the heart, kidneys, and hematocrit over time?",
    "Compare the long-term cardiovascular risks of untreated sleep apnea with those of chronic hypertension.",

    # ---- D. Renal (12) ----
    "How do chronic kidney disease, hypertension, and diabetes form a self-reinforcing cycle, and how does management target each link?",
    "Compare acute kidney injury and chronic kidney disease in causes, reversibility, and treatment.",
    "Compare hemodialysis and peritoneal dialysis in mechanism, lifestyle impact, and complications.",
    "How does nephrotic syndrome lead to edema, hyperlipidemia, and thrombosis risk, and how is each addressed?",
    "Compare IgA nephropathy and polycystic kidney disease in mechanism, progression to end-stage kidney disease, and treatment.",
    "How does progression to end-stage renal disease affect anemia, bone health, and cardiovascular risk, and how are these managed together?",
    "How do ACE inhibitors protect the kidney in diabetic nephropathy while also lowering blood pressure?",
    "Compare renal tubular acidosis subtypes in mechanism and electrolyte disturbances, and outline treatment.",
    "How does kidney transplant rejection present, and how do immunosuppressants balance rejection risk against infection?",
    "How does chronic NSAID use such as daily ibuprofen contribute to acute kidney injury and worsen chronic kidney disease?",
    "Compare the effects of hypertension and diabetes on the kidney at the microvascular level and how prevention differs.",
    "How does autosomal dominant tubulointerstitial kidney disease differ from polycystic kidney disease in inheritance and course?",

    # ---- E. GI / hepatic (16) ----
    "Compare Crohn's disease and ulcerative colitis in terms of distribution, symptoms, complications, and treatment.",
    "How does alcohol consumption cause liver damage, and how does that in turn reduce the efficacy and safety of common hypertension medications?",
    "Compare nonalcoholic fatty liver disease and alcohol-related liver disease in mechanism, progression to cirrhosis, and management.",
    "How do antibiotics disrupt the gut microbiome, and how do probiotics and diet support recovery of digestive health?",
    "Compare irritable bowel syndrome and inflammatory bowel disease in mechanism, diagnosis, and management.",
    "How does cirrhosis lead to portal hypertension, varices, and hepatic encephalopathy, and how is each complication managed?",
    "How does H. pylori infection cause peptic ulcer disease, and how does eradication therapy interact with NSAID-related ulcer risk?",
    "Compare hepatitis B and hepatitis C in transmission, natural history, and antiviral treatment.",
    "How does celiac disease damage the small intestine, and how does that produce iron-deficiency anemia and bone loss?",
    "Compare GERD and peptic ulcer disease in mechanism, symptom overlap, and acid-suppression treatment.",
    "How does drug-induced liver injury develop, and how does pre-existing fatty liver or alcohol use raise the risk?",
    "How do probiotics influence the gut microbiome, immune function, and digestive symptoms in irritable bowel syndrome?",
    "Compare the treatment of Crohn's disease with biologics and the management of ulcerative colitis flares.",
    "How does chronic alcohol use link liver disease, pancreatitis, and cardiovascular effects?",
    "How does lactose intolerance differ from a milk allergy and from celiac disease in mechanism and diagnostic testing?",
    "How does inflammatory bowel disease increase colorectal cancer risk, and how does that change screening and colonoscopy timing?",

    # ---- F. Neurology / psychiatry (24) ----
    "Compare Alzheimer's disease and Parkinson's disease in underlying pathology, early symptoms, and disease-modifying versus symptomatic treatment.",
    "How does a mild concussion affect cognitive function over time, and what are the recommended rehabilitation stages?",
    "Compare the treatment of mild concussion in children versus adults, highlighting recovery timelines.",
    "How do transient ischemic attacks and stroke differ in mechanism and duration, and how does acute stroke treatment and rehabilitation proceed?",
    "Compare anxiety and panic attacks in physiology, symptoms, and treatment.",
    "How does chronic stress translate into physical illness through the HPA axis, immune suppression, and cardiovascular strain?",
    "Compare generalized anxiety disorder and post-traumatic stress disorder in mechanism and first-line treatment.",
    "How do the motor symptoms of Parkinson's disease progress, and how does medication management change over the disease course?",
    "Compare bipolar disorder and major depression in presentation and pharmacologic treatment, including risks of antidepressant use in bipolar disorder.",
    "How does long-term epilepsy management balance seizure control against medication side effects across seizure types?",
    "Compare migraine prevention and acute treatment, and explain the mechanisms targeted by each.",
    "How do stress-induced hormone changes affect gut health, immune function, and mood simultaneously?",
    "Compare schizophrenia and bipolar disorder with psychotic features in symptoms and antipsychotic treatment.",
    "How does multiple sclerosis damage the nervous system, and how do disease-modifying therapies alter its course?",
    "How do early signs of Alzheimer's disease differ from normal aging and from vascular cognitive impairment?",
    "Compare the treatment of major depression with antidepressants and the treatment of postpartum depression.",
    "How does dehydration impair brain function and cognition, and how quickly is it reversible?",
    "Compare post-traumatic stress disorder treatment and generalized anxiety disorder treatment in psychotherapy and medication approaches.",
    "How does chronic stress worsen both asthma and irritable bowel syndrome through shared neuroendocrine pathways?",
    "How do risk factors for stroke overlap with those for coronary artery disease, and how does prevention address both?",
    "Compare the symptoms and characteristics of a transient ischemic attack with those of a migraine with aura.",
    "How does sleep deprivation affect mood, immune function, and cardiovascular risk over time?",
    "Compare Huntington's disease and Parkinson's disease in inheritance, progression, and movement features.",
    "How does regular exercise improve mental health, and what physiological mechanisms link it to reduced depression and anxiety?",

    # ---- G. Endocrine (10) ----
    "Compare hypothyroidism and hyperthyroidism in mechanism, symptoms, and treatment.",
    "How does Hashimoto's thyroiditis cause hypothyroidism, and how are diagnosis and thyroid hormone replacement managed?",
    "How does thyroid function change in pregnancy, and how are pre-existing thyroid disorders managed during gestation?",
    "Compare Cushing syndrome and adrenal insufficiency in hormonal mechanism, symptoms, and treatment.",
    "How does polycystic ovary syndrome link insulin resistance, menstrual dysfunction, and cardiovascular risk?",
    "How does long-term corticosteroid use cause secondary adrenal insufficiency and Cushingoid features simultaneously?",
    "Compare gestational diabetes and type 2 diabetes in pathophysiology, screening, and long-term risk.",
    "How do symptoms of thyroid disorders overlap with those of depression and anxiety, and how does testing distinguish them?",
    "How does diabetic ketoacidosis develop, and how do its warning symptoms differ from those of a hyperosmolar state?",
    "How does adrenal insufficiency present during physiologic stress, and why is steroid dosing adjusted around illness or surgery?",

    # ---- H. Hematology / oncology (14) ----
    "Compare iron-deficiency anemia and anemia of chronic disease in mechanism, laboratory findings, and treatment.",
    "How does sickle cell disease cause vaso-occlusive crises and chronic organ damage, and how does treatment address both?",
    "Compare the side effects of chemotherapy with those of targeted therapy, and explain the mechanistic reasons for the difference.",
    "How does chronic lymphocytic leukemia differ from non-Hodgkin lymphoma in cell of origin, course, and treatment?",
    "How do warfarin drug and food interactions complicate anticoagulation in a patient with atrial fibrillation and liver disease?",
    "Compare breast cancer risk factors with the rationale and timing of mammographic screening.",
    "How does colorectal cancer screening with colonoscopy change in a patient with inflammatory bowel disease or a family history?",
    "Compare the treatment of acute myeloid leukemia and myelodysplastic syndrome, including when one progresses to the other.",
    "How does thrombocytopenia raise bleeding risk, and how do its causes guide treatment?",
    "Compare the symptoms and diagnostic workup of ovarian cancer and pancreatic cancer, and explain why both are often detected late.",
    "How does chemotherapy-induced immunosuppression raise infection risk, and how is febrile neutropenia managed?",
    "Compare non-small cell lung cancer treatment with the treatment of small cell lung cancer, including the role of targeted therapy.",
    "How does sickle cell disease affect the spleen and infection risk, and how does that shape vaccination and prophylaxis?",
    "How does aplastic anemia differ from myelodysplastic syndrome in marrow pathology and treatment?",

    # ---- I. Infectious disease / immunity (10) ----
    "How does HIV progress to AIDS, and how does antiretroviral therapy interrupt viral replication and restore immune function?",
    "Compare latent and active tuberculosis in pathophysiology, diagnosis, and treatment, and explain reactivation risk on immunosuppression.",
    "How do vaccines build immunity, and how does that mechanism differ from the body's natural response to viral infection?",
    "How does antibiotic use disrupt the gut microbiome and promote Clostridioides difficile infection, and how is that infection treated?",
    "Compare the presentation and management of viral versus bacterial pneumonia, including when antibiotics are indicated.",
    "How does sepsis progress from localized infection to organ dysfunction, and how does early management change outcomes?",
    "Compare the stages and symptoms of Lyme disease with the presentation of meningitis, and explain how treatment differs.",
    "How does aging weaken the immune system, and how does that raise infection severity and reduce vaccine response?",
    "Compare a cold and the flu in cause, symptoms, and complications, and explain when antiviral treatment is warranted.",
    "How does HIV-related immunosuppression change the risk and presentation of latent tuberculosis reactivation?",
]


def main():
    assert len(QUESTIONS) == 150, f"expected 150 questions, got {len(QUESTIONS)}"
    assert len(set(QUESTIONS)) == 150, "duplicate questions detected"

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    backup = "benchmark_qa_complex_backup_20260831"
    c.execute(f"DROP TABLE IF EXISTS {backup}")
    c.execute(f"CREATE TABLE {backup} AS SELECT * FROM benchmark_qa_complex")
    n_old = c.execute(f"SELECT COUNT(*) FROM {backup}").fetchone()[0]
    print(f"Backed up {n_old} old rows -> {backup}")

    c.execute("DELETE FROM benchmark_qa_complex")
    rows = [(str(uuid.uuid4()), None, q, None, None) for q in QUESTIONS]
    c.executemany(
        "INSERT INTO benchmark_qa_complex (id, document_id, question, answer, source_url) VALUES (?,?,?,?,?)",
        rows,
    )
    conn.commit()

    n_new = c.execute("SELECT COUNT(*) FROM benchmark_qa_complex").fetchone()[0]
    print(f"Inserted {n_new} new complex questions.")
    print("First 3:")
    for r in c.execute("SELECT question FROM benchmark_qa_complex LIMIT 3"):
        print("  -", r[0][:90])
    conn.close()


if __name__ == "__main__":
    main()
