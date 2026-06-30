# MedQuAD Domain & Data Analysis Report

This report analyzes the URL structures and focus topics present in the MedQuAD dataset, with a specific deep-dive into the `www.nlm.nih.gov` (MedlinePlus) domain which was discovered to have restructured its links since the dataset's creation.

## 1. Global Domain Distribution

A scan of the entire 47,441-row dataset reveals exactly 11,254 unique web pages, distributed across 9 unique NIH/CDC subdomains.

| Domain | Unique Links | Content Type |
|--------|--------------|--------------|
| **`www.nlm.nih.gov`** | 6,758 | MedlinePlus Drug & Natural Info |
| **`rarediseases.info.nih.gov`** | 2,685 | GARD Rare Diseases |
| **`ghr.nlm.nih.gov`** | 1,086 | Genetics Home Reference |
| **`www.ninds.nih.gov`** | 273 | Neurological Disorders Fact Sheets |
| **`www.niddk.nih.gov`** | 151 | Kidney, Urologic, and Endocrine Diseases |
| **`www.cancer.gov`** | 106 | National Cancer Institute Info |
| **`www.nhlbi.nih.gov`** | 88 | Heart, Lung, and Blood Topics |
| **`www.cdc.gov`** | 59 | Parasites and Infectious Diseases |
| **`nihseniorhealth.gov`** | 48 | Senior Health Topics |

---

## 2. Deep Dive: The `www.nlm.nih.gov` Domain

The vast majority of the dataset's unique links (6,758) belong to MedlinePlus. Unfortunately, MedlinePlus has since consolidated its website, and all 6,758 of these unique URLs now redirect to a single generic A-Z index page (`herb_All.html`).

Furthermore, for **31,029** of the question-answer pairs pointing to these URLs, the dataset's `answer` column is completely blank, meaning the textual context is entirely lost without the Wayback Machine.

### What data was lost?
To understand what medical knowledge was contained in these dead links, we extracted the `question_focus` (the specific medical topic) for every single MedlinePlus row in the dataset. 

There are **6,604 unique topics** covered in the MedlinePlus section. Unlike the rest of the dataset which focuses heavily on rare diseases (GARD) and genetics (GHR), the MedlinePlus data consists almost entirely of **Pharmaceutical Drugs, Supplements, and Over-the-Counter Medications**.

### Top 50 Most Frequently Asked About Drugs (MedlinePlus)

1. **Senna** (19 questions)
2. **Ibuprofen** (13 questions)
3. **Rosiglitazone** (13 questions)
4. **Naproxen** (13 questions)
5. **Codeine** (13 questions)
6. **Folic Acid** (12 questions)
7. **Methamphetamine** (12 questions)
8. **Niacin** (12 questions)
9. **Abacavir** (12 questions)
10. **Quinapril** (12 questions)
11. **Diphenhydramine** (12 questions)
12. **Guaifenesin** (12 questions)
13. **Phenylephrine** (12 questions)
14. **Morphine Oral** (12 questions)
15. **Chlorpheniramine** (12 questions)
16. **Nadolol** (12 questions)
17. **Varenicline** (12 questions)
18. **Doxylamine** (12 questions)
19. **Brompheniramine** (12 questions)
20. **Piroxicam** (12 questions)
21. **Nabumetone** (12 questions)
22. **Indomethacin** (12 questions)
23. **Diflunisal** (12 questions)
24. **Peginterferon Alfa-2b** (12 questions)
25. **Liotrix** (12 questions)
26. **Olodaterol Oral** (12 questions)
27. **Tramadol** (12 questions)
28. **Tolmetin** (12 questions)
29. **Trandolapril** (12 questions)
30. **Benazepril** (12 questions)
31. **Amitriptyline** (12 questions)
32. **Tenofovir** (12 questions)
33. **Ketoprofen** (12 questions)
34. **Azilsartan** (12 questions)
35. **Olmesartan** (12 questions)
36. **Ketorolac** (12 questions)
37. **Oxycodone** (12 questions)
38. **Sulindac** (12 questions)
39. **Pseudoephedrine** (12 questions)
40. **Promethazine** (12 questions)
41. **Olanzapine** (12 questions)
42. **Formoterol Oral** (12 questions)
43. **Losartan** (12 questions)
44. **Oxaprozin** (12 questions)
45. **Atenolol** (12 questions)
46. **Propranolol** (12 questions)
47. **Ferrous Sulfate** (12 questions)
48. **Fenoprofen** (12 questions)
49. **Isoniazid** (12 questions)
50. **Flurbiprofen** (12 questions)

> [!WARNING]
> If we choose to drop the blank/dead rows rather than using the Wayback Machine, our RAG system will be highly knowledgeable about Rare Diseases and Genetics, but will be missing detailed pharmacology and drug-interaction data for thousands of common medications like Ibuprofen, Oxycodone, and Codeine.


## 3. Raw Dataset Samples per Domain (YAML)

Here is exactly what a raw row from the HuggingFace dataset looks like for each of the major domains (10 samples each), formatted with proper YAML spacing.

### Domain: www.nlm.nih.gov (6758 unique links)
```yaml
- document_id: 0000029
  document_url: https://www.nlm.nih.gov/medlineplus/druginfo/natural/1261.html
  question_focus: Coconut Water
  question_type: information
  question: What is Coconut Water ?
  answer_snippet: null

- document_id: 0000029
  document_url: https://www.nlm.nih.gov/medlineplus/druginfo/natural/1261.html
  question_focus: Coconut Water
  question_type: how effective is it
  question: How effective is Coconut Water ?
  answer_snippet: null

- document_id: 0000029
  document_url: https://www.nlm.nih.gov/medlineplus/druginfo/natural/1261.html
  question_focus: Coconut Water
  question_type: how does it work
  question: What is the action of Coconut Water and how does it work ?
  answer_snippet: null

- document_id: 0000029
  document_url: https://www.nlm.nih.gov/medlineplus/druginfo/natural/1261.html
  question_focus: Coconut Water
  question_type: precautions
  question: Are there safety concerns or special precautions about Coconut Water ?
  answer_snippet: null

- document_id: 0000029
  document_url: https://www.nlm.nih.gov/medlineplus/druginfo/natural/1261.html
  question_focus: Coconut Water
  question_type: interactions with medications
  question: Are there interactions between Coconut Water and other medications ?
  answer_snippet: null

- document_id: 0000029
  document_url: https://www.nlm.nih.gov/medlineplus/druginfo/natural/1261.html
  question_focus: Coconut Water
  question_type: interactions with herbs and supplements
  question: Are there interactions between Coconut Water and herbs and supplements
    ?
  answer_snippet: null

- document_id: 0000029
  document_url: https://www.nlm.nih.gov/medlineplus/druginfo/natural/1261.html
  question_focus: Coconut Water
  question_type: interactions with foods
  question: Are there interactions between Coconut Water and foods ?
  answer_snippet: null

- document_id: 0000029
  document_url: https://www.nlm.nih.gov/medlineplus/druginfo/natural/1261.html
  question_focus: Coconut Water
  question_type: dose
  question: What is the dosage of Coconut Water ?
  answer_snippet: null

- document_id: '0000001'
  document_url: https://www.nlm.nih.gov/medlineplus/druginfo/natural/269.html
  question_focus: Activated Charcoal
  question_type: information
  question: What is Activated Charcoal ?
  answer_snippet: null

- document_id: '0000001'
  document_url: https://www.nlm.nih.gov/medlineplus/druginfo/natural/269.html
  question_focus: Activated Charcoal
  question_type: how effective is it
  question: How effective is Activated Charcoal ?
  answer_snippet: null
```

### Domain: rarediseases.info.nih.gov (2685 unique links)
```yaml
- document_id: 0003093
  document_url: https://rarediseases.info.nih.gov/gard/2932/hypothalamic-dysfunction
  question_focus: Hypothalamic dysfunction
  question_type: information
  question: What is (are) Hypothalamic dysfunction ?
  answer_snippet: Hypothalamic dysfunction refers to a condition in which the hypothalamus
    is not working properly. The hypothalamus produces hormones that control body...

- document_id: 0003093
  document_url: https://rarediseases.info.nih.gov/gard/2932/hypothalamic-dysfunction
  question_focus: Hypothalamic dysfunction
  question_type: symptoms
  question: What are the symptoms of Hypothalamic dysfunction ?
  answer_snippet: What are the signs and symptoms of hypothalamic dysfunction? The
    signs and symptoms of hypothalamic dysfunction may vary from person to person
    dependi...

- document_id: 0003093
  document_url: https://rarediseases.info.nih.gov/gard/2932/hypothalamic-dysfunction
  question_focus: Hypothalamic dysfunction
  question_type: causes
  question: What causes Hypothalamic dysfunction ?
  answer_snippet: 'What causes hypothalamic dysfunction? Hypothalamic dysfunction
    may be caused by any of the following : Birth defects of the brain or hypothalamus
    (e.g...'

- document_id: 0003093
  document_url: https://rarediseases.info.nih.gov/gard/2932/hypothalamic-dysfunction
  question_focus: Hypothalamic dysfunction
  question_type: treatment
  question: What are the treatments for Hypothalamic dysfunction ?
  answer_snippet: How might hypothalamic dysfunction be treated? Treatment is based
    on the specific cause of the hypothalamic dysfunction. For instance, if the conditio...

- document_id: 0003939
  document_url: https://rarediseases.info.nih.gov/gard/4119/mental-retardation-x-linked-syndromic-11
  question_focus: Mental retardation X-linked syndromic 11
  question_type: symptoms
  question: What are the symptoms of Mental retardation X-linked syndromic 11 ?
  answer_snippet: What are the signs and symptoms of Mental retardation X-linked syndromic
    11? The Human Phenotype Ontology provides the following list of signs and sym...

- document_id: 0001848
  document_url: https://rarediseases.info.nih.gov/gard/12686/diffuse-cutaneous-mastocytosis
  question_focus: Diffuse cutaneous mastocytosis
  question_type: symptoms
  question: What are the symptoms of Diffuse cutaneous mastocytosis ?
  answer_snippet: What are the signs and symptoms of Diffuse cutaneous mastocytosis?
    The Human Phenotype Ontology provides the following list of signs and symptoms
    for ...

- document_id: 0001690
  document_url: https://rarediseases.info.nih.gov/gard/12867/cushing-disease
  question_focus: Cushing disease
  question_type: information
  question: What is (are) Cushing disease ?
  answer_snippet: Cushing disease is a condition caused by elevated levels of a hormone
    called cortisol. It is part of a group of diseases known as Cushings syndrome.
    T...

- document_id: 0001690
  document_url: https://rarediseases.info.nih.gov/gard/12867/cushing-disease
  question_focus: Cushing disease
  question_type: symptoms
  question: What are the symptoms of Cushing disease ?
  answer_snippet: What are the signs and symptoms of Cushing disease? The Human Phenotype
    Ontology provides the following list of signs and symptoms for Cushing disease...

- document_id: 0002399
  document_url: https://rarediseases.info.nih.gov/gard/6457/focal-dermal-hypoplasia
  question_focus: Focal dermal hypoplasia
  question_type: information
  question: What is (are) Focal dermal hypoplasia ?
  answer_snippet: Focal dermal hypoplasia is a genetic disorder that primarily affects
    the skin, skeleton, eyes, and face. The skin abnormalities are present from birth...

- document_id: 0002399
  document_url: https://rarediseases.info.nih.gov/gard/6457/focal-dermal-hypoplasia
  question_focus: Focal dermal hypoplasia
  question_type: symptoms
  question: What are the symptoms of Focal dermal hypoplasia ?
  answer_snippet: What are the signs and symptoms of Focal dermal hypoplasia? Focal
    dermal hypoplasia is usually evident from birth and primarily affects the skin,
    skel...
```

### Domain: ghr.nlm.nih.gov (1086 unique links)
```yaml
- document_id: 0000559
  document_url: https://ghr.nlm.nih.gov/condition/keratoderma-with-woolly-hair
  question_focus: keratoderma with woolly hair
  question_type: information
  question: What is (are) keratoderma with woolly hair ?
  answer_snippet: Keratoderma with woolly hair is a group of related conditions that
    affect the skin and hair and in many cases increase the risk of potentially life-th...

- document_id: 0000559
  document_url: https://ghr.nlm.nih.gov/condition/keratoderma-with-woolly-hair
  question_focus: keratoderma with woolly hair
  question_type: frequency
  question: How many people are affected by keratoderma with woolly hair ?
  answer_snippet: Keratoderma with woolly hair is rare; its prevalence worldwide is
    unknown.  Type I (Naxos disease) was first described in families from the Greek
    isla...

- document_id: 0000559
  document_url: https://ghr.nlm.nih.gov/condition/keratoderma-with-woolly-hair
  question_focus: keratoderma with woolly hair
  question_type: genetic changes
  question: What are the genetic changes related to keratoderma with woolly hair ?
  answer_snippet: Mutations in the JUP, DSP, DSC2, and KANK2 genes cause keratoderma
    with woolly hair types I through IV, respectively. The JUP, DSP, and DSC2 genes
    pro...

- document_id: 0000559
  document_url: https://ghr.nlm.nih.gov/condition/keratoderma-with-woolly-hair
  question_focus: keratoderma with woolly hair
  question_type: inheritance
  question: Is keratoderma with woolly hair inherited ?
  answer_snippet: Most cases of keratoderma with woolly hair have an autosomal recessive
    pattern of inheritance, which means both copies of the gene in each cell have
    m...

- document_id: 0000559
  document_url: https://ghr.nlm.nih.gov/condition/keratoderma-with-woolly-hair
  question_focus: keratoderma with woolly hair
  question_type: treatment
  question: What are the treatments for keratoderma with woolly hair ?
  answer_snippet: 'These resources address the diagnosis or management of keratoderma
    with woolly hair:  - Gene Review: Gene Review: Arrhythmogenic Right Ventricular
    Dys...'

- document_id: '0000565'
  document_url: https://ghr.nlm.nih.gov/condition/knobloch-syndrome
  question_focus: Knobloch syndrome
  question_type: information
  question: What is (are) Knobloch syndrome ?
  answer_snippet: Knobloch syndrome is a rare condition characterized by severe vision
    problems and a skull defect.  A characteristic feature of Knobloch syndrome is
    ex...

- document_id: '0000565'
  document_url: https://ghr.nlm.nih.gov/condition/knobloch-syndrome
  question_focus: Knobloch syndrome
  question_type: frequency
  question: How many people are affected by Knobloch syndrome ?
  answer_snippet: Knobloch syndrome is a rare condition. However, the exact prevalence
    of the condition is unknown....

- document_id: '0000565'
  document_url: https://ghr.nlm.nih.gov/condition/knobloch-syndrome
  question_focus: Knobloch syndrome
  question_type: genetic changes
  question: What are the genetic changes related to Knobloch syndrome ?
  answer_snippet: Mutations in the COL18A1 gene can cause Knobloch syndrome. The COL18A1
    gene provides instructions for making a protein that forms collagen XVIII, whic...

- document_id: '0000565'
  document_url: https://ghr.nlm.nih.gov/condition/knobloch-syndrome
  question_focus: Knobloch syndrome
  question_type: inheritance
  question: Is Knobloch syndrome inherited ?
  answer_snippet: This condition is inherited in an autosomal recessive pattern, which
    means both copies of the gene in each cell have mutations. The parents of an indi...

- document_id: '0000565'
  document_url: https://ghr.nlm.nih.gov/condition/knobloch-syndrome
  question_focus: Knobloch syndrome
  question_type: treatment
  question: What are the treatments for Knobloch syndrome ?
  answer_snippet: 'These resources address the diagnosis or management of Knobloch
    syndrome:  - American Academy of Ophthalmology: Eye Smart  - Genetic Testing Registry:...'
```

### Domain: www.ninds.nih.gov (273 unique links)
```yaml
- document_id: '0000203'
  document_url: http://www.ninds.nih.gov/disorders/msa/msa.htm
  question_focus: Multiple System Atrophy
  question_type: information
  question: What is (are) Multiple System Atrophy ?
  answer_snippet: Multiple system atrophy (MSA) is a progressive neurodegenerative
    disorder characterized by symptoms of autonomic nervous system failure such as
    fainti...

- document_id: '0000203'
  document_url: http://www.ninds.nih.gov/disorders/msa/msa.htm
  question_focus: Multiple System Atrophy
  question_type: treatment
  question: What are the treatments for Multiple System Atrophy ?
  answer_snippet: There is no cure for MSA. Currently, there are no treatments to
    delay the progress of neurodegeneration in the brain. But there are treatments
    availab...

- document_id: '0000203'
  document_url: http://www.ninds.nih.gov/disorders/msa/msa.htm
  question_focus: Multiple System Atrophy
  question_type: outlook
  question: What is the outlook for Multiple System Atrophy ?
  answer_snippet: The disease tends to advance rapidly over the course of 5 to 10
    years, with progressive loss of motor skills, eventual confinement to bed, and
    death. ...

- document_id: '0000203'
  document_url: http://www.ninds.nih.gov/disorders/msa/msa.htm
  question_focus: Multiple System Atrophy
  question_type: research
  question: what research (or clinical trials) is being done for Multiple System Atrophy
    ?
  answer_snippet: The NINDS supports research about MSA through grants to major medical
    institutions across the country. Researchers hope to learn why alpha-synuclein
    b...

- document_id: '0000217'
  document_url: http://www.ninds.nih.gov/disorders/neurotoxicity/neurotoxicity.htm
  question_focus: Neurotoxicity
  question_type: information
  question: What is (are) Neurotoxicity ?
  answer_snippet: Neurotoxicity occurs when the exposure to natural or manmade toxic
    substances (neurotoxicants) alters the normal activity of the nervous system.
    This ...

- document_id: '0000217'
  document_url: http://www.ninds.nih.gov/disorders/neurotoxicity/neurotoxicity.htm
  question_focus: Neurotoxicity
  question_type: treatment
  question: What are the treatments for Neurotoxicity ?
  answer_snippet: Treatment involves eliminating or reducing exposure to the toxic
    substance, followed by symptomatic and supportive therapy....

- document_id: '0000217'
  document_url: http://www.ninds.nih.gov/disorders/neurotoxicity/neurotoxicity.htm
  question_focus: Neurotoxicity
  question_type: outlook
  question: What is the outlook for Neurotoxicity ?
  answer_snippet: The prognosis depends upon the length and degree of exposure and
    the severity of neurological injury. In some instances, exposure to neurotoxicants
    ca...

- document_id: '0000217'
  document_url: http://www.ninds.nih.gov/disorders/neurotoxicity/neurotoxicity.htm
  question_focus: Neurotoxicity
  question_type: research
  question: what research (or clinical trials) is being done for Neurotoxicity ?
  answer_snippet: The NINDS supports research on disorders of the brain and nervous
    system such as neurotoxicity, aimed at learning more about these disorders and
    findi...

- document_id: 0000029
  document_url: http://www.ninds.nih.gov/disorders/chiari/chiari.htm
  question_focus: Chiari Malformation
  question_type: information
  question: What is (are) Chiari Malformation ?
  answer_snippet: Chiari malformations (CMs) are structural defects in the cerebellum,
    the part of the brain that controls balance. When the indented bony space at the
    ...

- document_id: 0000029
  document_url: http://www.ninds.nih.gov/disorders/chiari/chiari.htm
  question_focus: Chiari Malformation
  question_type: treatment
  question: What are the treatments for Chiari Malformation ?
  answer_snippet: Medications may ease certain symptoms, such as pain. Surgery is
    the only treatment available to correct functional disturbances or halt the progressio...
```

### Domain: www.niddk.nih.gov (151 unique links)
```yaml
- document_id: '0000203'
  document_url: http://www.niddk.nih.gov/health-information/health-topics/urologic-disease/kidney-stones-in-adults/Pages/facts.aspx
  question_focus: Kidney Stones in Adults
  question_type: information
  question: What is (are) Kidney Stones in Adults ?
  answer_snippet: A kidney stone is a solid piece of material that forms in a kidney
    when substances that are normally found in the urine become highly concentrated.
    A ...

- document_id: '0000203'
  document_url: http://www.niddk.nih.gov/health-information/health-topics/urologic-disease/kidney-stones-in-adults/Pages/facts.aspx
  question_focus: Kidney Stones in Adults
  question_type: information
  question: What is (are) Kidney Stones in Adults ?
  answer_snippet: The urinary tract is the bodys drainage system for removing wastes
    and extra water. The urinary tract includes two kidneys, two ureters, a bladder,
    an...

- document_id: '0000203'
  document_url: http://www.niddk.nih.gov/health-information/health-topics/urologic-disease/kidney-stones-in-adults/Pages/facts.aspx
  question_focus: Kidney Stones in Adults
  question_type: susceptibility
  question: Who is at risk for Kidney Stones in Adults? ?
  answer_snippet: Anyone can get a kidney stone, but some people are more likely to
    get one. Men are affected more often than women, and kidney stones are more common
    i...

- document_id: '0000203'
  document_url: http://www.niddk.nih.gov/health-information/health-topics/urologic-disease/kidney-stones-in-adults/Pages/facts.aspx
  question_focus: Kidney Stones in Adults
  question_type: causes
  question: What causes Kidney Stones in Adults ?
  answer_snippet: Kidney stones can form when substances in the urinesuch as calcium,
    oxalate, and phosphorusbecome highly concentrated. Certain foods may promote stone...

- document_id: '0000203'
  document_url: http://www.niddk.nih.gov/health-information/health-topics/urologic-disease/kidney-stones-in-adults/Pages/facts.aspx
  question_focus: Kidney Stones in Adults
  question_type: information
  question: What is (are) Kidney Stones in Adults ?
  answer_snippet: "Four major types of kidney stones can form:\n                \n\
    - Calcium stones are the most common type of kidney stone and occur in two major\
    \ forms: c..."

- document_id: '0000203'
  document_url: http://www.niddk.nih.gov/health-information/health-topics/urologic-disease/kidney-stones-in-adults/Pages/facts.aspx
  question_focus: Kidney Stones in Adults
  question_type: symptoms
  question: What are the symptoms of Kidney Stones in Adults ?
  answer_snippet: People with kidney stones may have pain while urinating, see blood
    in the urine, or feel a sharp pain in the back or lower abdomen. The pain may
    last ...

- document_id: '0000203'
  document_url: http://www.niddk.nih.gov/health-information/health-topics/urologic-disease/kidney-stones-in-adults/Pages/facts.aspx
  question_focus: Kidney Stones in Adults
  question_type: exams and tests
  question: How to diagnose Kidney Stones in Adults ?
  answer_snippet: To diagnose kidney stones, the health care provider will perform
    a physical exam and take a medical history. The medical history may include questions...

- document_id: '0000203'
  document_url: http://www.niddk.nih.gov/health-information/health-topics/urologic-disease/kidney-stones-in-adults/Pages/facts.aspx
  question_focus: Kidney Stones in Adults
  question_type: treatment
  question: What are the treatments for Kidney Stones in Adults ?
  answer_snippet: Treatment for kidney stones usually depends on their size and what
    they are made of, as well as whether they are causing pain or obstructing the
    urina...

- document_id: '0000203'
  document_url: http://www.niddk.nih.gov/health-information/health-topics/urologic-disease/kidney-stones-in-adults/Pages/facts.aspx
  question_focus: Kidney Stones in Adults
  question_type: prevention
  question: How to prevent Kidney Stones in Adults ?
  answer_snippet: The first step in preventing kidney stones is to understand what
    is causing the stones to form. The health care provider may ask the person to
    try to ...

- document_id: '0000203'
  document_url: http://www.niddk.nih.gov/health-information/health-topics/urologic-disease/kidney-stones-in-adults/Pages/facts.aspx
  question_focus: Kidney Stones in Adults
  question_type: considerations
  question: What to do for Kidney Stones in Adults ?
  answer_snippet: '- A kidney stone is a solid piece of material that forms in a kidney
    when substances that are normally found in the urine become highly concentrated.
    ...'
```

### Domain: www.cancer.gov (106 unique links)
```yaml
- document_id: '0000032_1'
  document_url: https://www.cancer.gov/types/lung/patient/non-small-cell-lung-treatment-pdq
  question_focus: Non-Small Cell Lung Cancer
  question_type: information
  question: What is (are) Non-Small Cell Lung Cancer ?
  answer_snippet: "Key Points\n                    - Non-small cell lung cancer is\
    \ a disease in which malignant (cancer) cells form in the tissues of the lung.\
    \     - Ther..."

- document_id: '0000032_1'
  document_url: https://www.cancer.gov/types/lung/patient/non-small-cell-lung-treatment-pdq
  question_focus: Non-Small Cell Lung Cancer
  question_type: susceptibility
  question: Who is at risk for Non-Small Cell Lung Cancer? ?
  answer_snippet: Smoking is the major risk factor for non-small cell lung cancer.
    Anything that increases your chance of getting a disease is called a risk factor.
    Hav...

- document_id: '0000032_1'
  document_url: https://www.cancer.gov/types/lung/patient/non-small-cell-lung-treatment-pdq
  question_focus: Non-Small Cell Lung Cancer
  question_type: symptoms
  question: What are the symptoms of Non-Small Cell Lung Cancer ?
  answer_snippet: Signs of non-small cell lung cancer include a cough that doesn't
    go away and shortness of breath. Sometimes lung cancer does not cause any signs
    or sy...

- document_id: '0000032_1'
  document_url: https://www.cancer.gov/types/lung/patient/non-small-cell-lung-treatment-pdq
  question_focus: Non-Small Cell Lung Cancer
  question_type: exams and tests
  question: How to diagnose Non-Small Cell Lung Cancer ?
  answer_snippet: Tests that examine the lungs are used to detect (find), diagnose,
    and stage non-small cell lung cancer. Tests and procedures to detect, diagnose,
    and ...

- document_id: '0000032_1'
  document_url: https://www.cancer.gov/types/lung/patient/non-small-cell-lung-treatment-pdq
  question_focus: Non-Small Cell Lung Cancer
  question_type: outlook
  question: What is the outlook for Non-Small Cell Lung Cancer ?
  answer_snippet: Certain factors affect prognosis (chance of recovery) and treatment
    options. The prognosis (chance of recovery) and treatment options depend on the
    fo...

- document_id: '0000032_1'
  document_url: https://www.cancer.gov/types/lung/patient/non-small-cell-lung-treatment-pdq
  question_focus: Non-Small Cell Lung Cancer
  question_type: stages
  question: What are the stages of Non-Small Cell Lung Cancer ?
  answer_snippet: "Key Points\n                    - After lung cancer has been diagnosed,\
    \ tests are done to find out if cancer cells have spread within the lungs or to\
    \ o..."

- document_id: '0000032_1'
  document_url: https://www.cancer.gov/types/lung/patient/non-small-cell-lung-treatment-pdq
  question_focus: Non-Small Cell Lung Cancer
  question_type: treatment
  question: What are the treatments for Non-Small Cell Lung Cancer ?
  answer_snippet: "Key Points\n                    - There are different types of\
    \ treatment for patients with non-small cell lung cancer.    - Nine types of standard\
    \ trea..."

- document_id: '0000032_1'
  document_url: https://www.cancer.gov/types/lung/patient/non-small-cell-lung-treatment-pdq
  question_focus: Non-Small Cell Lung Cancer
  question_type: research
  question: what research (or clinical trials) is being done for Non-Small Cell Lung
    Cancer ?
  answer_snippet: "New types of treatment are being tested in clinical trials.\n \
    \                   This summary section describes treatments that are being studied\
    \ in cl..."

- document_id: '0000014_2'
  document_url: https://www.cancer.gov/types/uterine/patient/uterine-sarcoma-treatment-pdq
  question_focus: Uterine Sarcoma
  question_type: information
  question: What is (are) Uterine Sarcoma ?
  answer_snippet: "Key Points\n                    - Uterine sarcoma is a disease\
    \ in which malignant (cancer) cells form in the muscles of the uterus or other\
    \ tissues tha..."

- document_id: '0000014_2'
  document_url: https://www.cancer.gov/types/uterine/patient/uterine-sarcoma-treatment-pdq
  question_focus: Uterine Sarcoma
  question_type: susceptibility
  question: Who is at risk for Uterine Sarcoma? ?
  answer_snippet: Being exposed to x-rays can increase the risk of uterine sarcoma.
    Anything that increases your risk of getting a disease is called a risk factor.
    Havi...
```

### Domain: www.nhlbi.nih.gov (88 unique links)
```yaml
- document_id: 0000029
  document_url: http://www.nhlbi.nih.gov/health/health-topics/topics/cm
  question_focus: Cardiomyopathy
  question_type: information
  question: What is (are) Cardiomyopathy ?
  answer_snippet: "Cardiomyopathy refers to diseases of the heart muscle. These diseases\
    \ have many causes, signs and symptoms, and treatments.\n                \nIn\
    \ cardio..."

- document_id: 0000029
  document_url: http://www.nhlbi.nih.gov/health/health-topics/topics/cm
  question_focus: Cardiomyopathy
  question_type: causes
  question: What causes Cardiomyopathy ?
  answer_snippet: Cardiomyopathy can be acquired or inherited. Acquired means you
    arent born with the disease, but you develop it due to another disease, condition,
    or ...

- document_id: 0000029
  document_url: http://www.nhlbi.nih.gov/health/health-topics/topics/cm
  question_focus: Cardiomyopathy
  question_type: susceptibility
  question: Who is at risk for Cardiomyopathy? ?
  answer_snippet: "People of all ages and races can have cardiomyopathy. However,\
    \ certain types of the disease are more common in certain groups.\n          \
    \      \nDilate..."

- document_id: 0000029
  document_url: http://www.nhlbi.nih.gov/health/health-topics/topics/cm
  question_focus: Cardiomyopathy
  question_type: symptoms
  question: What are the symptoms of Cardiomyopathy ?
  answer_snippet: "Some people who have cardiomyopathy never have signs or symptoms.\
    \ Others don't have signs or symptoms in the early stages of the disease.\n  \
    \          ..."

- document_id: 0000029
  document_url: http://www.nhlbi.nih.gov/health/health-topics/topics/cm
  question_focus: Cardiomyopathy
  question_type: exams and tests
  question: How to diagnose Cardiomyopathy ?
  answer_snippet: "Your doctor will diagnose cardiomyopathy based on your medical\
    \ and family histories, a physical exam, and the results from tests and procedures.\n\
    \     ..."

- document_id: 0000029
  document_url: http://www.nhlbi.nih.gov/health/health-topics/topics/cm
  question_focus: Cardiomyopathy
  question_type: treatment
  question: What are the treatments for Cardiomyopathy ?
  answer_snippet: People who have cardiomyopathy but no signs or symptoms may not
    need treatment. Sometimes, dilated cardiomyopathy that comes on suddenly may go
    away o...

- document_id: 0000029
  document_url: http://www.nhlbi.nih.gov/health/health-topics/topics/cm
  question_focus: Cardiomyopathy
  question_type: prevention
  question: How to prevent Cardiomyopathy ?
  answer_snippet: You can't prevent inherited types of cardiomyopathy. However, you
    can take steps to lower your risk for diseases or conditions that may lead to
    or com...

- document_id: '0000001'
  document_url: http://www.nhlbi.nih.gov/health/health-topics/topics/aat
  question_focus: Alpha-1 Antitrypsin Deficiency
  question_type: information
  question: What is (are) Alpha-1 Antitrypsin Deficiency ?
  answer_snippet: Alpha-1 antitrypsin (an-tee-TRIP-sin) deficiency, or AAT deficiency,
    is a condition that raises your risk for lung disease (especially if you smoke)
    a...

- document_id: '0000001'
  document_url: http://www.nhlbi.nih.gov/health/health-topics/topics/aat
  question_focus: Alpha-1 Antitrypsin Deficiency
  question_type: causes
  question: What causes Alpha-1 Antitrypsin Deficiency ?
  answer_snippet: "Alpha-1 antitrypsin (AAT) deficiency is an inherited disease. \"\
    Inherited\" means it's passed from parents to children through genes.\n      \
    \          \nC..."

- document_id: '0000001'
  document_url: http://www.nhlbi.nih.gov/health/health-topics/topics/aat
  question_focus: Alpha-1 Antitrypsin Deficiency
  question_type: susceptibility
  question: Who is at risk for Alpha-1 Antitrypsin Deficiency? ?
  answer_snippet: "Alpha-1 antitrypsin (AAT) deficiency occurs in all ethnic groups.\
    \ However, the condition occurs most often in White people of European descent.\n\
    \      ..."
```

### Domain: www.cdc.gov (59 unique links)
```yaml
- document_id: '0000001'
  document_url: http://www.cdc.gov/parasites/acanthamoeba/
  question_focus: Acanthamoeba - Granulomatous Amebic Encephalitis (GAE); Keratitis
  question_type: information
  question: What is (are) Acanthamoeba - Granulomatous Amebic Encephalitis (GAE);
    Keratitis ?
  answer_snippet: Acanthamoeba is a microscopic, free-living ameba (single-celled
    living organism) commonly found in the environment that can cause rare, but severe,
    il...

- document_id: '0000001'
  document_url: http://www.cdc.gov/parasites/acanthamoeba/
  question_focus: Acanthamoeba - Granulomatous Amebic Encephalitis (GAE); Keratitis
  question_type: susceptibility
  question: Who is at risk for Acanthamoeba - Granulomatous Amebic Encephalitis (GAE);
    Keratitis? ?
  answer_snippet: "Acanthamoeba keratitis\n  \n   \nAcanthamoeba keratitis is a rare\
    \ disease that can affect anyone, but is most common in individuals who wear contact\
    \ lens..."

- document_id: '0000001'
  document_url: http://www.cdc.gov/parasites/acanthamoeba/
  question_focus: Acanthamoeba - Granulomatous Amebic Encephalitis (GAE); Keratitis
  question_type: exams and tests
  question: How to diagnose Acanthamoeba - Granulomatous Amebic Encephalitis (GAE);
    Keratitis ?
  answer_snippet: Early diagnosis is essential for effective treatment of Acanthamoeba
    keratitis. The infection is usually diagnosed by an eye specialist based on sympt...

- document_id: '0000001'
  document_url: http://www.cdc.gov/parasites/acanthamoeba/
  question_focus: Acanthamoeba - Granulomatous Amebic Encephalitis (GAE); Keratitis
  question_type: treatment
  question: What are the treatments for Acanthamoeba - Granulomatous Amebic Encephalitis
    (GAE); Keratitis ?
  answer_snippet: Early diagnosis is essential for effective treatment of Acanthamoeba
    keratitis. Several prescription eye medications are available for treatment. Howe...

- document_id: '0000001'
  document_url: http://www.cdc.gov/parasites/acanthamoeba/
  question_focus: Acanthamoeba - Granulomatous Amebic Encephalitis (GAE); Keratitis
  question_type: prevention
  question: How to prevent Acanthamoeba - Granulomatous Amebic Encephalitis (GAE);
    Keratitis ?
  answer_snippet: Topics...

- document_id: '0000015'
  document_url: http://www.cdc.gov/parasites/angiostrongylus/
  question_focus: Parasites - Angiostrongyliasis (also known as Angiostrongylus Infection)
  question_type: information
  question: What is (are) Parasites - Angiostrongyliasis (also known as Angiostrongylus
    Infection) ?
  answer_snippet: Angiostrongylus cantonensis is a parasitic worm of rats. It is also
    called the rat lungworm. The adult form of the parasite is found only in rodents.
    ...

- document_id: '0000015'
  document_url: http://www.cdc.gov/parasites/angiostrongylus/
  question_focus: Parasites - Angiostrongyliasis (also known as Angiostrongylus Infection)
  question_type: susceptibility
  question: Who is at risk for Parasites - Angiostrongyliasis (also known as Angiostrongylus
    Infection)? ?
  answer_snippet: "Angiostrongylus cantonensis\n    \nAngiostrongylus cantonensis,\
    \ also known as the rat lungworm, is a parasitic nematode (worm) that is transmitted\
    \ betwe..."

- document_id: '0000015'
  document_url: http://www.cdc.gov/parasites/angiostrongylus/
  question_focus: Parasites - Angiostrongyliasis (also known as Angiostrongylus Infection)
  question_type: exams and tests
  question: How to diagnose Parasites - Angiostrongyliasis (also known as Angiostrongylus
    Infection) ?
  answer_snippet: "Angiostrongylus cantonensis\n    \nDiagnosing A. cantonensis infections\
    \ can be difficult, in part because there are no readily available blood tests.\
    \ Im..."

- document_id: '0000015'
  document_url: http://www.cdc.gov/parasites/angiostrongylus/
  question_focus: Parasites - Angiostrongyliasis (also known as Angiostrongylus Infection)
  question_type: treatment
  question: What are the treatments for Parasites - Angiostrongyliasis (also known
    as Angiostrongylus Infection) ?
  answer_snippet: "Angiostrongylus cantonensis\n    \nThere is no specific treatment\
    \ for A. cantonensis infection. There is some evidence that certain supportive\
    \ treatment..."

- document_id: '0000015'
  document_url: http://www.cdc.gov/parasites/angiostrongylus/
  question_focus: Parasites - Angiostrongyliasis (also known as Angiostrongylus Infection)
  question_type: prevention
  question: How to prevent Parasites - Angiostrongyliasis (also known as Angiostrongylus
    Infection) ?
  answer_snippet: "Angiostrongylus cantonensis\n    \nPrevention of A. cantonensis\
    \ infections involves educating persons residing in or traveling to areas where\
    \ the parasi..."
```

### Domain: nihseniorhealth.gov (48 unique links)
```yaml
- document_id: 0000029
  document_url: http://nihseniorhealth.gov/periodontaldisease/toc.html
  question_focus: Gum (Periodontal) Disease
  question_type: information
  question: What is (are) Gum (Periodontal) Disease ?
  answer_snippet: An Infection of the Gums and Surrounding Tissues Gum (periodontal)
    disease is an infection of the gums and surrounding tissues that hold teeth in
    plac...

- document_id: 0000029
  document_url: http://nihseniorhealth.gov/periodontaldisease/toc.html
  question_focus: Gum (Periodontal) Disease
  question_type: prevention
  question: How to prevent Gum (Periodontal) Disease ?
  answer_snippet: Risk Factors There are a number of risk factors that can increase
    your chances of developing periodontal disease. - Smoking is one of the most signifi...

- document_id: 0000029
  document_url: http://nihseniorhealth.gov/periodontaldisease/toc.html
  question_focus: Gum (Periodontal) Disease
  question_type: symptoms
  question: What are the symptoms of Gum (Periodontal) Disease ?
  answer_snippet: 'Symptoms Symptoms of gum disease may include: - bad breath that
    won''t go away   - red or swollen gums   - tender or bleeding gums  - painful
    chewing  ...'

- document_id: 0000029
  document_url: http://nihseniorhealth.gov/periodontaldisease/toc.html
  question_focus: Gum (Periodontal) Disease
  question_type: treatment
  question: What are the treatments for Gum (Periodontal) Disease ?
  answer_snippet: Controlling the Infection The main goal of treatment is to control
    the infection. The number and types of treatment will vary, depending on how far
    th...

- document_id: 0000029
  document_url: http://nihseniorhealth.gov/periodontaldisease/toc.html
  question_focus: Gum (Periodontal) Disease
  question_type: information
  question: What is (are) Gum (Periodontal) Disease ?
  answer_snippet: Gum disease is an infection of the tissues that hold your teeth
    in place. In its early stages, it is usually painless, and many people are not
    aware t...

- document_id: 0000029
  document_url: http://nihseniorhealth.gov/periodontaldisease/toc.html
  question_focus: Gum (Periodontal) Disease
  question_type: causes
  question: What causes Gum (Periodontal) Disease ?
  answer_snippet: Gum disease is caused by dental plaque -- a sticky film of bacteria
    that builds up on teeth. Regular brushing and flossing help get rid of plaque.
    But...

- document_id: 0000029
  document_url: http://nihseniorhealth.gov/periodontaldisease/toc.html
  question_focus: Gum (Periodontal) Disease
  question_type: information
  question: What is (are) Gum (Periodontal) Disease ?
  answer_snippet: Gingivitis is inflammation of the gums. In gingivitis, the gums
    become red, swollen and can bleed easily. Gingivitis is a mild form of gum disease.
    It...

- document_id: 0000029
  document_url: http://nihseniorhealth.gov/periodontaldisease/toc.html
  question_focus: Gum (Periodontal) Disease
  question_type: information
  question: What is (are) Gum (Periodontal) Disease ?
  answer_snippet: When gingivitis is not treated, it can advance to periodontitis
    (which means "inflammation around the tooth.") In periodontitis, gums pull away
    from t...

- document_id: 0000029
  document_url: http://nihseniorhealth.gov/periodontaldisease/toc.html
  question_focus: Gum (Periodontal) Disease
  question_type: treatment
  question: What are the treatments for Gum (Periodontal) Disease ?
  answer_snippet: If left untreated, gum disease can lead to tooth loss. Gum disease
    is the leading cause of tooth loss in older adults....

- document_id: 0000029
  document_url: http://nihseniorhealth.gov/periodontaldisease/toc.html
  question_focus: Gum (Periodontal) Disease
  question_type: causes
  question: What causes Gum (Periodontal) Disease ?
  answer_snippet: In some studies, researchers have observed that people with periodontal
    disease (when compared to people without periodontal disease) were more likely...
```

