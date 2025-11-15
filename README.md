# 🧬 Genomic Text Curation & Topic Grouping

A lightweight NLP pipeline for extracting genetic entities and grouping biomedical texts into interpretable research topics.

---

## 📌 Overview

This project implements an end-to-end genomic text–curation system consisting of:

1. **Entity & Relation Extraction** (Variant, Gene, Disease/Phenotype, Relation)
2. **Document Topic Clustering** (TF-IDF + KMeans + PCA visualization)
3. **GraphQL API** (FastAPI + Strawberry)
4. **Streamlit Curator UI**

   * Highlighted entities
   * Cluster explorer with plots
   * Side-by-side LLM vs Regex extraction

This pipeline helps curators triage literature faster and produce structured evidence from unstructured text.

---

## 📄 Dataset Construction (`texts.csv`)

For this project, I created a custom dataset of **20 manually-curated genomic text snippets** following the instructions in the assignment.

To ensure biological realism and domain alignment:

* I referenced **six publications** listed on the official project guideline source:
  **[https://advp.niagads.org/publications](https://advp.niagads.org/publications)**
* I opened a **separate Jupyter notebook (`notebook.ipynb`)** where I scraped and read through the abstracts, figure captions, and key variant–gene–phenotype descriptions from these papers.
* From these documents, I **manually wrote 20 short phrasing-style snippets**, each mimicking the style of genetics literature (variants, genes, phenotypes, relation verbs).
* These curated snippets were finally saved into the repository as **`texts.csv`**, with two columns:

  * `id` — a short identifier (e.g., `T1`, `T2`)
  * `text` — the genomic phrase to be extracted and clustered

This handcrafted dataset ensures:

* full control over content,
* diversity in entity structures,
* realistic vocabulary for Alzheimer’s and cognitive phenotype genetics,
* and relevance for evaluating extraction + topic grouping.

---

## 🧰 Tech Stack & Rationale

This project uses a lightweight, fully open-source stack designed to meet the assignment requirement of **zero-cost NLP processing** while still providing a clean, modern, and extensible architecture.

### **Backend**

| Component                 | Why It’s Used                                                                                                                                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Python 3.10+**          | Core language for NLP, data processing, and fast prototyping.                                                                                                                                                |
| **FastAPI**               | Extremely fast API framework; easy to build async, high-performance microservices.                                                                                                                           |
| **Strawberry GraphQL**    | Simple, modern GraphQL library with Python type-hint support. Allows elegant schema definitions (`@strawberry.type`) and avoids the boilerplate of REST. Ideal for structured data like entities & clusters. |
| **scikit-learn**          | Provides TF-IDF vectorization, KMeans clustering, and PCA for topic modeling and visualization—lightweight and assignment-compliant.                                                                         |
| **Matplotlib**            | Required to generate topic barplots and PCA scatter plots for the UI.                                                                                                                                        |
| **pandas / numpy**        | Standard libraries for loading and manipulating texts.                                                                                                                                                       |
| **regex (re)**            | Enables rule-based extraction for variants, genes, and relations.                                                                                                                                            |
| **OpenAI API (optional)** | The LLM extractor is included as an **optional** enhancement. The system gracefully falls back to regex when the API key is not set.                                                                         |

---

### **Frontend**

| Component                        | Why It’s Used                                                                                                                         |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Streamlit**                    | Fastest way to build an interactive NLP dashboard. Auto-refresh columns, tables, code execution. Perfect for curator-style workflows. |
| **Aiohttp (async)**              | Allows async GraphQL calls from Streamlit to the backend without blocking UI interactions.                                            |
| **HTML-based Text Highlighting** | Makes extracted entities visually obvious to curators.                                                                                |
| **Base64 Plot Embedding**        | Sends images directly from backend → GraphQL → Streamlit without saving files.                                                        |

---

### ⭐ Why Use **Strawberry GraphQL**?

Strawberry was chosen because:

#### **1. Clean, pythonic schema**

You define your GraphQL schema with Python classes:

```python
@strawberry.type
class ExtractionResult:
    variant: str | None
    gene: list[str] | None
```

Much easier than Graphene or “manual” GraphQL JSON building.

#### **2. Automatic type-checking and validation**

Strawberry uses Python type hints (`List[str]`, `Optional[str]`) to automatically generate the GraphQL schema.

#### **3. Simple FastAPI integration**

Just:

```python
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")
```

No complex setup.

#### **4. Perfect fit for structured NLP outputs**

Entity extraction returns structured objects:

* variant
* gene list
* disease
* relation
* raw text

GraphQL is ideal for nested structured data like this.

#### **5. Zero-cost, zero-GPU, zero-complexity**

Fits the assignment guideline of building a lightweight NLP pipeline with simple, clear code.

---
🏗️ System Architecture 

                         ┌──────────────────────────┐
                         │        User (UI)         │
                         │  Streamlit Web App       │
                         └─────────────▲────────────┘
                                       │ HTTP (GraphQL)
                                       │
                     ┌─────────────────┴──────────────────┐
                     │         FastAPI Backend            │
                     │     with Strawberry GraphQL        │
                     └─────────────────▲──────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
        ┌───────────┴───────┐  ┌──────┴────────┐   ┌─────┴────────┐
        │ Entity Extraction  │  │ Topic Clustering│   │ Visualization │
        │   (Regex + LLM)    │  │   (TF-IDF,      │   │ (Matplotlib   │
        │ Regex patterns for │  │   KMeans, PCA)  │   │  Convex Hulls)│
        │ rsID / Gene / AD   │  │ Document groups │   │ Bar + Scatter │
        └──────────▲─────────┘  └──────▲─────────┘   └─────▲────────┘
                   │                   │                   │
                   │                   │                   │
                ┌──┴───────────────────┴───────────────────┴───┐
                │          texts.csv (Local Dataset)            │
                │    20 curated biomedical text snippets        │
                └───────────────────────────────────────────────┘

---

## 📂 Repository Structure

```
.
├── backend/
│   ├── app.py                    # FastAPI + GraphQL backend
│   ├── texts.csv                 # Input dataset
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── ui.py                     # Streamlit dashboard
│   ├── requirements.txt
│   └── Dockerfile
│
├── docker-compose.yml
└── README.md
```

---

## 🚀 Features

### 🔍 A. Entity & Relation Extraction

Extracts:

| Field    | Example               |
| -------- | --------------------- |
| Variant  | `rs429358`            |
| Gene     | `APOE`                |
| Disease  | `Alzheimer’s disease` |
| Relation | `increases risk of`   |

Extraction methods:

* **Regex-based** (fast, deterministic)
* **LLM-based** (GPT-4o-mini)
* **Auto mode** (fallback: LLM → regex)
* **Both** (side-by-side comparison)

Unified schema:

```json
{
  "variant": "rs429358",
  "gene": ["APOE"],
  "disease": "Alzheimer’s disease",
  "relation": "increases risk of",
  "error": null
}
```

---

### 📊 B. Topic Clustering

Pipeline:

* TF-IDF vectorization (stopwords removed)
* KMeans clustering
* PCA 2-D projection
* Convex-hull boundaries around clusters

Backend returns:

* Top keywords per topic
* Example texts per topic
* Bar plot (docs per cluster)
* PCA scatter plot (colored clusters, with boundries marked)

---

### 🖥 C. Streamlit Curator UI

**Entity View**

* Option to choose mode of entity extraction: auto, llm, regex, both
* Highlighted entities (colors)
* Clean comparison table

**Clustering View**

* Select `No of clusters(1-7)`
* Select `Top-K examples(1-5)`
* Alternating color blocks per topic for easy understanding
* Plots displayed directly in UI

---

## 🧱 Installation

### 1. Clone repository

```bash
git clone <ssh/http genomic-curation-pipeline>
cd genomic-curation-pipeline
```

### 2. Create virtual environment
```
pythoin3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

Backend:

```bash
pip install -r backend/requirements.txt
```

Frontend:

```bash
pip install frontend/requirements.txt
```

---

## ▶️ Running Locally (Without Docker)

### Backend

```bash
uvicorn app:app --reload --port 8000
```

GraphQL Playground:

```
http://localhost:8000/graphql
```

### Frontend

```bash
streamlit run ui.py
```

UI:

```
http://localhost:8501
```

---

## 🐳 Running With Docker Compose

```bash
docker compose up --build
```

Services:

* Backend → `http://localhost:8000/graphql`
* Frontend → `http://localhost:8501`

---

## 🔍 Example GraphQL Query (Extraction)

```graphql
{
  extract(text: "rs13334456 in MPHOSPH1 is an increases alzhimers", mode: "auto") {
    ... on ExtractionResult {
      variant
      gene
      disease
      relation
      error
    }
  }
}


{
  extract(text: "rs13334456 in MPHOSPH1 is an increases alzhimers", mode: "llm") {
    ... on ExtractionResult {
      variant
      gene
      disease
      relation
      error
    }
  }
}


{
  extract(text: "rs13334456 in MPHOSPH1 is an increases alzhimers", mode: "regex") {
    ... on ExtractionResult {
      variant
      gene
      disease
      relation
      error
    }
  }
}

{
  extract(text: "s13334456 in MPHOSPH1 is an increases alzhimers", mode: "both") {
    ... on BothExtractionResult {
      llm { variant gene disease relation error }
      regex { variant gene disease relation error }
    }
  }
}
```

---

## 📊 Example GraphQL Query (Clustering)

```graphql
query Cluster {
  clusterTopics(nTopics: 4, topK: 3) {
    topics {
      topicId
      keywords
      exampleTexts
    }
    topicPlot
    scatterPlot
  }
}
```
---

# ❗ Error Analysis

This section summarizes the main failure cases and observations from the entity extraction and topic-clustering components.
All findings were **verified manually**, and I will attach screenshots of the outputs as proof.

---

## 🔍 1. Regex vs. LLM Extraction: Variant–Gene–Disease Detection

### ✅ Case 1 — Correct extraction by *both* LLM and Regex

**Input:**
`rs2666895 in CHST1, MIR7154 is shown to increase alzhmer-’s disease risk`

* **LLM Extraction:** Correctly identifies:

  * Variant: `rs2666895`
  * Genes: `CHST1`, `MIR7154`
  * Disease: normalized form *“Alzheimer’s disease”*
  * Relation: *increases risk*
    
* **Regex Extraction:** Also succeeds because:

  * The variant matches `rs\d+`
  * Genes are uppercase tokens
  * Disease phrase is close enough to match the Alzheimer's regex mapping

**Conclusion:**
👉 **Both systems perform correctly on mildly misspelled disease names.**
<img width="1920" height="1080" alt="Screenshot 2025-11-14 at 7 12 12 PM (2)" src="https://github.com/user-attachments/assets/42947233-314c-40af-94dc-8d1d521e9986" />

---

## ❌ Case 2 — Regex fails on slightly different misspelling

**Input:**
`rs2666895 in CHST1, MIR7154 is shown to increase lzhimer’s disease risk`

* **LLM Extraction:**
  Correctly normalizes “lzhimer’s” → *“Alzheimer’s disease”*.

* **Regex Extraction:**
  ❌ Fails to detect disease because the misspelling does **not** match any included variants
  (`Alzhimer, Alzheimers, Alzheimer's, AD` etc).

**Conclusion:**
👉 The regex method **breaks on unseen misspellings**, while the LLM extractor generalizes better. But is not able to hightlight the word.
<img width="1920" height="1080" alt="Screenshot 2025-11-14 at 7 12 12 PM (2)" src="https://github.com/user-attachments/assets/81e860de-959c-4353-ad0c-4bdc43592322" />


---

## ❗ Case 3 — Ambiguity in gene symbols (regex confusion)

**Input:**
`rs2666895 in CHST1, RS7154 is shown to increase lzhimer’s disease risk`

* **LLM Extraction:**
  Correctly extracts **CHST1** and **RS7154** as gene symbols.

* **Regex Extraction:**
  ❌ Extracts **RS7154** incorrectly as both:

  * a *variant* (since it starts with “rs” / “RS”)
  * a *gene* (uppercase token rule)

**Note:**
This is due to regex rules, not biology (I do not know real gene nomenclature).
Regex treats:

* `rs\d+` → variant
* `RS###` → also matches uppercase gene token

**Conclusion:**
👉 Regex is brittle and cannot distinguish between gene symbols and variants in edge cases where prefixes overlap.
<img width="1920" height="1080" alt="Screenshot 2025-11-14 at 7 12 12 PM (2)" src="https://github.com/user-attachments/assets/0280a1b7-fd78-4426-a8b2-d63aa3381f6a" />


---

## 🚧 Case 4 — Missing examples in Topic Clustering (Top-K not always met)

For **4 clusters** and **Top-K = 5**, not every cluster shows 5 examples.

Reason:

* We only have **20 texts**.
* KMeans assigns documents unevenly.
* Some clusters receive only 2–3 texts.

**Conclusion:**
👉 This is expected behavior — top-k limits the maximum, not the minimum.
👉 Requires more documents or cluster-size constraints to fix.
<img width="1920" height="1080" alt="Screenshot 2025-11-14 at 7 14 39 PM (2)" src="https://github.com/user-attachments/assets/38adef9e-8792-4da0-870b-0f074dd8063f" />


---

## 🔑 Case 5 — LLM Unavailable → Error Handling

If the OpenAI API key is missing:

* **LLM extractor returns an error**
* **Auto mode automatically falls back to Regex**

This was manually tested by unsetting the environment variable (`OPENAI_API_KEY`).

**Conclusion:**
👉 The system is robust: *auto mode* guarantees extraction even without OpenAI access.
<img width="1920" height="1080" alt="Screenshot 2025-11-14 at 7 14 39 PM (2)" src="https://github.com/user-attachments/assets/f3f37d86-d08f-4f00-a24d-6852215a48d2" />
<img width="1920" height="1080" alt="Screenshot 2025-11-14 at 7 14 39 PM (2)" src="https://github.com/user-attachments/assets/facd5155-fd69-4eba-8c5b-9f96251de635" />



---

# ✔ Summary Table

| Case | Input                | LLM Result    | Regex Result      | Root Cause                         |
| ---- | -------------------- | ------------- | ----------------- | ---------------------------------- |
| 1    | *alzhmer-’s disease* | Correct       | Correct           | Misspelling close enough for regex |
| 2    | *lzhimer’s disease*  | Correct       | ❌ Wrong           | Regex misspelling coverage         |
| 3    | *RS7154*             | Correct       | ❌ Ambiguous       | Regex gene/variant confusion       |
| 4    | K=4, top-5           | OK            | OK but incomplete | Not enough documents               |
| 5    | Missing API key      | Auto fallback | Works             | LLM unavailable                    |

---

## 🚧 Limitations

* Regex is brittle for novel gene naming patterns
* LLM extraction depends on API availability
* PCA projection may distort cluster boundaries
* No deep biomedical model (by design — cost restriction)
* No dependency-parsing–based relation extraction yet

---

## 🔮 Future Enhancements

* Add **SpaCy + SciSpaCy hybrid NER**
* Add **UMAP** for cleaner nonlinear embeddings
* Use **sentence-transformers** embeddings for richer clusters
* Add **interactive cluster explorer** (hover-text)
* Add **downloadable curation tables**
* Add **relation dependency patterns** using SpaCy
* Add **loggers** to help with debugging, and to match production code standards

---

## 📘 Curation Schema (Final)

| Field         | Description          |
| ------------- | -------------------- |
| text_id       | From `texts.csv`     |
| raw_text      | Original snippet     |
| variant       | Extracted rsID       |
| gene          | List of gene names   |
| disease       | Normalized phenotype |
| relation      | Type of effect       |
| cluster_id    | Topic assigned       |
| keywords      | Topic keywords       |

---

## 📎 Example Output Screenshot
<img width="1920" height="1080" alt="Screenshot 2025-11-14 at 7 08 04 PM (2)" src="https://github.com/user-attachments/assets/9750d5ac-85da-450b-9d8e-7c05f9d5a612" />

<img width="1920" height="1080" alt="Screenshot 2025-11-14 at 7 07 53 PM (2)" src="https://github.com/user-attachments/assets/5c741ed6-6c64-4693-8ee5-a576e635c5c6" />

<img width="1920" height="1080" alt="Screenshot 2025-11-14 at 7 07 46 PM (2)" src="https://github.com/user-attachments/assets/769689df-2009-48da-ad28-46dd171bc821" />

<img width="1920" height="1080" alt="Screenshot 2025-11-14 at 7 07 40 PM (2)" src="https://github.com/user-attachments/assets/97259618-633a-4044-a00b-921d72b2f205" />

<img width="1920" height="1080" alt="Screenshot 2025-11-14 at 7 04 26 PM (2)" src="https://github.com/user-attachments/assets/91848a19-b109-406d-9053-6921f758f842" />

<img width="1920" height="1080" alt="Screenshot 2025-11-14 at 7 04 29 PM (2)" src="https://github.com/user-attachments/assets/c227f1e7-49df-4b74-bc98-ac9bf79061f7" />

<img width="1920" height="1080" alt="Screenshot 2025-11-14 at 7 04 36 PM (2)" src="https://github.com/user-attachments/assets/4493fe13-54f3-43b3-b789-da2446f90649" />


---

## 🏁 Summary

This project delivers a complete genomic text–curation stack:

✔ Regex + LLM hybrid extraction
✔ Topic modeling with visualization
✔ Streamlit curator UI
✔ GraphQL backend API
✔ Docker-deployable

Everything is lightweight and designed to meet the **zero-cost requirement**.

