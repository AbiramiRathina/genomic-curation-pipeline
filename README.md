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

  * `id` — a short identifier (e.g., `T001`, `T002`)
  * `text` — the genomic phrase to be extracted and clustered

This handcrafted dataset ensures:

* full control over content,
* diversity in entity structures,
* realistic vocabulary for Alzheimer’s and cognitive phenotype genetics,
* and relevance for evaluating extraction + topic grouping.

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
  "text_id": "T0034",
  "variant": "rs429358",
  "gene": ["APOE"],
  "phenotype": "Alzheimer’s disease",
  "relation": "increases risk of",
  "evidence_span": "rs429358 in APOE increases AD risk"
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
* PCA scatter plot (colored clusters)

---

### 🖥 C. Streamlit Curator UI

**Entity View**

* Side-by-side LLM vs Regex
* Highlighted entities (colors)
* Clean comparison table

**Clustering View**

* Select `# clusters`
* Select `Top-K examples`
* Alternating color blocks per topic
* Plots displayed directly in UI

---

## 🧱 Installation

### 1. Clone repository

```bash
git clone <your-repo>
cd your-repo
```

### 2. Install dependencies

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

Here is a **clean, copy-paste-ready Error Analysis section in Markdown**, matching the style of your README.
Just paste it directly into your README.md — no extra formatting needed.

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
👉 The regex method **breaks on unseen misspellings**, while the LLM extractor generalizes better.

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

---

## 🔑 Case 5 — LLM Unavailable → Error Handling

If the OpenAI API key is missing:

* **LLM extractor returns an error**
* **Auto mode automatically falls back to Regex**

This was manually tested by unsetting the environment variable (`OPENAI_API_KEY`).

**Conclusion:**
👉 The system is robust: *auto mode* guarantees extraction even without OpenAI access.

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

If you'd like, I can also generate a **“Limitations & Next Steps”** section in the same style.


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
| evidence_span | Concise summary      |
| cluster_id    | Topic assigned       |
| keywords      | Topic keywords       |

---

## 📎 Example Output Screenshot

(*Add your Streamlit screenshots here*)

---

## 🏁 Summary

This project delivers a complete genomic text–curation stack:

✔ Regex + LLM hybrid extraction
✔ Topic modeling with visualization
✔ Streamlit curator UI
✔ GraphQL backend API
✔ Docker-deployable

Everything is lightweight and designed to meet the **zero-cost requirement**.

