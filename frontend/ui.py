import streamlit as st
import aiohttp
import asyncio
import base64
import pandas as pd
import re
import os

GRAPHQL_URL = os.getenv("BACKEND_URL", "http://localhost:8000/graphql")

# ---------------------------
# Helper: Async GraphQL call
# ---------------------------
async def graphql_request(query: str, variables: dict = None):
    async with aiohttp.ClientSession() as session:
        payload = {
            "query": query,
            "variables": variables or {}
        }
        async with session.post(GRAPHQL_URL, json=payload) as resp:
            return await resp.json()

# ---------------------------
# Text highlighter
# ---------------------------
def highlight_entities(text: str):
    if not text:
        return text

    text = re.sub(r"(rs\d+)", r"<b style='color:#b30000'>\1</b>", text)
    text = re.sub(r"\b[A-Z0-9]{3,15}\b", r"<b style='color:#0047b3'>\g<0></b>", text)
    text = re.sub(r"Alzheimer['’]s disease|Cognitive phenotype|Alzheimer|AD",
                  lambda m: f"<b style='color:#00802b'>{m.group(0)}</b>",
                  text,
                  flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Genomic Curation Dashboard",
                   layout="wide",
                   page_icon="🧬")

st.title("🧬 Genomic Curation Dashboard")
tabs = st.tabs([
    "🔍 Entity Extraction",
    "📊 Topic Clustering"
])


# ===========================================================
# TAB 1 — ENTITY EXTRACTION (NEW UI)
# ===========================================================
with tabs[0]:
    st.header("Extract Variant / Gene / Disease / Relation")

    col1, col2 = st.columns([2, 1])

    with col1:
        text_input = st.text_area("Input biomedical text", height=150)

    with col2:
        mode = st.radio("Extraction Mode",
                        ["auto", "llm", "regex", "both"],
                        index=0)

        run_button = st.button("🚀 Run Extraction", use_container_width=True)

    if run_button:
        query = """
        query Extract($text: String!, $mode: String!) {
            extract(text: $text, mode: $mode) {
                __typename
                ... on ExtractionResult {
                    variant
                    gene
                    disease
                    relation
                    rawText
                    error
                }
                ... on BothExtractionResult {
                    llm {
                        variant
                        gene
                        disease
                        relation
                        rawText
                        error
                    }
                    regex {
                        variant
                        gene
                        disease
                        relation
                        rawText
                        error
                    }
                }
            }
        }
        """

        with st.spinner("Extracting..."):
            result = asyncio.run(graphql_request(query, {"text": text_input, "mode": mode}))

        if "errors" in result:
            st.error(result["errors"])
        else:
            data = result["data"]["extract"]
            st.subheader("🔬 Extraction Results")
        if result["data"]["extract"]["__typename"] == "ExtractionResult":
            res = data

            df = pd.DataFrame({
                "Field": ["Variant", "Genes", "Disease", "Relation", "Error"],
                "Value": [
                    res["variant"],
                    ", ".join(res["gene"]) if res["gene"] else None,
                    res["disease"],
                    res["relation"],
                    res["error"]
                ]
            })
            
            st.table(df)

            st.markdown("### Highlighted Text")
            st.markdown(highlight_entities(res["rawText"]), unsafe_allow_html=True)

        else:
            # LLM & Regex Combined — side by side comparison table
            colA, colB = st.columns(2)

            with colA:
                st.markdown("### 🤖 LLM Extraction")
                llm = data["llm"]
                df = pd.DataFrame({
                    "Field": ["Variant", "Genes", "Disease", "Relation", "Error"],
                    "Value": [
                        llm["variant"],
                        ", ".join(llm["gene"]) if llm["gene"] else None,
                        llm["disease"],
                        llm["relation"],
                        llm["error"]
                    ]
                })
                st.table(df)
                st.markdown(highlight_entities(llm["rawText"]), unsafe_allow_html=True)

            with colB:
                st.markdown("### 🔧 Regex Extraction")
                rr = data["regex"]
                df = pd.DataFrame({
                    "Field": ["Variant", "Genes", "Disease", "Relation", "Error"],
                    "Value": [
                        rr["variant"],
                        ", ".join(rr["gene"]) if rr["gene"] else None,
                        rr["disease"],
                        rr["relation"],
                        rr["error"]
                    ]
                })
                st.table(df)
                st.markdown(highlight_entities(rr["rawText"]), unsafe_allow_html=True)

# ===========================================================
# TAB 2 — TOPIC CLUSTERING (FINAL FIXED)
# ===========================================================
with tabs[1]:
    st.header("Topic Clustering & Visualization")

    colA, colB = st.columns(2)
    with colA:
        n_topics = st.slider("Number of clusters", min_value=1, max_value=7, value=4)

    with colB:
        top_k = st.slider("Top-K examples per topic", min_value=1, max_value=5, value=2)


    run_cluster_btn = st.button("🔍 Run Topic Clustering", use_container_width=True)

    if run_cluster_btn:
        query = """
        query Cluster($n: Int!, $topK: Int!) {
            clusterTopics(nTopics: $n, topK: $topK) {
                topics {
                    topicId
                    keywords
                    exampleIds
                    exampleTexts
                }
                topicPlot
                scatterPlot
            }
        }
        """
        with st.spinner("Clustering documents..."):
            result = asyncio.run(graphql_request(query, {"n": n_topics,  "topK": top_k}))

        if "errors" in result:
            st.error(result["errors"])
        else:
            data = result["data"]["clusterTopics"]

            # -------- Display Topics -------
            st.subheader("🧩 Topic Groups")

            colors = ["#f0f0f0", "#e8f4ff", "#fff5e6", "#e6ffe6"]

            for topic in data["topics"]:
                t = topic["topicId"] % len(colors)
                bg = colors[t]

                with st.container():
                    st.markdown(
                        f"""
                        <div style='padding:12px;border-radius:8px;background:{bg}'>
                            <h4>🧠 Topic {topic["topicId"]}</h4>
                            <b>Keywords:</b> {", ".join(topic["keywords"])}
                            <br><br>
                        """, unsafe_allow_html=True)

                    # LIMIT HERE (backend must provide all!)
                    examples = topic["exampleTexts"][:top_k]

                    for text in examples:
                        st.markdown(highlight_entities(text), unsafe_allow_html=True)
                        st.markdown("---")

                    st.markdown("</div>", unsafe_allow_html=True)

            # --------- Plots --------
            st.subheader("📊 Topic Keyword Barplot")
            if data["topicPlot"]:
                st.image(base64.b64decode(data["topicPlot"]))

            st.subheader("📉 2-D UMAP Plot")
            if data["scatterPlot"]:
                st.image(base64.b64decode(data["scatterPlot"]))
