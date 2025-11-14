import os
import json
import re
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
import strawberry
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import base64
import pandas as pd 
import matplotlib.pyplot as plt
import io
from scipy.spatial import ConvexHull


load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

# ========= Load texts.csv ==========
df = pd.read_csv("texts.csv")
TEXT_IDS = df["id"].tolist()
TEXTS = df["text"].tolist()


# =========================================================
# REGEX FALLBACK
# =========================================================
def regex_extract(text: str) -> Dict[str, Any]:
    print("Using regex extractor...")

    variant = re.search(r"(rs\d+)", text)

    genes = re.findall(r"\b[A-Z0-9][A-Z0-9\-\.]{1,20}\b", text)
    genes = [g for g in genes if not g.lower().startswith("rs")]

    if not genes:
        genes = []

    disease_match = re.search(
        r"(Alzheimer['’]s disease(?: phenotype)?|Alzheimer|Alzheimers?|Alzhimers?|"
        r"Alzhimer['’]s disease|AD|Alzheimer phenotype|"
        r"Alzheimer['’]s phenotype|Cognitive phenotype)",
        text,
        re.IGNORECASE
    )

    if disease_match:
        d = disease_match.group(0).lower()
        disease_clean = "Cognitive phenotype" if "cognitive" in d else "Alzheimer's disease"
    else:
        disease_clean = None

    relation = re.search(
        r"(eQTL|increases risk|increases|increased|associated with|linked to|Endophenotype|risk)",
        text,
        re.IGNORECASE
    )

    return {
        "variant": variant.group(1) if variant else None,
        "gene": genes,
        "disease": disease_clean,
        "relation": relation.group(1) if relation else None,
        "error": None
    }


# =========================================================
# GPT-4o-mini LLM EXTRACTION
# =========================================================
def llm_extract(text: str) -> Dict[str, Any]:

    if client is None:
        print("No OpenAI key → returning LLM error")
        return {
            "variant": None,
            "gene": None,
            "disease": None,
            "relation": None,
            "error": "LLM unavailable"
        }

    print("Using GPT-4o-mini extractor...")

    prompt = f"""
        You are a biomedical text extraction assistant.

        Extract:
        - variant: rsID
        - gene: list of gene symbols
        - disease: normalized disease/phenotype
        - relation: key phrase describing effect

        Normalize:
        - Alzheimer's misspellings
        - AD → Alzheimer's disease
        - Cognitive phenotype as-is

        Ignore demographic info.

        Return STRICT JSON only.

        ### Example:
        Input: "rs11185978 in MPHOSPH1, HTR7 is an Endophenotype for Cognitive phenotype"
        Output:
        {{"variant":"rs11185978","gene":["MPHOSPH1","HTR7"],"disease":"Cognitive phenotype","relation":"Endophenotype"}}

        Input: "{text}"
        Output:
        """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = response.choices[0].message.content
        return json.loads(content)

    except Exception as e:
        print("LLM error:", e)
        return {
            "variant": None,
            "gene": None,
            "disease": None,
            "relation": None,
            "error": f"LLM failure: {str(e)}"
        }


# =========================================================
# EXTRACTION MODES
# =========================================================
def extract_with_mode(text: str, mode: str):

    mode = mode.lower()

    if mode == "llm":
        return {"mode": "llm", "llm": llm_extract(text)}

    if mode == "regex":
        return {"mode": "regex", "regex": regex_extract(text)}

    if mode == "both":
        return {
            "mode": "both",
            "llm": llm_extract(text),
            "regex": regex_extract(text)
        }

    if mode == "auto":
        llm_result = llm_extract(text)
        if llm_result.get("error"):
            return {"mode": "regex", "regex": regex_extract(text)}
        else:
            return {"mode": "llm", "llm": llm_result}

    # default
    return extract_with_mode(text, "auto")


def plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


# =========================================================
# GRAPHQL TYPES
# =========================================================
@strawberry.type
class ExtractionResult:
    variant: Optional[str]
    gene: Optional[List[str]]
    disease: Optional[str]
    relation: Optional[str]
    raw_text: str
    error: Optional[str]


@strawberry.type
class BothExtractionResult:
    llm: ExtractionResult
    regex: ExtractionResult
    mode: str


ExtractionUnion = strawberry.union(
    "ExtractionUnion",
    types=[ExtractionResult, BothExtractionResult]
)

@strawberry.type
class TopicItem:
    topic_id: int
    keywords: list[str]
    example_ids: list[str]
    example_texts: list[str]


@strawberry.type
class TopicResult:
    topics: list[TopicItem]
    topic_plot: str  # base64 PNG
    scatter_plot: str  # base64 PNG


# =========================================================
# GRAPHQL QUERY RESOLVER
# =========================================================
@strawberry.type
class Query:
    @strawberry.field
    def extract(self, text: str, mode: str = "auto") -> ExtractionUnion:

        result = extract_with_mode(text, mode)

        # BOTH mode
        if result["mode"] == "both":
            return BothExtractionResult(
                mode="both",
                llm=ExtractionResult(
                    variant=result["llm"]["variant"],
                    gene=result["llm"]["gene"],
                    disease=result["llm"]["disease"],
                    relation=result["llm"]["relation"],
                    raw_text=text,
                    error=result["llm"].get("error")
                ),
                regex=ExtractionResult(
                    variant=result["regex"]["variant"],
                    gene=result["regex"]["gene"],
                    disease=result["regex"]["disease"],
                    relation=result["regex"]["relation"],
                    raw_text=text,
                    error=result["regex"].get("error")
                )
            )

        # SINGLE mode (llm OR regex)
        single = result.get("llm") or result.get("regex")

        return ExtractionResult(
            variant=single["variant"],
            gene=single["gene"],
            disease=single["disease"],
            relation=single["relation"],
            raw_text=text,
            error=single.get("error")
        )
    
     # ----- Topic Clustering -----
    @strawberry.field
    def cluster_topics(self, n_topics: int = 4, top_k: int = 2) -> TopicResult:
        vect = TfidfVectorizer(stop_words="english", max_features=1500)
        X = vect.fit_transform(TEXTS)

        model = KMeans(n_clusters=n_topics, random_state=42)
        labels = model.fit_predict(X)
        center_words = model.cluster_centers_

        feature_names = vect.get_feature_names_out()

        topics = []
        for i in range(n_topics):
            idx = center_words[i].argsort()[::-1][:8]
            keywords = [feature_names[j] for j in idx]

            docs = np.where(labels == i)[0][:top_k]

            topics.append(
                TopicItem(
                    topic_id=i,
                    keywords=keywords,
                    example_ids=[TEXT_IDS[d] for d in docs],
                    example_texts=[TEXTS[d] for d in docs],
                )
            )

       # ----- PLOTS -----

        # ------------------------------
        # 1. BAR CHART
        # ------------------------------
        plt.figure(figsize=(8, 5))
        _, counts = np.unique(labels, return_counts=True)

        plt.bar(
            range(n_topics),
            counts,
            color=plt.cm.tab10(np.linspace(0, 1, n_topics)),
            edgecolor="black"
        )

        plt.title("Document Count per Topic Cluster", fontsize=16, weight="bold")
        plt.xlabel("Topic ID", fontsize=14)
        plt.ylabel("Number of Documents", fontsize=14)
        plt.xticks(range(n_topics), fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.4)

        bar_png = plot_to_base64()


        # ------------------------------
        # 2. PCA SCATTER WITH CONVEX HULLS
        # ------------------------------
        coords = PCA(n_components=2).fit_transform(X.toarray())

        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, n_topics))

        for k in range(n_topics):
            cluster_points = coords[labels == k]

            # scatter
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                s=50,
                color=colors[k],
                edgecolor="black",
                alpha=0.7,
                label=f"Topic {k}"
            )

            # convex hull
            if len(cluster_points) >= 3:
                hull = ConvexHull(cluster_points)
                hull_pts = cluster_points[hull.vertices]
                plt.fill(
                    hull_pts[:, 0],
                    hull_pts[:, 1],
                    color=colors[k],
                    alpha=0.2,
                    edgecolor="black",
                    linewidth=1.2
                )

        plt.title("PCA Topic Cluster Visualization", fontsize=18, weight="bold")
        plt.xlabel("PCA Component 1", fontsize=14)
        plt.ylabel("PCA Component 2", fontsize=14)

        plt.grid(alpha=0.3, linestyle="--")
        plt.legend(fontsize=12, loc="best")

        scatter_png = plot_to_base64()



        return TopicResult(
            topics=topics,
            topic_plot=bar_png,
            scatter_plot=scatter_png,
        )


# =========================================================
# SERVER
# =========================================================
schema = strawberry.Schema(query=Query)
app = FastAPI()

graphql_app = GraphQLRouter(schema, graphiql=True)
app.include_router(graphql_app, prefix="/graphql")

@app.get("/")
def root():
    return {"message": "GraphQL running at /graphql"}
