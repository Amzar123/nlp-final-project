import json
import sys
from pathlib import Path
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.paths import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR  # noqa: E402
from src.utils import JsonlFile  # noqa: E402

CORPUS_PATH = PROCESSED_DIR / "wiki_subset.jsonl"
CLAIMS_PATH = PROCESSED_DIR / "claims_sample.jsonl"
TASK1_DIR = MODELS_DIR / "task1_roberta-base_label"
TASK1_MODEL_DIR = TASK1_DIR / "model"
TEST_RESULTS_PATH = OUTPUTS_DIR / "test_cases" / "results.json"
TASK2_METRICS_PATH = OUTPUTS_DIR / "task2" / "metrics.json"
TASK3_METRICS = {
    "retrieved": OUTPUTS_DIR / "task3" / "metrics_retrieved.json",
    "oracle": OUTPUTS_DIR / "task3" / "metrics_oracle.json",
}

def rel(path):
    return str(Path(path).relative_to(PROJECT_DIR))

# data

@st.cache_data(show_spinner=False)
def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None

@st.cache_resource(show_spinner="Opening the Wikipedia library…")
def load_corpus():
    return JsonlFile.read(CORPUS_PATH)

@st.cache_resource(show_spinner=False)
def corpus_index():
    return {p["id"]: p for p in load_corpus()}

def sentence_text(page_id, line_id):
    page = corpus_index().get(page_id)
    if page is None:
        return None
    return next((t for lid, t in page["sentences"] if int(lid) == int(line_id)), None)

@st.cache_data(show_spinner=False)
def load_claims():
    return JsonlFile.read(CLAIMS_PATH) if CLAIMS_PATH.exists() else []

# models
@st.cache_resource(show_spinner="Loading the statement spotter…")
def load_detector():
    from src.task1_claim_detection.infer import ClaimDetector
    return ClaimDetector(TASK1_MODEL_DIR)

@st.cache_resource(show_spinner="Indexing the library for search…")
def load_retriever():
    from src.task2_evidence_retrieval.retriever import EvidenceRetriever
    return EvidenceRetriever(load_corpus())

@st.cache_resource(show_spinner="Loading the re-ranker…")
def load_reranker():
    from src.task2_evidence_retrieval.reranker import CrossEncoderReranker
    return CrossEncoderReranker()

@st.cache_resource(show_spinner="Loading the reading model (downloads once)…")
def load_verifier():
    from src.task3_verification.nli import NLIVerifier
    return NLIVerifier()

def run_evidence(claim, top_k=5, doc_k=5, n_candidates=30, keep_stages=False):
    """Task 2 with every stage exposed (same logic as EvidencePipeline.run)."""
    retriever, reranker = load_retriever(), load_reranker()
    docs = retriever.retrieve_documents(claim, k=doc_k)
    candidates = retriever.retrieve(claim, top_k=n_candidates, doc_k=doc_k)
    bi_order = [c.to_dict() for c in candidates]  # snapshot before the reranker re-sorts
    reranked = reranker.rerank(claim, candidates, top_k=len(candidates))
    final = [c.to_dict() for c in reranked[:top_k]]
    if not keep_stages:
        return final
    return final, {
        "docs": [{"page_id": retriever.pages[i]["id"], "title": retriever.pages[i]["title"], "bm25": float(s),
                  "source": retriever.pages[i].get("source")} for i, s in docs if s > 0],
        "bi_encoder": bi_order,
        "reranked": [c.to_dict() for c in reranked],
    }
