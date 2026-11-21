import json
from dataclasses import asdict, dataclass

from .bm25 import BM25, top_k_indices


@dataclass
class Evidence:
    page_id: str
    line_id: int
    title: str
    text: str
    doc_score: float = 0.0
    retrieval_score: float = 0.0
    rerank_score: float = None

    @property
    def key(self):
        return (self.page_id, self.line_id)

    def to_dict(self):
        return asdict(self)


def get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_corpus(path):
    pages = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))
    return pages


class EvidenceRetriever:
    def __init__(
        self,
        pages,
        bi_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        title_weight=2.0,
        device=None,
    ):
        self.pages = pages
        self.title_weight = title_weight

        print(f"Bikin index BM25 untuk {len(pages)} halaman...")
        self.title_index = BM25().fit([p["title"] for p in pages])
        self.body_index = BM25().fit([" ".join(s[1] for s in p["sentences"]) for p in pages])

        self.device = device or get_device()
        self.bi_encoder_name = bi_encoder_name
        self._encoder = None 

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.bi_encoder_name, device=self.device)
        return self._encoder

    # ---------- tahap 1: document retrieval ----------

    def retrieve_documents(self, claim, k=5):
        scores = self.title_weight * self.title_index.get_scores(claim) + self.body_index.get_scores(claim)
        return top_k_indices(scores, k)

    # ---------- tahap 2: sentence retrieval ----------

    def candidate_sentences(self, claim, doc_k=5):
        """Semua kalimat dari top doc_k halaman, belum diurutkan."""
        candidates = []
        for page_idx, doc_score in self.retrieve_documents(claim, doc_k):
            if doc_score <= 0:
                continue
            page = self.pages[page_idx]
            for line_id, text in page["sentences"]:
                candidates.append(Evidence(
                    page_id=page["id"],
                    line_id=int(line_id),
                    title=page["title"],
                    text=text,
                    doc_score=doc_score,
                ))
        return candidates

    def score_sentences(self, claim, candidates):
        if not candidates:
            return candidates
        texts = [f"{c.title}: {c.text}" for c in candidates]
        claim_emb = self.encoder.encode([claim], normalize_embeddings=True)
        sent_emb = self.encoder.encode(texts, normalize_embeddings=True, batch_size=64)
        sims = (sent_emb @ claim_emb[0]).astype(float)
        for c, s in zip(candidates, sims):
            c.retrieval_score = float(s)
        return candidates

    def retrieve(self, claim, top_k=30, doc_k=5):
        candidates = self.score_sentences(claim, self.candidate_sentences(claim, doc_k))
        candidates.sort(key=lambda c: c.retrieval_score, reverse=True)
        return candidates[:top_k]
