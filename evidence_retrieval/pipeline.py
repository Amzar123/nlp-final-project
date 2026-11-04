import argparse

from .data import PROCESSED_DIR
from .reranker import CrossEncoderReranker
from .retriever import EvidenceRetriever, load_corpus


class EvidencePipeline:
    def __init__(self, retriever, reranker=None, doc_k=5, n_candidates=30):
        self.retriever = retriever
        self.reranker = reranker
        self.doc_k = doc_k
        self.n_candidates = n_candidates

    @classmethod
    def from_corpus_file(cls, corpus_path=PROCESSED_DIR / "wiki_subset.jsonl", use_reranker=True, **kwargs):
        retriever = EvidenceRetriever(load_corpus(corpus_path))
        reranker = CrossEncoderReranker() if use_reranker else None
        return cls(retriever, reranker, **kwargs)

    def run(self, claim, top_k=5, as_dict=True):
        candidates = self.retriever.retrieve(claim, top_k=self.n_candidates, doc_k=self.doc_k)
        if self.reranker is not None:
            results = self.reranker.rerank(claim, candidates, top_k=top_k)
        else:
            results = candidates[:top_k]
        return [r.to_dict() for r in results] if as_dict else results

    def run_batch(self, claims, top_k=5):
        return [{"claim": c, "evidence": self.run(c, top_k=top_k)} for c in claims]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("claims", nargs="+")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    pipe = EvidencePipeline.from_corpus_file(use_reranker=not args.no_rerank)
    for item in pipe.run_batch(args.claims, top_k=args.top_k):
        print("\nClaim:", item["claim"])
        for i, ev in enumerate(item["evidence"], 1):
            score = ev["rerank_score"] if ev["rerank_score"] is not None else ev["retrieval_score"]
            print(f"  {i}. [{ev['title']} #{ev['line_id']}] ({score:.3f}) {ev['text']}")


if __name__ == "__main__":
    main()
