import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from .paths import OUTPUTS_DIR, PROCESSED_DIR
from .task2_evidence_retrieval.data import FeverClaims
from .task2_evidence_retrieval.reranker import CrossEncoderReranker
from .task2_evidence_retrieval.retriever import EvidenceRetriever
from .utils import JsonlFile


class RetrievalMetrics:
    @staticmethod
    def full_set_found(evidence_sets, found_keys):
        """True if every sentence of at least one gold evidence set was found."""
        found_keys = set(found_keys)
        return any(all(tuple(s) in found_keys for s in ev) for ev in evidence_sets)

    @staticmethod
    def doc_set_found(evidence_sets, found_pages):
        found_pages = set(found_pages)
        return any(all(page in found_pages for page, _ in ev) for ev in evidence_sets)

    @staticmethod
    def reciprocal_rank(ranked_keys, gold_keys):
        for rank, key in enumerate(ranked_keys, 1):
            if key in gold_keys:
                return 1.0 / rank
        return 0.0


class RetrievalEvaluator:
    """Document recall, sentence recall, hit@1 and MRR, before and after reranking."""

    def __init__(self, retriever, reranker=None, doc_k=5, n_candidates=30, top_k=5, doc_ks=(1, 5, 10)):
        self.retriever = retriever
        self.reranker = reranker
        self.doc_k = doc_k
        self.n_candidates = n_candidates
        self.top_k = top_k
        self.doc_ks = doc_ks

    def evaluate(self, claims):
        m = RetrievalMetrics
        totals = {}
        predictions = []

        def add(name, value):
            totals[name] = totals.get(name, 0.0) + float(value)

        for c in tqdm(claims, desc="Evaluasi"):
            gold_sets = c["evidence_sets"]
            gold_keys = {tuple(s) for ev in gold_sets for s in ev}

            # --- document retrieval ---
            docs = self.retriever.retrieve_documents(c["claim"], k=max(self.doc_ks))
            doc_ids = [self.retriever.pages[i]["id"] for i, _ in docs]
            for k in self.doc_ks:
                add(f"doc_recall@{k}", m.doc_set_found(gold_sets, doc_ids[:k]))

            # --- sentence retrieval (bi-encoder) ---
            candidates = self.retriever.retrieve(c["claim"], top_k=self.n_candidates, doc_k=self.doc_k)
            before = [x.key for x in candidates]
            add("candidate_recall", m.full_set_found(gold_sets, before))
            add("before/evidence_recall@5", m.full_set_found(gold_sets, before[:self.top_k]))
            add("before/hit@1", bool(before) and before[0] in gold_keys)
            add("before/mrr", m.reciprocal_rank(before, gold_keys))

            # --- reranking (cross-encoder) ---
            final = candidates[:self.top_k]
            if self.reranker is not None:
                reranked = self.reranker.rerank(c["claim"], candidates, top_k=len(candidates))
                after = [x.key for x in reranked]
                add("after/evidence_recall@5", m.full_set_found(gold_sets, after[:self.top_k]))
                add("after/hit@1", bool(after) and after[0] in gold_keys)
                add("after/mrr", m.reciprocal_rank(after, gold_keys))
                final = reranked[:self.top_k]

            predictions.append({
                "id": c["id"],
                "claim": c["claim"],
                "label": c["label"],
                "gold_evidence": gold_sets,
                "predicted_evidence": [x.to_dict() for x in final],
            })

        metrics = {k: v / len(claims) for k, v in totals.items()}
        return metrics, predictions

    @staticmethod
    def print_metrics(metrics):
        print("\n=== Document retrieval (BM25) ===")
        for k, v in metrics.items():
            if k.startswith("doc_"):
                print(f"  {k:<22} {v:.4f}")
        print(f"  {'candidate_recall':<22} {metrics['candidate_recall']:.4f}  (upper bound utk reranker)")

        print("\n=== Sentence retrieval ===")
        print(f"  {'metrik':<20} {'sebelum rerank':>15} {'sesudah rerank':>15}")
        for name in ["evidence_recall@5", "hit@1", "mrr"]:
            before = metrics.get(f"before/{name}")
            after = metrics.get(f"after/{name}")
            after_str = f"{after:.4f}" if after is not None else "-"
            print(f"  {name:<20} {before:>15.4f} {after_str:>15}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", default=str(PROCESSED_DIR / "claims_sample.jsonl"))
    parser.add_argument("--corpus", default=str(PROCESSED_DIR / "wiki_subset.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="pakai sebagian claim saja (untuk tes cepat)")
    parser.add_argument("--doc-k", type=int, default=5)
    parser.add_argument("--n-candidates", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--out-dir", default=str(OUTPUTS_DIR / "task2"))
    args = parser.parse_args()

    claims = FeverClaims.read_cache(args.claims)
    if args.limit:
        claims = claims[:args.limit]
    print(f"Jumlah claim yang dievaluasi: {len(claims)}")

    retriever = EvidenceRetriever.from_corpus_file(args.corpus)
    reranker = None if args.no_rerank else CrossEncoderReranker()

    start = time.time()
    evaluator = RetrievalEvaluator(retriever, reranker, doc_k=args.doc_k, n_candidates=args.n_candidates, top_k=args.top_k)
    metrics, predictions = evaluator.evaluate(claims)
    elapsed = time.time() - start
    evaluator.print_metrics(metrics)
    print(f"\nWaktu: {elapsed:.1f} detik ({elapsed / len(claims):.2f} detik/claim)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"n_claims": len(claims), **vars(args), **metrics}, f, indent=2)
    JsonlFile.write(predictions, out_dir / "predictions.jsonl")
    print(f"Hasil disimpan di {out_dir}")


if __name__ == "__main__":
    main()
