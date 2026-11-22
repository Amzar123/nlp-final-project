"""End-to-end fact-checking pipeline: raw text -> claims -> evidence -> verdict.

    Task 1  ClaimDetector    (src/task1_claim_detection)  keep sentences that are factual claims
    Task 2  EvidencePipeline (src/task2_evidence_retrieval) BM25 -> bi-encoder -> cross-encoder, top-k sentences
    Task 3  NLIVerifier      (src/task3_verification)      NLI over (evidence, claim) -> SUPPORTS / REFUTES / NEI

    python -m src.pipeline "Hello everyone! The Eiffel Tower is located in Berlin. I love pizza."
    python -m src.pipeline --no-claim-detection "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."
    python -m src.pipeline --json "..."        # print the full JSON output

In code:
    from src.pipeline import FactCheckPipeline
    pipe = FactCheckPipeline()
    results = pipe.check(raw_text)
"""
import argparse
import json

from .paths import PROCESSED_DIR
from .task2_evidence_retrieval.pipeline import EvidencePipeline

from .task3_verification.nli import DEFAULT_MODEL, NLIVerifier

class FactCheckPipeline:
    def __init__(self, claim_detector=None, evidence_pipeline=None, verifier=None, top_k=5):
        self.claim_detector = claim_detector
        self.evidence_pipeline = evidence_pipeline
        self.verifier = verifier
        self.top_k = top_k

    @classmethod
    def load(cls, corpus_path=PROCESSED_DIR / "wiki_subset.jsonl", claim_model_dir=None,
             nli_model=DEFAULT_MODEL, nli_threshold=0.5, use_claim_detection=True, top_k=5):
        detector = None
        if use_claim_detection:
            from .task1_claim_detection.infer import DEFAULT_MODEL_DIR, ClaimDetector
            detector = ClaimDetector(claim_model_dir or DEFAULT_MODEL_DIR)
        return cls(
            claim_detector=detector,
            evidence_pipeline=EvidencePipeline.from_corpus_file(corpus_path),
            verifier=NLIVerifier(nli_model, threshold=nli_threshold),
            top_k=top_k,
        )

    def verify_claim(self, claim, check_worthy_confidence=None):
        evidence = self.evidence_pipeline.run(claim, top_k=self.top_k)
        result = self.verifier.verify(claim, evidence)
        return {
            "claim": claim,
            "check_worthy_confidence": check_worthy_confidence,
            "verdict": result["label"],
            "verdict_confidence": result["confidence"],
            "scores": result["scores"],
            "evidence": result["evidence"],
        }

    def check(self, text):
        """Run every claim found in `text` through retrieval + NLI.
        Without a claim detector the whole text is treated as one claim.
        """
        if self.claim_detector is None:
            claims = [{"sentence": text.strip(), "p_claim": None}]
        else:
            claims = self.claim_detector.detect(text)
        return [self.verify_claim(c["sentence"], c["p_claim"]) for c in claims]

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text")
    parser.add_argument("--corpus", default=str(PROCESSED_DIR / "wiki_subset.jsonl"))
    parser.add_argument("--claim-model-dir", default=None)
    parser.add_argument("--no-claim-detection", action="store_true", help="treat the input as a single claim")
    parser.add_argument("--nli-model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.5, help="NLI threshold below which the verdict is NEI")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="print the full JSON output")
    args = parser.parse_args()

    pipe = FactCheckPipeline.load(
        corpus_path=args.corpus, claim_model_dir=args.claim_model_dir, nli_model=args.nli_model,
        nli_threshold=args.threshold, use_claim_detection=not args.no_claim_detection, top_k=args.top_k,
    )
    results = pipe.check(args.text)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return
    if not results:
        print("No factual claims found.")
    for r in results:
        cw = r["check_worthy_confidence"]
        print(f"\nClaim: {r['claim']}" + (f"  (p_claim={cw:.3f})" if cw is not None else ""))
        print(f"Verdict: {r['verdict']} ({r['verdict_confidence']:.3f})")
        for i, ev in enumerate(r["evidence"], 1):
            nli = ev["nli"]
            print(f"  {i}. [{ev['title']} #{ev['line_id']}] E={nli['entailment']:.2f} "
                  f"C={nli['contradiction']:.2f}  {ev['text']}")
            
if __name__ == "__main__":
    main()
