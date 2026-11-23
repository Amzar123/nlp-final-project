import argparse
import json
from pathlib import Path

from .paths import OUTPUTS_DIR, PROCESSED_DIR, PROJECT_DIR
from .task1_claim_detection.infer import DEFAULT_MODEL_DIR, ClaimDetector


class TestCaseRunner:
    def __init__(self, detector, pipe=None):
        self.detector = detector
        self.pipe = pipe

    def run_claim_detection(self, cases):
        scores = self.detector.score([c["text"] for c in cases])
        results = []
        for case, p in zip(cases, scores):
            got = "claim" if p >= self.detector.threshold else "not_claim"
            results.append({**case, "got": got, "p_claim": float(p), "passed": got == case["expected"]})
        return results

    def run_verification(self, cases):
        p_claims = self.detector.score([c["claim"] for c in cases])
        results = []
        for case, p in zip(cases, p_claims):
            r = self.pipe.verify_claim(case["claim"], float(p))
            results.append({**case, "got": r["verdict"], "confidence": r["verdict_confidence"], "p_claim": float(p),
                            "passed": r["verdict"] == case["expected"], "evidence": r["evidence"]})
        return results

    @staticmethod
    def print_claim_detection(results, threshold):
        print(f"\n=== Task 1: claim detection (threshold {threshold:.2f}) ===")
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"[{status}] {r['p_claim']:.3f}  {r['got']:<9}  ({r.get('category', '-')})  {r['text']}")

    @staticmethod
    def print_verification(results):
        print("\n=== Task 1 + 2 + 3: verification ===")
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"\n[{status}] {r['claim']}")
            print(f"       expected {r['expected']}, got {r['got']} ({r['confidence']:.3f}) | p_claim {r['p_claim']:.3f}")
            if r["evidence"]:
                top = max(r["evidence"], key=lambda ev: max(ev["nli"]["entailment"], ev["nli"]["contradiction"]))
                nli = top["nli"]
                print(f"       strongest evidence: [{top['title']} #{top['line_id']}] "
                      f"E={nli['entailment']:.2f} C={nli['contradiction']:.2f}  {top['text'][:100]}")
            if r.get("note"):
                print(f"       note: {r['note']}")

    @staticmethod
    def summary(name, results):
        passed = sum(r["passed"] for r in results)
        return f"{name}: {passed}/{len(results)} passed"

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", nargs="?", default=str(PROJECT_DIR / "tests" / "test_cases.json"))
    parser.add_argument("--only", choices=["claim_detection", "verification"], default=None)
    parser.add_argument("--corpus", default=str(PROCESSED_DIR / "wiki_subset.jsonl"))
    parser.add_argument("--claim-model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default=str(OUTPUTS_DIR / "test_cases" / "results.json"))
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text())
    sections = [args.only] if args.only else [s for s in ("claim_detection", "verification") if cases.get(s)]
    runner = TestCaseRunner(ClaimDetector(args.claim_model_dir))
    output = {}

    if "claim_detection" in sections:
        output["claim_detection"] = runner.run_claim_detection(cases["claim_detection"])
        runner.print_claim_detection(output["claim_detection"], runner.detector.threshold)

    if "verification" in sections:
        # imported here so --only claim_detection works without the corpus or the NLI model
        from .pipeline import ClaimCheckPipeline
        runner.pipe = ClaimCheckPipeline.load(corpus_path=args.corpus, use_claim_detection=False, top_k=args.top_k)
        output["verification"] = runner.run_verification(cases["verification"])
        runner.print_verification(output["verification"])

    print("\n=== Summary ===")
    for name, results in output.items():
        print(runner.summary(name, results))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=float))
    print(f"Saved to {out}")

if __name__ == "__main__":
    main()
