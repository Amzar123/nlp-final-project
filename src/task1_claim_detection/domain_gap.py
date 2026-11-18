"""Domain gap check (plan step 1.5).

    python -m src.task1_claim_detection.domain_gap

Runs the trained claim detector on every claim of FEVER paper_dev.jsonl and reports the pass
rate: the share of real FEVER claims predicted as claim (1). Every FEVER claim is a factual
statement, so 1 - pass_rate is the share of claims Task 1 would drop before Task 2 ever sees
them, an upper bound on end-to-end recall.

Each claim is scored as one sentence (as the pipeline would after splitting). The script also
reports how often spaCy would split a FEVER claim into more than one sentence.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .infer import DEFAULT_MODEL_DIR, PROJECT_DIR, ClaimDetector

RAW_DIR = PROJECT_DIR / "data" / "raw"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--split", default="paper_dev")
    parser.add_argument("--out", default=None, help="default: <model run dir>/domain_gap_<split>.json")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(RAW_DIR / f"{args.split}.jsonl", encoding="utf-8")]
    claims = [r["claim"] for r in rows]
    labels = np.array([r["label"] for r in rows])

    detector = ClaimDetector(args.model_dir)
    probs = detector.score(claims)

    n_sentences = [len(detector.split_sentences(c)) for c in claims]
    split_counts = Counter(min(n, 3) for n in n_sentences)

    def pass_rates(threshold):
        passed = probs >= threshold
        out = {"all": float(passed.mean())}
        for label in sorted(set(labels)):
            out[label] = float(passed[labels == label].mean())
        return out

    order = np.argsort(probs)
    report = {
        "split": args.split,
        "n_claims": len(claims),
        "threshold": detector.threshold,
        "pass_rate@tuned": pass_rates(detector.threshold),
        "pass_rate@0.5": pass_rates(0.5),
        "p_claim_mean": float(probs.mean()),
        "p_claim_median": float(np.median(probs)),
        "spacy_sentences_per_claim": {("3+" if k == 3 else str(k)): v for k, v in sorted(split_counts.items())},
        "lowest_scored_claims": [{"claim": claims[i], "p_claim": float(probs[i]), "label": labels[i]}
                                 for i in order[:25]],
    }

    out_path = Path(args.out) if args.out else Path(args.model_dir).parent / f"domain_gap_{args.split}.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(f"{args.split}: {len(claims):,} claims, threshold {detector.threshold:.2f}")
    for key in ("pass_rate@tuned", "pass_rate@0.5"):
        print(f"  {key:<16} " + "  ".join(f"{k}={v:.4f}" for k, v in report[key].items()))
    print(f"  spaCy sentences per claim: {report['spacy_sentences_per_claim']}")
    print("  lowest scored claims:")
    for item in report["lowest_scored_claims"][:8]:
        print(f"    {item['p_claim']:.3f}  {item['claim']}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
