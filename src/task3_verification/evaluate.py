"""Evaluate Task 3 (NLI verification) on top of the Task 2 retrieval output.

    python -m src.task3_verification.evaluate --limit 50          # quick test
    python -m src.task3_verification.evaluate                     # 500 S/R claims + 250 NEI claims
    python -m src.task3_verification.evaluate --oracle-evidence   # gold evidence instead of retrieval

Claims: `claims_sample.jsonl` (the SUPPORTS/REFUTES claims the wiki subset was built for) plus
`--n-nei` NOT ENOUGH INFO claims sampled from `shared_task_dev.jsonl`.

NLI probabilities are computed once per claim, then the NEI threshold is swept for free.
The claims are split in half (seeded): the threshold is tuned on the `tune` half and reported
on the `test` half, so the tuned number is not picked on the claims it is scored on.

Metrics: label accuracy, macro F1, per-class P/R/F1, confusion matrix and FEVER score
(label correct AND, for SUPPORTS/REFUTES, a complete gold evidence set in the top-k evidence).
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from tqdm import tqdm

from ..paths import OUTPUTS_DIR, PROCESSED_DIR, RAW_DIR
from ..task2_evidence_retrieval.data import load_claims, read_claims_cache
from ..task2_evidence_retrieval.evaluate import full_set_found
from ..task2_evidence_retrieval.pipeline import EvidencePipeline

from .nli import DEFAULT_MODEL, LABELS, NLIVerifier, aggregate

THRESHOLDS = [round(t, 2) for t in np.arange(0.30, 0.96, 0.05)]


def load_eval_claims(claims_path, dev_path, n_nei, seed):
    claims = read_claims_cache(claims_path)
    if n_nei:
        nei = [c for c in load_claims(dev_path, include_nei=True) if c["label"] == "NOT ENOUGH INFO"]
        rng = random.Random(seed)
        claims += rng.sample(nei, min(n_nei, len(nei)))
    return claims


def gold_evidence(claim, sentence_lookup):
    """First gold evidence set whose sentences are all in the corpus, as evidence dicts."""
    for ev_set in claim["evidence_sets"]:
        keys = [tuple(k) for k in ev_set]
        if all(k in sentence_lookup for k in keys):
            return [{"page_id": p, "line_id": l, **sentence_lookup[(p, l)]} for p, l in keys]
    return None


def compute_metrics(rows, threshold):
    y_true, y_pred, fever = [], [], []
    for r in rows:
        label, _ = aggregate(r["sentence_probs"], r["concat_probs"], threshold)
        y_true.append(r["label"])
        y_pred.append(label)
        evidence_ok = r["label"] == "NOT ENOUGH INFO" or full_set_found(r["gold_evidence"], r["evidence_keys"])
        fever.append(label == r["label"] and evidence_ok)

    p, rc, f, support = precision_recall_fscore_support(y_true, y_pred, labels=list(LABELS), zero_division=0)
    return {
        "threshold": threshold,
        "n": len(rows),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0),
        "fever_score": float(np.mean(fever)),
        "per_class": {
            lab: {"precision": float(p[i]), "recall": float(rc[i]), "f1": float(f[i]), "support": int(support[i])}
            for i, lab in enumerate(LABELS)
        },
        "confusion_matrix": {"labels": list(LABELS),
                             "matrix": confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist()},
    }


def print_metrics(name, m):
    print(f"\n=== {name} (n={m['n']}, threshold={m['threshold']:.2f}) ===")
    print(f"  accuracy    {m['accuracy']:.4f}")
    print(f"  macro F1    {m['macro_f1']:.4f}")
    print(f"  FEVER score {m['fever_score']:.4f}")
    print(f"  {'class':<16} {'P':>6} {'R':>6} {'F1':>6} {'n':>5}")
    for lab, s in m["per_class"].items():
        print(f"  {lab:<16} {s['precision']:>6.3f} {s['recall']:>6.3f} {s['f1']:>6.3f} {s['support']:>5}")
    print("  confusion (rows = gold, cols = predicted S / R / NEI):")
    for lab, row in zip(LABELS, m["confusion_matrix"]["matrix"]):
        print(f"    {lab:<16} {row}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--claims", default=str(PROCESSED_DIR / "claims_sample.jsonl"))
    parser.add_argument("--dev", default=str(RAW_DIR / "shared_task_dev.jsonl"), help="source of NEI claims")
    parser.add_argument("--corpus", default=str(PROCESSED_DIR / "wiki_subset.jsonl"))
    parser.add_argument("--n-nei", type=int, default=250)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N claims (after shuffling)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-concat", action="store_true", help="score evidence sentences one by one only")
    parser.add_argument("--oracle-evidence", action="store_true",
                        help="use gold evidence for SUPPORTS/REFUTES claims (upper bound for NLI)")
    parser.add_argument("--out-dir", default=str(OUTPUTS_DIR / "task3"))
    args = parser.parse_args()

    claims = load_eval_claims(args.claims, args.dev, args.n_nei, args.seed)
    random.Random(args.seed).shuffle(claims)
    if args.limit:
        claims = claims[:args.limit]
    print(f"Claims: {len(claims)}  " + ", ".join(f"{lab}={sum(c['label'] == lab for c in claims)}" for lab in LABELS))

    pipe = EvidencePipeline.from_corpus_file(args.corpus)
    verifier = NLIVerifier(args.model, threshold=args.threshold, use_concat=not args.no_concat)

    sentence_lookup = {}
    if args.oracle_evidence:
        for page in pipe.retriever.pages:
            for line_id, text in page["sentences"]:
                sentence_lookup[(page["id"], int(line_id))] = {"title": page["title"], "text": text}

    rows = []
    n_oracle_missing = 0
    start = time.time()
    for c in tqdm(claims, desc="NLI"):
        evidence = pipe.run(c["claim"], top_k=args.top_k)
        nli_evidence = evidence
        if args.oracle_evidence and c["label"] != "NOT ENOUGH INFO":
            gold = gold_evidence(c, sentence_lookup)
            if gold is None:
                n_oracle_missing += 1
            else:
                nli_evidence = gold
        sentence_probs, concat_probs = verifier.score(c["claim"], nli_evidence)
        rows.append({
            "id": c["id"],
            "claim": c["claim"],
            "label": c["label"],
            "gold_evidence": c["evidence_sets"],
            "evidence_keys": [(e["page_id"], e["line_id"]) for e in nli_evidence],
            "evidence": [{"page_id": e["page_id"], "line_id": e["line_id"], "title": e["title"], "text": e["text"]}
                         for e in nli_evidence],
            "sentence_probs": sentence_probs.tolist(),
            "concat_probs": None if concat_probs is None else concat_probs.tolist(),
        })
    elapsed = time.time() - start
    if args.oracle_evidence:
        print(f"Oracle: {n_oracle_missing} S/R claims have no gold set fully in the corpus, used retrieval instead")

    # --- fixed threshold on all claims ---
    fixed = compute_metrics(rows, args.threshold)
    print_metrics("All claims, fixed threshold", fixed)

    # --- tune threshold on one half, report on the other ---
    half = len(rows) // 2
    tune_rows, test_rows = rows[:half], rows[half:]
    sweep = [compute_metrics(tune_rows, t) for t in THRESHOLDS]
    best = max(sweep, key=lambda m: m["macro_f1"])
    print("\nThreshold sweep on the tune half (macro F1 / FEVER score):")
    for m in sweep:
        mark = "  <- best" if m is best else ""
        print(f"  {m['threshold']:.2f}  {m['macro_f1']:.4f}  {m['fever_score']:.4f}{mark}")
    tuned = compute_metrics(test_rows, best["threshold"])
    print_metrics("Test half, tuned threshold", tuned)
    print(f"\nTime: {elapsed:.1f} s ({elapsed / max(len(rows), 1):.2f} s/claim)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "oracle" if args.oracle_evidence else "retrieved"
    with open(out_dir / f"metrics_{tag}.json", "w") as f:
        json.dump({"args": vars(args), "elapsed_s": elapsed, "fixed_threshold": fixed,
                   "tuned_threshold": best["threshold"], "tuned_test_half": tuned,
                   "sweep_tune_half": [{k: m[k] for k in ("threshold", "accuracy", "macro_f1", "fever_score")}
                                       for m in sweep]}, f, indent=2)
    with open(out_dir / f"predictions_{tag}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            label, scores = aggregate(r["sentence_probs"], r["concat_probs"], args.threshold)
            f.write(json.dumps({**r, "predicted": label, "scores": scores}, ensure_ascii=False) + "\n")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
