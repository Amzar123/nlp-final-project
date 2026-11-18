"""Non-neural baseline for Task 1: TF-IDF + logistic regression.

    python -m src.task1_claim_detection.baseline

Same splits, label mappings, metrics and threshold tuning as train.py, so the numbers can be
put in the same table as the transformer. Runs in about a minute on a laptop CPU.
Output: models/task1_tfidf-logreg/metrics.json
"""
import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .train import MODELS_DIR, binary_metrics, load_split, tune_threshold

LABEL_COLUMNS = ("label", "label_checkworthy")


def run(label_col, seed, c_values):
    (train_df, y_train), (val_df, y_val), (test_df, y_test) = (
        load_split(name, label_col, None, seed) for name in ("train", "val", "test")
    )

    best = None
    for c in c_values:  # pick the regularization strength on validation macro F1
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=seed),
        )
        model.fit(train_df["text"], y_train)
        val_probs = model.predict_proba(val_df["text"])[:, 1]
        score = binary_metrics(y_val, val_probs, 0.5)["macro_f1"]
        if best is None or score > best[0]:
            best = (score, c, model, val_probs)

    _, c, model, val_probs = best
    test_probs = model.predict_proba(test_df["text"])[:, 1]
    threshold = tune_threshold(y_val, val_probs)
    return {
        "C": c,
        "tuned_threshold": threshold,
        "val@0.5": binary_metrics(y_val, val_probs, 0.5),
        "val@tuned": binary_metrics(y_val, val_probs, threshold),
        "test@0.5": binary_metrics(y_test, test_probs, 0.5),
        "test@tuned": binary_metrics(y_test, test_probs, threshold),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = MODELS_DIR / "task1_tfidf-logreg"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {col: run(col, args.seed, c_values=(0.25, 1.0, 4.0, 16.0, 64.0)) for col in LABEL_COLUMNS}
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=float))

    print(f"{'mapping':<20}{'split':<12}{'macroF1':>9}{'prec':>8}{'recall':>8}{'rocAUC':>8}")
    for col, res in results.items():
        for key in ("val@tuned", "test@0.5", "test@tuned"):
            m = res[key]
            print(f"{col:<20}{key:<12}{m['macro_f1']:>9.4f}{m['precision']:>8.4f}{m['recall']:>8.4f}{m['roc_auc']:>8.4f}")
        print(f"{'':<20}C={res['C']}, tuned threshold={res['tuned_threshold']:.2f}")
    print(f"Saved {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
