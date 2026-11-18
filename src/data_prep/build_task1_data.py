"""Build Task 1 (claim detection) data from ClaimBuster (plan step 1.1).

    python -m src.data_prep.build_task1_data

Labels:
  label             = 1 for CFS and UFS (Verdict 1 or 0), 0 for NFS (Verdict -1)   <- main mapping
  label_checkworthy = 1 for CFS only (original ClaimBuster mapping, comparison run in 1.3)

Splits: crowdsourced.csv -> train / val (stratified on Verdict, so both mappings stay balanced),
groundtruth.csv -> test. Sentences whose text also appears in the test set are removed from
crowdsourced, and duplicate texts inside crowdsourced are collapsed (dropped if their verdicts disagree).

Outputs in data/processed/: task1_train.parquet, task1_val.parquet, task1_test.parquet, task1_meta.json
"""
import argparse
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_prep.download import CLAIMBUSTER_DIR, RAW_DIR

PROCESSED_DIR = RAW_DIR.parent / "processed"
LABEL_COLUMNS = ("label", "label_checkworthy")
KEEP_COLUMNS = ["sentence_id", "text", "verdict", "label", "label_checkworthy", "speaker", "file_id"]


def normalize(text):
    return " ".join(str(text).lower().split())


def load(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"Sentence_id": "sentence_id", "Text": "text", "Verdict": "verdict",
                            "Speaker": "speaker", "File_id": "file_id"})
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].copy()
    df["verdict"] = df["verdict"].astype(int)
    df["label"] = (df["verdict"] >= 0).astype(int)
    df["label_checkworthy"] = (df["verdict"] == 1).astype(int)
    df["norm"] = df["text"].map(normalize)
    return df


def dedupe(df):
    """Collapse duplicate texts. Drop the text entirely if its copies have different verdicts."""
    verdicts_per_text = df.groupby("norm")["verdict"].nunique()
    conflicting = set(verdicts_per_text[verdicts_per_text > 1].index)
    kept = df[~df["norm"].isin(conflicting)].drop_duplicates("norm", keep="first")
    return kept, len(conflicting), len(df) - len(kept)


def class_weights(labels):
    """n / (n_classes * count_c), so the loss gives both classes equal total weight."""
    counts = labels.value_counts().sort_index()
    return [float(len(labels) / (len(counts) * counts[c])) for c in counts.index]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(PROCESSED_DIR))
    args = parser.parse_args()

    crowd = load(CLAIMBUSTER_DIR / "crowdsourced.csv")
    test = load(CLAIMBUSTER_DIR / "groundtruth.csv")
    stats = {"crowdsourced_raw": len(crowd), "groundtruth_raw": len(test)}

    in_test = crowd["norm"].isin(set(test["norm"]))
    stats["removed_overlap_with_test"] = int(in_test.sum())
    crowd = crowd[~in_test]

    crowd, stats["conflicting_duplicate_texts"], stats["removed_duplicates"] = dedupe(crowd)

    train, val = train_test_split(crowd, test_size=args.val_size, random_state=args.seed, stratify=crowd["verdict"])

    out = {"train": train, "val": val, "test": test}
    for name, df in out.items():
        stats[f"{name}_size"] = len(df)
        stats[f"{name}_verdict_counts"] = {int(k): int(v) for k, v in df["verdict"].value_counts().sort_index().items()}
        for col in LABEL_COLUMNS:
            stats[f"{name}_{col}_positive_rate"] = round(float(df[col].mean()), 4)

    meta = {
        "seed": args.seed,
        "val_size": args.val_size,
        "label_mapping": {"label": "CFS(1) + UFS(0) -> 1, NFS(-1) -> 0",
                          "label_checkworthy": "CFS(1) -> 1, UFS(0) + NFS(-1) -> 0"},
        "class_weights": {col: class_weights(train[col]) for col in LABEL_COLUMNS},
        "stats": stats,
    }

    out_dir = args.out_dir
    for name, df in out.items():
        df[KEEP_COLUMNS].reset_index(drop=True).to_parquet(f"{out_dir}/task1_{name}.parquet", index=False)
    with open(f"{out_dir}/task1_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
