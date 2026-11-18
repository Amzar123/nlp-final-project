"""Train the Task 1 claim detector (plan steps 1.2, 1.3, threshold part of 1.4).

    # main run: factual (CFS + UFS) vs non-factual
    python -m src.task1_claim_detection.train

    # comparison run with the original check-worthiness mapping (CFS only = 1)
    python -m src.task1_claim_detection.train --label-col label_checkworthy

    # on a CUDA GPU (Colab): full batch, fp16 is switched on automatically
    python -m src.task1_claim_detection.train --batch-size 32 --grad-accum 1

    # quick pipeline test
    python -m src.task1_claim_detection.train --model hf-internal-testing/tiny-random-RobertaForSequenceClassification --limit 400 --max-steps 20

Weighted cross-entropy (class weights from task1_meta.json). The best step is chosen by
validation macro F1 and written straight to model/ (one copy only, no Trainer checkpoints).
Defaults (batch 16 x grad-accum 2) fit roberta-base on an 8 GB Apple M1.
After training, the decision threshold on P(claim) is tuned on validation, and the test set
(groundtruth.csv) is scored once at threshold 0.5 and once at the tuned threshold.

Output in models/task1_<model>_<label-col>/: model/ (weights + tokenizer + threshold.json),
metrics.json, test_predictions.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
ID2LABEL = {0: "non_factual", 1: "claim"}


class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.encodings.items()}
        item["labels"] = int(self.labels[i])
        return item


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs.logits, labels, weight=self.class_weights.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss


class SaveBestModel(TrainerCallback):
    """Overwrite <save_dir> with the model whenever the validation metric improves.

    Only one copy is ever on disk, and nothing extra is held in (unified) memory, which matters
    on 8 GB Apple Silicon machines.
    """

    def __init__(self, save_dir, metric="eval_macro_f1"):
        self.save_dir = Path(save_dir)
        self.metric = metric
        self.best_score = -np.inf
        self.best_step = None

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        score = metrics.get(self.metric)
        if score is not None and score > self.best_score:
            self.best_score = score
            self.best_step = state.global_step
            model.save_pretrained(self.save_dir)


def softmax_pos(logits):
    return torch.softmax(torch.as_tensor(logits, dtype=torch.float32), dim=-1)[:, 1].numpy()


def binary_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "accuracy": accuracy_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan"),
        "threshold": float(threshold),
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return binary_metrics(np.asarray(labels), softmax_pos(logits))


def tune_threshold(labels, probs):
    """Threshold on P(claim) that maximizes macro F1 on the given (validation) data."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [f1_score(labels, (probs >= t).astype(int), average="macro") for t in grid]
    return float(grid[int(np.argmax(scores))])


def predict_probs(trainer, dataset):
    return softmax_pos(trainer.predict(dataset).predictions)


def load_split(name, label_col, limit, seed):
    df = pd.read_parquet(PROCESSED_DIR / f"task1_{name}.parquet")
    if limit and limit < len(df):
        df = df.sample(n=limit, random_state=seed)
    return df.reset_index(drop=True), df[label_col].to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--label-col", choices=["label", "label_checkworthy"], default="label")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16, help="per step; effective batch = batch-size * grad-accum")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--no-fp16", action="store_true", help="disable mixed precision on CUDA (it is on by default there)")
    parser.add_argument("--limit", type=int, default=None, help="subsample every split (debug)")
    parser.add_argument("--max-steps", type=int, default=-1, help="stop after this many steps (debug)")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    run_name = f"task1_{args.model.split('/')[-1]}_{args.label_col}"
    out_dir = Path(args.out_dir) if args.out_dir else MODELS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((PROCESSED_DIR / "task1_meta.json").read_text())
    class_weights = [1.0, 1.0] if args.no_class_weights else meta["class_weights"][args.label_col]
    print(f"Run {run_name} | class weights {class_weights}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label=ID2LABEL, label2id={v: k for k, v in ID2LABEL.items()}
    )
    # Some checkpoints (e.g. microsoft/deberta-v3-small) are stored in fp16 and recent transformers
    # versions load them as fp16. fp16 mixed-precision training needs fp32 weights, otherwise the
    # grad scaler fails with "Attempting to unscale FP16 gradients".
    model = model.float()

    splits = {name: load_split(name, args.label_col, args.limit, args.seed) for name in ("train", "val", "test")}
    datasets = {
        name: SentenceDataset(df["text"], labels, tokenizer, args.max_length)
        for name, (df, labels) in splits.items()
    }
    print({name: len(ds) for name, ds in datasets.items()})

    steps_per_epoch = int(np.ceil(len(datasets["train"]) / (args.batch_size * args.grad_accum)))
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    eval_steps = min(args.eval_steps, max(1, total_steps // 2))

    training_args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.batch_size * 2,
        warmup_steps=int(args.warmup_ratio * total_steps),
        weight_decay=args.weight_decay,
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=max(1, eval_steps // 4),
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        fp16=torch.cuda.is_available() and not args.no_fp16,
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    model_dir = out_dir / "model"
    keep_best = SaveBestModel(model_dir)
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[keep_best],
        class_weights=class_weights,
    )
    trainer.train()

    print(f"Best validation macro F1 {keep_best.best_score:.4f} at step {keep_best.best_step}")
    best = AutoModelForSequenceClassification.from_pretrained(model_dir)
    trainer.model.load_state_dict(best.state_dict())
    del best

    val_labels, test_labels = splits["val"][1], splits["test"][1]
    val_probs = predict_probs(trainer, datasets["val"])
    test_probs = predict_probs(trainer, datasets["test"])
    threshold = tune_threshold(val_labels, val_probs)

    results = {
        "run": run_name,
        "args": vars(args),
        "class_weights": class_weights,
        "best_step": keep_best.best_step,
        "tuned_threshold": threshold,
        "val@0.5": binary_metrics(val_labels, val_probs, 0.5),
        "val@tuned": binary_metrics(val_labels, val_probs, threshold),
        "test@0.5": binary_metrics(test_labels, test_probs, 0.5),
        "test@tuned": binary_metrics(test_labels, test_probs, threshold),
        "log_history": trainer.state.log_history,
    }

    tokenizer.save_pretrained(model_dir)  # weights were already saved by SaveBestModel
    (model_dir / "threshold.json").write_text(json.dumps(
        {"threshold": threshold, "label_col": args.label_col, "selected_on": "val macro F1"}, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    test_df = splits["test"][0][["sentence_id", "text", "verdict", args.label_col]].copy()
    test_df["p_claim"] = test_probs
    test_df["pred"] = (test_probs >= threshold).astype(int)
    test_df.to_csv(out_dir / "test_predictions.csv", index=False)

    print(f"\nTuned threshold (val): {threshold:.2f}")
    print(f"{'split':<12}{'macroF1':>9}{'prec':>8}{'recall':>8}{'rocAUC':>8}")
    for key in ("val@0.5", "val@tuned", "test@0.5", "test@tuned"):
        m = results[key]
        print(f"{key:<12}{m['macro_f1']:>9.4f}{m['precision']:>8.4f}{m['recall']:>8.4f}{m['roc_auc']:>8.4f}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
