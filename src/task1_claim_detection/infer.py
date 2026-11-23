import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..paths import MODELS_DIR
from ..utils import Device

DEFAULT_MODEL_DIR = MODELS_DIR / "task1_roberta-base_label" / "model"
SPACY_EXCLUDE = ["ner", "lemmatizer", "attribute_ruler", "tagger"]

class ClaimDetector:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, threshold=None, device=None, batch_size=64, max_length=128, spacy_model="en_core_web_sm"):
        model_dir = Path(model_dir)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"Task 1 model not found at {model_dir}.\n"
                "Trained models are not in git. Download task1_roberta-base_label.zip (from the shared "
                "Drive, produced by notebooks/task1_colab.ipynb) and unzip it in the project root, "
                "or pass --claim-model-dir. To skip claim detection, run with --no-claim-detection.\n"
                "See src/task1_claim_detection/README.md (Training on Google Colab)."
            )
        self.device = device or Device.auto()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.batch_size = batch_size
        self.max_length = max_length
        self.spacy_model = spacy_model
        self._nlp = None

        threshold_file = model_dir / "threshold.json"
        if threshold is None:
            threshold = json.loads(threshold_file.read_text())["threshold"] if threshold_file.exists() else 0.5
        self.threshold = float(threshold)

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(self.spacy_model, exclude=SPACY_EXCLUDE)
        return self._nlp

    def split_sentences(self, text):
        return [s.text.strip() for s in self.nlp(text).sents if s.text.strip()]

    @torch.no_grad()
    def score(self, sentences):
        """P(claim) for each sentence."""
        probs = []
        for i in range(0, len(sentences), self.batch_size):
            batch = self.tokenizer(sentences[i:i + self.batch_size], truncation=True, max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
            logits = self.model(**batch).logits.float()
            probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(probs) if probs else np.zeros(0)

    def detect(self, text, threshold=None, return_all=False):
        threshold = self.threshold if threshold is None else threshold
        sentences = self.split_sentences(text)
        results = [
            {"index": i, "sentence": s, "p_claim": float(p), "is_claim": bool(p >= threshold)}
            for i, (s, p) in enumerate(zip(sentences, self.score(sentences)))
        ]
        return results if return_all else [r for r in results if r["is_claim"]]
