import numpy as np

from ..task2_evidence_retrieval.data import FeverText
from ..utils import Device

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
NLI_KEYS = ("entailment", "neutral", "contradiction")


class NLIVerifier:
    def __init__(self, model_name=DEFAULT_MODEL, threshold=0.5, use_concat=True, device=None,
                 batch_size=16, max_length=512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.threshold = float(threshold)
        self.use_concat = use_concat
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or Device.auto()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()

        # Label order differs between NLI checkpoints, so read it from the config.
        id2label = {i: name.lower() for i, name in self.model.config.id2label.items()}
        self.column_order = []
        for key in NLI_KEYS:
            matches = [i for i, name in id2label.items() if name.startswith(key[:7])]
            if len(matches) != 1:
                raise ValueError(f"{model_name}: cannot find '{key}' in id2label {self.model.config.id2label}")
            self.column_order.append(matches[0])

    @staticmethod
    def format_premise(ev):
        """Evidence dict (or plain string) -> premise text."""
        if isinstance(ev, str):
            return FeverText.clean(ev)
        text = FeverText.clean(ev["text"])
        title = ev.get("title")
        return f"{title}: {text}" if title else text

    @staticmethod
    def aggregate(sentence_probs, concat_probs=None, threshold=0.5):
        """Turn NLI probabilities into a FEVER verdict. Needs no model, so evaluation can sweep thresholds cheaply.
        sentence_probs: array (n_evidence, 3) in NLI_KEYS order; concat_probs: array (3,) or None.
        Returns (label, scores) where scores has one value per FEVER label.
        """
        rows = [np.asarray(p, dtype=float) for p in sentence_probs]
        if concat_probs is not None:
            rows.append(np.asarray(concat_probs, dtype=float))
        if not rows:
            return "NOT ENOUGH INFO", {"SUPPORTS": 0.0, "REFUTES": 0.0, "NOT ENOUGH INFO": 1.0}

        probs = np.stack(rows)
        support = float(probs[:, 0].max())
        refute = float(probs[:, 2].max())
        scores = {"SUPPORTS": support, "REFUTES": refute, "NOT ENOUGH INFO": 1.0 - max(support, refute)}

        if max(support, refute) < threshold:
            return "NOT ENOUGH INFO", scores
        return ("SUPPORTS" if support >= refute else "REFUTES"), scores

    def predict_pairs(self, premises, hypotheses):
        """NLI probabilities, shape (n, 3) in NLI_KEYS order."""
        torch = self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(premises), self.batch_size):
                batch = self.tokenizer(
                    premises[i:i + self.batch_size], hypotheses[i:i + self.batch_size],
                    truncation="only_first", max_length=self.max_length, padding=True, return_tensors="pt",
                ).to(self.device)
                logits = self.model(**batch).logits.float()
                out.append(torch.softmax(logits, dim=-1)[:, self.column_order].cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 3))

    def score(self, claim, evidence):
        """Raw NLI probabilities for one claim: (sentence_probs (n, 3), concat_probs (3,) or None)."""
        premises = [self.format_premise(ev) for ev in evidence]
        if not premises:
            return np.zeros((0, 3)), None
        pairs = list(premises)
        if self.use_concat and len(premises) > 1:
            pairs.append(" ".join(premises))
        probs = self.predict_pairs(pairs, [claim] * len(pairs))
        if len(pairs) > len(premises):
            return probs[:-1], probs[-1]
        return probs, None

    def verify(self, claim, evidence, threshold=None):
        """Verdict for one claim given its evidence (list of evidence dicts or strings)."""
        threshold = self.threshold if threshold is None else threshold
        sentence_probs, concat_probs = self.score(claim, evidence)
        label, scores = self.aggregate(sentence_probs, concat_probs, threshold)

        scored_evidence = []
        for ev, p in zip(evidence, sentence_probs):
            ev = {"text": ev} if isinstance(ev, str) else dict(ev)
            ev["nli"] = {k: float(v) for k, v in zip(NLI_KEYS, p)}
            ev["nli_label"] = NLI_KEYS[int(np.argmax(p))]
            scored_evidence.append(ev)

        return {
            "claim": claim,
            "label": label,
            "confidence": scores[label],
            "scores": scores,
            "concat_nli": None if concat_probs is None else {k: float(v) for k, v in zip(NLI_KEYS, concat_probs)},
            "evidence": scored_evidence,
        }
