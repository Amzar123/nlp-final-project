from .retriever import get_device

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device=None, batch_size=32):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=device or get_device())

    def rerank(self, claim, candidates, top_k=5):
        if not candidates:
            return []
        pairs = [(claim, f"{c.title}: {c.text}") for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        for c, s in zip(candidates, scores):
            c.rerank_score = float(s)
        ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
        return ranked[:top_k]
