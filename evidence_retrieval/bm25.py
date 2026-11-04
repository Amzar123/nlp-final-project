import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class BM25:
    def __init__(self, k1=1.5, b=0.75, stop_words="english"):
        self.k1 = k1
        self.b = b
        self.vectorizer = CountVectorizer(
            lowercase=True,
            stop_words=stop_words,
            token_pattern=r"(?u)\b\w+\b",
            dtype=np.float32,
        )
        self.matrix = None
        self.idf = None

    def fit(self, texts):
        tf = self.vectorizer.fit_transform(texts).tocsr()
        n_docs = tf.shape[0]

        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = doc_len.mean() if n_docs else 0.0

        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        nnz_per_row = np.diff(tf.indptr)
        len_per_value = np.repeat(doc_len, nnz_per_row)
        denom = tf.data + self.k1 * (1 - self.b + self.b * len_per_value / avgdl)
        tf.data = tf.data * (self.k1 + 1) / denom

        self.matrix = tf.tocsc()
        return self

    def get_scores(self, query):
        q = self.vectorizer.transform([query])
        term_ids = q.indices
        if len(term_ids) == 0:
            return np.zeros(self.matrix.shape[0], dtype=np.float32)
        weights = self.idf[term_ids] * q.data
        scores = self.matrix[:, term_ids] @ weights
        return np.asarray(scores).ravel()

    def search(self, query, top_k=10):
        scores = self.get_scores(query)
        return top_k_indices(scores, top_k)


def top_k_indices(scores, k):
    k = min(k, len(scores))
    if k == 0:
        return []

    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]
