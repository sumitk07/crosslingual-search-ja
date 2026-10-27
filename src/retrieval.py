"""Three retrievers: BM25 (keyword), dense (multilingual E5), and a hybrid (reciprocal rank fusion)."""
import hashlib
import re

import numpy as np


def ja_bigrams(text):
    """Japanese has no spaces, so we split into overlapping 2-character pieces (a common simple choice)."""
    t = re.sub(r"\s+", "", text.lower())
    return [t[i:i + 2] for i in range(len(t) - 1)] or [t]


class BM25:
    def __init__(self, doc_ids, doc_texts):
        from rank_bm25 import BM25Okapi
        self.ids = doc_ids
        self.bm25 = BM25Okapi([ja_bigrams(t) for t in doc_texts])

    def search(self, query, k=100):
        scores = self.bm25.get_scores(ja_bigrams(query))
        top = np.argsort(-scores)[:k]
        return [self.ids[i] for i in top]


class Dense:
    """multilingual-E5 needs the prefixes 'query: ' and 'passage: '."""

    def __init__(self, model_name, doc_ids, doc_texts, device="cpu", batch_size=32):
        from sentence_transformers import SentenceTransformer
        print(f"Loading dense model {model_name}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.ids, self.bs = doc_ids, batch_size
        print(f"Encoding {len(doc_texts)} passages (a few minutes on CPU)...")
        self.doc_emb = self.model.encode(["passage: " + t for t in doc_texts], batch_size=batch_size,
                                         normalize_embeddings=True, show_progress_bar=True)

    def search_many(self, queries, k=100):
        q = self.model.encode(["query: " + t for t in queries], batch_size=self.bs, normalize_embeddings=True)
        scores = q @ self.doc_emb.T
        return [[self.ids[i] for i in np.argsort(-s)[:k]] for s in scores]


class FakeDense:
    """For --synthetic tests only: hashed character-bigram vectors, no model download."""

    def __init__(self, model_name, doc_ids, doc_texts, **_):
        self.ids = doc_ids
        self.doc_emb = np.stack([self._vec(t) for t in doc_texts])

    @staticmethod
    def _vec(text, dim=512):
        v = np.zeros(dim)
        for g in ja_bigrams(text.split("] ", 1)[-1]):
            v[int(hashlib.md5(g.encode()).hexdigest(), 16) % dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def search_many(self, queries, k=100):
        q = np.stack([self._vec(t) for t in queries])
        return [[self.ids[i] for i in np.argsort(-s)[:k]] for s in q @ self.doc_emb.T]


def rrf(rankings, k=60, depth=100):
    """Reciprocal rank fusion of several ranked lists (Cormack et al., SIGIR 2009)."""
    score = {}
    for ranking in rankings:
        for r, d in enumerate(ranking):
            score[d] = score.get(d, 0.0) + 1.0 / (k + r + 1)
    return [d for d, _ in sorted(score.items(), key=lambda x: -x[1])[:depth]]
