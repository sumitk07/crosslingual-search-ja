"""Standard IR measures with binary relevance."""
import math


def ndcg_at_k(ranking, relevant, k=10):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranking[:k]) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def recall_at_k(ranking, relevant, k):
    return len(set(ranking[:k]) & set(relevant)) / len(relevant) if relevant else 0.0


def hit_at_k(ranking, relevant, k):
    return 1.0 if set(ranking[:k]) & set(relevant) else 0.0
