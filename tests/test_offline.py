"""Offline checks (no downloads). Run: python -m tests.test_offline"""
from src.metrics import ndcg_at_k, recall_at_k, hit_at_k
from src.retrieval import BM25, ja_bigrams, rrf


def test_metrics():
    assert ndcg_at_k(["a", "b"], {"a"}, 10) == 1.0
    assert abs(ndcg_at_k(["x", "a"], {"a"}, 10) - 0.6309) < 1e-3
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, 2) == 0.5
    assert hit_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_bigrams():
    assert ja_bigrams("住民票") == ["住民", "民票"]


def test_bm25_finds_match():
    ids = ["d1", "d2", "d3"]
    texts = ["住民票の取り方", "ゴミ出しの曜日", "国民健康保険の手続き"]
    assert BM25(ids, texts).search("住民票はどこで取れますか", k=1) == ["d1"]


def test_rrf():
    assert rrf([["a", "b"], ["b", "a"]])[:2] in (["a", "b"], ["b", "a"])
    assert rrf([["a", "b"], ["b", "c"]])[0] == "b"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
