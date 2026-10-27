"""Load a small, fixed test set from MIRACL Japanese (dev split).

MIRACL gives, for each Japanese question, passages that native speakers judged
relevant (positive) or not relevant (negative). We take N questions and put all
their judged passages into one shared pool. Every question is then searched
against the whole pool, so the other questions' passages act as distractors.

This is a small "pooled" test set, not the full 6.9M-passage Japanese corpus.
Report it that way.
"""
import random


def load_miracl_ja(n_queries=500, seed=0):
    from datasets import load_dataset  # imported here so --synthetic runs without it

    try:
        ds = load_dataset("miracl/miracl", "ja", split="dev", trust_remote_code=True)
    except Exception as e:  # newest "datasets" versions dropped loading scripts
        raise SystemExit(
            f"\nCould not load MIRACL ({type(e).__name__}: {e}).\n"
            "Fix: pip install \"datasets>=2.19,<4\"  and run again.\n")
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)

    queries, qrels, corpus = {}, {}, {}
    for i in order:
        row = ds[i]
        pos = row["positive_passages"]
        if not pos:
            continue
        qid = str(row["query_id"])
        queries[qid] = row["query"]
        qrels[qid] = {p["docid"]: 1 for p in pos}
        for p in pos + row["negative_passages"]:
            corpus[p["docid"]] = {"title": p["title"], "text": p["text"]}
        if len(queries) >= n_queries:
            break
    return queries, qrels, corpus


def synthetic_data(n_queries=40, seed=0):
    """Tiny fake data set, only for testing the code without downloads."""
    rng = random.Random(seed)
    topics = ["住民票", "国民健康保険", "ゴミ出し", "在留カード", "マイナンバー", "年金", "地震", "病院",
              "銀行口座", "運転免許", "税金", "保育園", "電気代", "引っ越し", "ビザ", "郵便局"]
    queries, qrels, corpus = {}, {}, {}
    for q in range(n_queries):
        t = topics[q % len(topics)]
        qid = f"q{q}"
        queries[qid] = f"{t}の手続きはどうすればいいですか{q}"
        pos_id = f"d{q}_pos"
        corpus[pos_id] = {"title": t, "text": f"{t}の手続きについて説明します。窓口で申請してください。番号{q}"}
        qrels[qid] = {pos_id: 1}
        for k in range(4):
            other = topics[rng.randrange(len(topics))]
            corpus[f"d{q}_neg{k}"] = {"title": other, "text": f"{other}に関する一般的な情報です。番号{rng.randrange(999)}"}
    return queries, qrels, corpus
