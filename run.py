"""Cross-lingual search prototype: English / Hindi questions -> Japanese passages.

Steps
 1. Take N Japanese questions from MIRACL (dev) and pool their judged passages.
 2. Machine-translate the questions into English and Hindi (NLLB-200).
 3. Search the Japanese pool with:
      BM25   - keyword search; English/Hindi questions are first translated back into Japanese
      Dense  - multilingual-E5, no translation needed
      Hybrid - reciprocal rank fusion of the two
 4. Score with nDCG@10, Recall@20, Recall@100 and save failed examples for reading by hand.

Usage
  python run.py --synthetic          # 10-second test, no downloads
  python run.py                      # real run, 500 questions
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np

from src.data import load_miracl_ja, synthetic_data
from src.metrics import hit_at_k, ndcg_at_k, recall_at_k
from src.retrieval import BM25, Dense, FakeDense, rrf
from src.translate import FakeTranslator, Translator

LANG_NAME = {"ja": "Japanese (same language)", "en": "English", "hi": "Hindi"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dense-model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--mt-model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results")
    ap.add_argument("--synthetic", action="store_true", help="fake data and models, for testing only")
    a = ap.parse_args()

    out = a.out + ("_synthetic" if a.synthetic else "")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    # 1. data
    if a.synthetic:
        queries, qrels, corpus = synthetic_data()
    else:
        print("Loading MIRACL Japanese dev set...")
        queries, qrels, corpus = load_miracl_ja(a.n_queries, a.seed)
    qids = list(queries)
    doc_ids = list(corpus)
    doc_texts = [(corpus[d]["title"] + " " + corpus[d]["text"]).strip() for d in doc_ids]
    n_pos = np.mean([len(qrels[q]) for q in qids])
    print(f"{len(qids)} questions, {len(doc_ids)} passages in the pool, {n_pos:.1f} relevant passages per question")

    # 2. translation
    tr = FakeTranslator() if a.synthetic else Translator(a.mt_model, os.path.join(out, "translation_cache.json"),
                                                       a.device, a.beams)
    ja = [queries[q] for q in qids]
    q_text = {"ja": ja}
    q_back = {"ja": ja}  # Japanese version used by BM25
    for lang in ("en", "hi"):
        print(f"Translating questions ja -> {lang} ...")
        q_text[lang] = tr.translate(ja, "ja", lang)
        print(f"Translating {lang} -> ja for the BM25 (query translation) setting ...")
        q_back[lang] = tr.translate(q_text[lang], lang, "ja")
    with open(os.path.join(out, "queries.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["qid", "ja", "en", "hi", "en_back_to_ja", "hi_back_to_ja"])
        for i, q in enumerate(qids):
            w.writerow([q, ja[i], q_text["en"][i], q_text["hi"][i], q_back["en"][i], q_back["hi"][i]])

    # 3. retrieval
    print("Building BM25 index...")
    bm25 = BM25(doc_ids, doc_texts)
    dense = (FakeDense if a.synthetic else Dense)(a.dense_model, doc_ids, doc_texts, device=a.device)
    runs = {}
    for lang in ("ja", "en", "hi"):
        print(f"Searching with {LANG_NAME[lang]} questions...")
        b = [bm25.search(t) for t in q_back[lang]]
        d = dense.search_many(q_text[lang])
        h = [rrf([x, y]) for x, y in zip(b, d)]
        runs[(lang, "BM25 (query translated to ja)" if lang != "ja" else "BM25")] = b
        runs[(lang, "Dense mE5")] = d
        runs[(lang, "Hybrid RRF")] = h

    # 4. scoring
    rows, per_query = [], {}
    for (lang, system), rankings in runs.items():
        nd = [ndcg_at_k(r, qrels[q], 10) for r, q in zip(rankings, qids)]
        r20 = [recall_at_k(r, qrels[q], 20) for r, q in zip(rankings, qids)]
        r100 = [recall_at_k(r, qrels[q], 100) for r, q in zip(rankings, qids)]
        per_query[(lang, system)] = nd
        rows.append({"question_language": LANG_NAME[lang], "system": system, "nDCG@10": np.mean(nd),
                     "Recall@20": np.mean(r20), "Recall@100": np.mean(r100), "n": len(qids)})

    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # gap vs Japanese, same system
    lines = ["# Results", "",
             f"{len(qids)} MIRACL Japanese dev questions, pool of {len(doc_ids)} judged passages"
             + (" (SYNTHETIC TEST DATA - not real results)" if a.synthetic else "") + ".", "",
             "| Question language | System | nDCG@10 | Recall@20 | Recall@100 | nDCG@10 vs Japanese |",
             "|---|---|---|---|---|---|"]
    base = {r["system"].split(" (")[0]: r["nDCG@10"] for r in rows if r["question_language"].startswith("Japanese")}
    for r in rows:
        b = base.get(r["system"].split(" (")[0], 0)
        gap = "-" if r["question_language"].startswith("Japanese") else f"{(r['nDCG@10'] - b) / b * 100:+.1f}%" if b else "n/a"
        lines.append(f"| {r['question_language']} | {r['system']} | {r['nDCG@10']:.3f} | {r['Recall@20']:.3f} | "
                     f"{r['Recall@100']:.3f} | {gap} |")
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # failures: Japanese dense finds a relevant passage in top 10, the other language does not
    title = {d: corpus[d]["title"] for d in doc_ids}
    for lang in ("en", "hi"):
        ja_run, xx_run = runs[("ja", "Dense mE5")], runs[(lang, "Dense mE5")]
        with open(os.path.join(out, f"failures_{lang}.tsv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["qid", "ja_question", f"{lang}_question", "correct_passage_title", f"{lang}_top1_title",
                        "your_error_type (fill in by hand)"])
            n = 0
            for i, q in enumerate(qids):
                if hit_at_k(ja_run[i], qrels[q], 10) and not hit_at_k(xx_run[i], qrels[q], 10):
                    gold = next(iter(qrels[q]))
                    w.writerow([q, ja[i], q_text[lang][i], title[gold], title[xx_run[i][0]], ""])
                    n += 1
        print(f"{n} questions where Japanese worked but {LANG_NAME[lang]} failed -> failures_{lang}.tsv")

    # translation check sample
    rng = random.Random(a.seed)
    sample = rng.sample(range(len(qids)), min(30, len(qids)))
    with open(os.path.join(out, "translation_check.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["qid", "ja", "en", "hi", "hi_ok? (y/n, fill in by hand)", "en_ok? (y/n)"])
        for i in sample:
            w.writerow([qids[i], ja[i], q_text["en"][i], q_text["hi"][i], "", ""])

    json.dump({"n_queries": len(qids), "n_passages": len(doc_ids), "seed": a.seed, "dense_model": a.dense_model,
               "mt_model": a.mt_model, "beams": a.beams, "synthetic": a.synthetic,
               "minutes": round((time.time() - t0) / 60, 1)},
              open(os.path.join(out, "config.json"), "w"), indent=2)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Results are in ./{out}/")


if __name__ == "__main__":
    main()
