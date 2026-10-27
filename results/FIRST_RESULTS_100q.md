# First results (run on Sumit's MacBook, 27 Oct 2026)

`python run.py --n-queries 100 --beams 2`. It took 18.4 minutes on CPU.

100 MIRACL Japanese dev questions, pool of 936 judged passages.

| Question language | System | nDCG@10 | Recall@20 | Recall@100 | nDCG@10 vs Japanese |
|---|---|---|---|---|---|
| Japanese (same language) | BM25 | 0.688 | 0.983 | 0.990 | - |
| Japanese (same language) | Dense mE5 | 0.818 | 1.000 | 1.000 | - |
| Japanese (same language) | Hybrid RRF | 0.767 | 1.000 | 1.000 | - |
| English | BM25 (query translated to ja) | 0.557 | 0.825 | 0.875 | -19.0% |
| English | Dense mE5 | 0.680 | 0.908 | 0.952 | -16.9% |
| English | Hybrid RRF | 0.641 | 0.908 | 0.955 | -16.4% |
| Hindi | BM25 (query translated to ja) | 0.540 | 0.793 | 0.861 | -21.5% |
| Hindi | Dense mE5 | 0.644 | 0.885 | 0.947 | -21.3% |
| Hindi | Hybrid RRF | 0.619 | 0.871 | 0.932 | -19.3% |

- **English failures:** 13 questions where Japanese worked but English failed (see `results/failures_en.tsv`).
- **Hindi failures:** 15 questions where Japanese worked but Hindi failed (see `results/failures_hi.tsv`).

## What to say if a sensei asks
- **What was run.** It is a small first test with 100 questions, translated by machine with NLLB-200 (beam 2). It searched a pool of judged passages, not the full 6.9M-passage corpus.
- **Main finding.** Every method loses accuracy across languages, and Hindi loses more than English.
- **A surprise.** Simple rank fusion scored lower than multilingual E5 alone, because BM25 is much weaker and pulls the fused ranking down.
- **Next step.** Read the failed questions by hand, then repeat the test with 500 questions and 4-beam translation.
