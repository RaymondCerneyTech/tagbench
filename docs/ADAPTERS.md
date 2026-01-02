# Adapter Interface

This guide covers the adapter contract, supported adapters, and tuning knobs.

## Plug in your model (adapter interface)

Implement a tiny adapter (module with `create_adapter()` returning an object that has `.predict(row, protocol="...") -> {"value": ..., "support_ids": [...]}`).

Reference adapter (wraps the ledger baseline): `goldevidencebench.adapters.ledger_adapter:create_adapter`.
Two-phase adapter example (build book once, answer many): `goldevidencebench.adapters.log_to_book_adapter:create_adapter` implements `build_artifact(document, episode_id, protocol)` then answers closed-book.
Closed-book Llama example (uses `llama-cpp-python`, set `GOLDEVIDENCEBENCH_MODEL` to a GGUF path): `goldevidencebench.adapters.llama_cpp_adapter:create_adapter`.
The Llama adapter extracts the `## State Ledger` section and keeps the most recent ledger tokens to fit context.
Streaming state-builder (chunked log -> compact ledger, then Llama answers): `goldevidencebench.adapters.streaming_llama_cpp_adapter:create_adapter`.
Set `GOLDEVIDENCEBENCH_STREAM_CHUNK_TOKENS` (default 512) to control chunk size. Set `GOLDEVIDENCEBENCH_STREAM_MODE=llm`
(default) to let the model extract updates per chunk, or `GOLDEVIDENCEBENCH_STREAM_MODE=parse` for a deterministic parser.
Teacher book-builder (LLM builds artifacts, answerer stays fixed): `goldevidencebench.adapters.llm_book_builder_adapter:create_adapter`.
Set `GOLDEVIDENCEBENCH_BUILDER_MODEL` to use a stronger model for artifact construction (defaults to `GOLDEVIDENCEBENCH_MODEL`).
Set `GOLDEVIDENCEBENCH_BUILDER_CHUNK_TOKENS` to control builder chunk size.
Set `GOLDEVIDENCEBENCH_BUILDER_MODE` to `heuristic`, `llm_fullscan` (default), or `llm_perkey`.
Set `GOLDEVIDENCEBENCH_BUILDER_PER_KEY_LLM=0` to disable per-key LLM calls (deterministic fallback).
Retrieval-first answerer (use only the latest ledger entry for the key): `goldevidencebench.adapters.retrieval_llama_cpp_adapter:create_adapter`.
Env var names use the `GOLDEVIDENCEBENCH_` prefix.
Set `GOLDEVIDENCEBENCH_RETRIEVAL_K` to include top-k latest entries for the key (default 1). Set
`GOLDEVIDENCEBENCH_RETRIEVAL_WRONG_TYPE` to `none`, `same_key`, or `other_key` to inject a wrong line for robustness
testing. Use `GOLDEVIDENCEBENCH_RETRIEVAL_INCLUDE_CLEAR=0` to skip CLEAR entries.
Use `GOLDEVIDENCEBENCH_RETRIEVAL_DROP_PROB` (0-1) to probabilistically drop the correct line, and
`GOLDEVIDENCEBENCH_RETRIEVAL_DROP_SEED` to make the drop deterministic by row id.
Use `GOLDEVIDENCEBENCH_RETRIEVAL_AUTHORITY_SPOOF_RATE` (0-1) to flip NOTE/UPDATE labels in the candidate set, and
`GOLDEVIDENCEBENCH_RETRIEVAL_AUTHORITY_SPOOF_SEED` for deterministic spoofing by row id.
Use `GOLDEVIDENCEBENCH_RETRIEVAL_ORDER=shuffle|gold_first|gold_middle|gold_last` (and optional
`GOLDEVIDENCEBENCH_RETRIEVAL_ORDER_SEED`) to control ordering and test positional bias under ambiguity.
Set `GOLDEVIDENCEBENCH_RETRIEVAL_QUERY_SANDWICH=1` to repeat the question before and after the
candidate ledger lines (query sandwich mitigation).
Set `GOLDEVIDENCEBENCH_RETRIEVAL_PICK_THEN_ANSWER=1` to force a two-step flow: pick a support_id
first, then answer using only that line.
Set `GOLDEVIDENCEBENCH_RETRIEVAL_RERANK=latest_step|last_occurrence|prefer_set_latest|linear` to deterministically
choose a candidate before answering (non-LLM selector baseline). For `linear`, also set
`GOLDEVIDENCEBENCH_RETRIEVAL_LINEAR_MODEL` to the JSON model file from `train_selector_linear.py`.
Set `GOLDEVIDENCEBENCH_RETRIEVAL_SELECTOR_ONLY=1` to skip answer generation and emit only `support_ids`.
Use selection metrics (`gold_present_rate`, `selection_rate`) for speed-focused iterations; value accuracy is not meaningful
in selector-only mode.

Retrieval order bias (example, k=4, s3q16, kv/standard, gold always present):

- `gold_first`: value/exact/cite_f1 = 0.0
- `gold_middle`: value/exact/cite_f1 = 0.4167
- `shuffle`: value/exact/cite_f1 = 0.4583
- `gold_last`: value/exact/cite_f1 = 1.0

Entailment stays 1.0 and `gold_in_context_rate` is 1.0, so the failures are positional
selection errors (not retrieval or hallucination).

Order-bias (LLM-only, k=4, same_key, shuffle, s5q24):

| order | selection_rate | accuracy_when_gold_present | value_acc |
| --- | --- | --- | --- |
| gold_first | 0.0000 | 0.0000 | 0.0000 |
| gold_middle | 0.0083 | 0.0083 | 0.0083 |
| gold_last | 1.0000 | 1.0000 | 1.0000 |
| shuffle | 0.4417 | 0.4417 | 0.4417 |

Gold is always present; the collapse in gold_first/middle is pure selection bias.
This aligns with the smaller s3q16 run and shows the effect is stable with more samples.

Plot the order-bias figure:

```powershell
python .\scripts\plot_order_bias.py --in-csv .\runs\summary_all.csv --out .\docs\figures\order_bias_s5q24_llm.png
```

![Order bias (LLM-only, k=4, same_key, s5q24)](docs/figures/order_bias_s5q24_llm.png)

Order-bias (selector on: latest_step, same settings):

| order | selection_rate | accuracy_when_gold_present | value_acc |
| --- | --- | --- | --- |
| gold_first | 1.0000 | 0.9750 | 0.9750 |
| gold_middle | 1.0000 | 0.9750 | 0.9750 |
| gold_last | 1.0000 | 0.9750 | 0.9750 |
| shuffle | 1.0000 | 0.9750 | 0.9750 |

Selector removes order bias in this regime.

Plot the reranker k-curve:

```powershell
python .\scripts\plot_rerank_curve.py --in-csv .\runs\summary_all.csv --out .\docs\figures\rerank_k_curve_s5q24.png
```

![Reranker k-curve (same_key, shuffle, s5q24)](docs/figures/rerank_k_curve_s5q24.png)

Multi-model spot check (Meta-Llama-3.1-8B-Instruct Q4_K_M vs Qwen, s3q16):

| preset | model | selection_rate | accuracy_when_gold_present | value_acc |
| --- | --- | --- | --- | --- |
| hard selection (same_key, k=4, shuffle) | Qwen 2.5 7B | 0.4167 | 0.4167 | 0.4167 |
| hard selection (same_key, k=4, shuffle) | Meta-Llama 3.1 8B Q4_K_M | 0.3125 | 0.2708 | 0.2708 |
| order bias (gold_first) | Qwen 2.5 7B | 0.0000 | 0.0000 | 0.0000 |
| order bias (gold_first) | Meta-Llama 3.1 8B Q4_K_M | 0.0417 | 0.0417 | 0.0417 |
| order bias (gold_last) | Qwen 2.5 7B | 1.0000 | 1.0000 | 1.0000 |
| order bias (gold_last) | Meta-Llama 3.1 8B Q4_K_M | 0.5208 | 0.2917 | 0.2917 |

Meta-Llama shows the same order-bias pattern but lower selection under ambiguity in these runs.
Drop sweep (k=4, shuffle): Meta-Llama selection_rate falls from 0.2708 (drop=0) to 0.1875 (drop=0.2) and 0.0833 (drop=0.4).

Stability check (k=4, s5q24, kv/standard, gold always present):

- `gold_first`: value/exact/cite_f1 = 0.0
- `gold_middle`: value/exact/cite_f1 = 0.25
- `shuffle`: value/exact/cite_f1 = 0.2167
- `gold_last`: value/exact/cite_f1 = 1.0

Retrieval remains perfect (gold present in all contexts), so the ordering effect persists
with more samples.

Takeaway: with perfect retrieval and identical evidence, answer accuracy is dominated by
line ordering, exposing a strong positional bias in selection.

Drop-prob sweep (k=4, shuffle, s5q24):

- `drop_prob=0.0`: value/exact/cite_f1 = 0.5083, drop_rate = 0.0
- `drop_prob=0.2`: value/exact/cite_f1 = 0.4333, drop_rate = 0.1667
- `drop_prob=0.4`: value/exact/cite_f1 = 0.4333, drop_rate = 0.2917

Entailment stays 1.0; accuracy tracks evidence loss, and the curve flattens once
selection ambiguity dominates.

When retrieval stats and predictions are available, summary output also reports:

- `accuracy_when_gold_present`: value accuracy conditioned on the gold line being present.
- `selection_rate`: fraction of gold-present rows where the prediction cites the gold line.
- `decomposition_line`: `gold_present_rate -> selection_rate -> accuracy_when_gold_present -> overall accuracy`.

Query-sandwich A/B (same_key, k=4, s3q16):

- sandwich off: gold_first 0.0, gold_middle 0.0208, gold_last 0.5625, shuffle 0.4583
- sandwich on: gold_first 0.0, gold_middle 0.0, gold_last 0.5625, shuffle 0.1875

Selection rate tracks accuracy_when_gold_present in these runs; sandwich does not improve
selection under ambiguity and can make shuffle worse.

Pick-then-answer A/B (same_key, k=4, s3q16):

- pick off: gold_first 0.0, gold_middle 0.0, gold_last 0.5625, shuffle 0.2083
- pick on: gold_first 0.0, gold_middle 0.0208, gold_last 0.4167, shuffle 0.2083

Selection rate matches accuracy_when_gold_present; pick-then-answer does not improve
selection in this regime and can lower gold_last.

Matched s5q24 run (same_key, k=4, shuffle): pick-then-answer on/off are identical.
Both show selection_rate 1.0 and accuracy_when_gold_present 0.975, so the prompt trick adds no benefit in this easier regime.

Reranker baseline (same_key, k=4, shuffle, s3q16): see the Reference proof table above.

Selector failure mode (kv_commentary: NOTE lines are non-authoritative):

NOTE-aware rerank: set `GOLDEVIDENCEBENCH_RETRIEVAL_RERANK=prefer_update_latest` to ignore NOTE lines unless no UPDATE-style lines exist.

```powershell
$env:GOLDEVIDENCEBENCH_RETRIEVAL_RERANK = "latest_step"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_K = "4"
goldevidencebench sweep --out runs --seeds 2 --episodes 1 --steps 120 --queries 8 `
  --state-modes kv_commentary --distractor-profiles standard `
  --adapter goldevidencebench.adapters.retrieval_llama_cpp_adapter:create_adapter --no-derived-queries `
  --no-twins --require-citations --results-json .\runs\combined.json --max-book-tokens 400 --note-rate 0.25
python .\scripts\summarize_results.py --in .\runs\combined.json --out-json .\runs\summary.json
```

Example outcome (kv_commentary, prefer_set_latest): value_acc 0.75, exact_acc 0.75, entailment 1.0. With naive latest_step in the same setting, value_acc was 0.25 and exact_acc 0.0, showing NOTE lines can break recency-based selectors and a simple policy fixes it.

KV commentary sanity check (from runs/summary_all.csv, matched A/B: same seeds + settings):

| preset | k | rerank | gold_present | selection_rate | value_acc | entailment |
| --- | --- | --- | --- | --- | --- | --- |
| custom | 4 | latest_step | 1 | 1.0 | 0.75 | 0.75 |
| custom | 4 | prefer_set_latest | 1 | 0.75 | 0.75 | 1.0 |

Matched A/B shows `latest_step` maximizes selection but can cite non-authoritative NOTE lines (entailment drops), while `prefer_set_latest` preserves entailment.

KV commentary selector A/B (s3q16, k=4, same_key, shuffle):

| preset | k | rerank | gold_present | selection_rate | accuracy_when_gold_present | value_acc | entailment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| custom | 4 | latest_step | 1.0 | 1.0 | 0.7292 | 0.7292 | 0.8542 |
| custom | 4 | prefer_set_latest | 1.0 | 0.7292 | 1.0 | 1.0 | 1.0 |
| custom | 4 | prefer_update_latest | 1.0 | 0.7292 | 1.0 | 1.0 | 1.0 |
| custom | 4 | linear | 1.0 | 1.0 | 0.7292 | 0.7292 | 0.8542 |

In kv_commentary, `prefer_set_latest` and `prefer_update_latest` keep end-to-end accuracy/entailment perfect even though selection_rate is lower, while the linear selector always chooses the gold line but still produces incorrect answers in ~27% of cases. This suggests the linear model is not robust to NOTE-line noise yet.

KV commentary selector bake-off (s5q24, k=4, same_key, shuffle):

| preset | k | rerank | gold_present | selection_rate | accuracy_when_gold_present | value_acc | entailment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| custom | 4 | prefer_set_latest | 1.0 | 0.5833 | 1.0 | 1.0 | 1.0 |
| custom | 4 | prefer_update_latest | 1.0 | 0.5833 | 1.0 | 1.0 | 1.0 |
| custom | 4 | linear | 1.0 | 0.9833 | 0.5667 | 0.5667 | 0.7250 |

KV commentary grid (s3q16, same_key, k in {2,4,8}):

| rerank | k | selection_rate | accuracy_when_gold_present | value_acc | entailment |
| --- | --- | --- | --- | --- | --- |
| prefer_set_latest | 2 | 0.7500 | 0.9792 | 0.9792 | 1.0000 |
| prefer_set_latest | 4 | 0.7292 | 1.0000 | 1.0000 | 1.0000 |
| prefer_set_latest | 8 | 0.7292 | 1.0000 | 1.0000 | 1.0000 |
| linear | 2 | 1.0000 | 0.7292 | 0.7292 | 0.8542 |
| linear | 4 | 1.0000 | 0.7292 | 0.7292 | 0.8542 |
| linear | 8 | 1.0000 | 0.7292 | 0.7292 | 0.8542 |

The learned selector still fails on NOTE authoritativeness across k, while prefer_set_latest stays perfect end-to-end.

Authority filter baseline (GOLDEVIDENCEBENCH_RETRIEVAL_AUTHORITY_FILTER=1, linear rerank, s3q16):

| k | selection_rate | accuracy_when_gold_present | value_acc | entailment |
| --- | --- | --- | --- | --- |
| 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Authority filter stability (s5q24):

| k | selection_rate | accuracy_when_gold_present | value_acc | entailment |
| --- | --- | --- | --- | --- |
| 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

With authority filtering, the selector becomes fully NOTE-robust across k.

| preset | k | rerank | gold_present | selection_rate | accuracy_when_gold_present | value_acc | entailment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| custom | 4 | prefer_set_latest | 1.0 | 0.5833 | 1.0 | 1.0 | 1.0 |
| custom | 4 | prefer_update_latest | 1.0 | 0.5833 | 1.0 | 1.0 | 1.0 |
| custom | 4 | linear | 1.0 | 0.9833 | 0.5667 | 0.5667 | 0.7250 |

At s5q24, NOTE-aware deterministic policies remain perfect end-to-end, while the linear selector still loses accuracy and entailment despite near-perfect selection.

KV selector bake-off (s5q24, k=4, same_key, shuffle):

| rerank | gold_present | selection_rate | accuracy_when_gold_present | value_acc | entailment |
| --- | --- | --- | --- | --- | --- |
| none | 1.0 | 0.3583 | 0.3583 | 0.3583 | 0.9917 |
| latest_step | 1.0 | 1.0 | 0.9750 | 0.9750 | 0.9750 |
| prefer_set_latest | 1.0 | 1.0 | 0.9750 | 0.9750 | 0.9750 |
| linear | 1.0 | 1.0 | 0.9750 | 0.9750 | 0.9750 |

![Compute vs quality (kv, s5q24)](docs/figures/compute_vs_quality_kv_s5q24.png)

Multi-model check (kv, latest_step, s5q24):

| model | gold_present | selection_rate | accuracy_when_gold_present | value_acc | entailment |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B Q5_K_M | 1.0 | 1.0 | 0.9750 | 0.9750 | 0.9750 |
| Llama-3.1-8B Q4_K_M | 1.0 | 1.0 | 0.8083 | 0.8083 | 0.8083 |

Even with identical retrieval/selection, model quality still matters for answer extraction; Qwen outperforms Llama here.

![KV model check (latest_step, s5q24)](docs/figures/model_compare_kv_latest_step_s5q24.png)

Linear selector order generalization (kv_commentary, k=4, s5q24):

| order | selection_rate | accuracy_when_gold_present | value_acc | entailment |
| --- | --- | --- | --- | --- |
| gold_first | 0.9833 | 0.5667 | 0.5667 | 0.7250 |
| gold_middle | 0.9833 | 0.5667 | 0.5667 | 0.7250 |
| gold_last | 0.9750 | 0.5583 | 0.5583 | 0.7167 |
| shuffle | 0.9750 | 0.5583 | 0.5583 | 0.7167 |

Order effects are mostly eliminated, but NOTE authoritativeness still breaks end-to-end accuracy.

Pick-then-answer A/B (kv_commentary, s3q16, authority filter on):

| mode | selection_rate | accuracy_when_gold_present | value_acc | entailment |
| --- | --- | --- | --- | --- |
| pick off | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pick on | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Pick-then-answer adds no benefit once authority filtering removes NOTE noise.

![Order bias (kv_commentary, linear, k=4, s5q24)](docs/figures/order_bias_kv_commentary_linear_s5q24.png)

In this mode, NOTE lines appear after real updates, so `latest_step` can fail while `prefer_set_latest` holds.

Order-bias check with NOTE-aware rerank (kv_commentary, k=4, s3q16):

| order | selection_rate | accuracy_when_gold_present | value_acc | entailment |
| --- | --- | --- | --- | --- |
| gold_first | 0.7292 | 1.0 | 1.0 | 1.0 |
| gold_middle | 0.7292 | 1.0 | 1.0 | 1.0 |
| gold_last | 0.7292 | 1.0 | 1.0 | 1.0 |
| shuffle | 0.7292 | 1.0 | 1.0 | 1.0 |

With `prefer_update_latest`, order bias disappears in kv_commentary while end-to-end accuracy stays perfect.

![Order bias (kv_commentary, prefer_update_latest, k=4, s3q16)](docs/figures/order_bias_kv_commentary_prefer_update_s3q16.png)

- rerank none: accuracy_when_gold_present 0.3333, selection_rate 0.3333
- rerank latest_step: accuracy_when_gold_present 0.625, selection_rate 0.625

A simple deterministic selector can outperform the LLM under ambiguity.

Reranker k-curve (same_key, shuffle, s3q16), accuracy_when_gold_present:

- k=1: rerank none 0.50, rerank latest_step 0.4375
- k=2: rerank none 0.3958, rerank latest_step 0.625
- k=4: rerank none 0.25, rerank latest_step 0.4375
- k=8: rerank none 0.2083, rerank latest_step 0.6875

Deterministic reranking dominates as k grows, highlighting selection as the bottleneck.

Reranker k-curve (same_key, shuffle, s5q24), accuracy_when_gold_present:

Selector quick preset (same_key, shuffle, s2q12):

- no rerank: k=2 0.5417, k=4 0.3333, k=8 0.2917
- rerank latest_step: k=2 1.0, k=4 1.0, k=8 1.0

Quick preset takeaway: even a small reranker makes selection near-perfect in fast runs.

## One command reproduces the headline

Run one script, get one CSV, and compare the table below:

```powershell
.\scripts\run_reference.ps1 -Preset standard -ModelPath "C:\AI\models\your-model.gguf"
```

This writes `runs\summary_all.csv`. The "Report preset (s5q24)" table below is a direct copy of those rows.

Headline takeaways:

- Selection degrades as k grows when using LLM-only selection.
- Deterministic reranking (latest_step/prefer_set_latest) restores near-perfect selection when gold is present.
- Naive positional heuristics (last_occurrence) fail under shuffle, showing order bias is real.

## Reference proof (selector vs LLM)

latest_step reranking uses only fields present in the candidate ledger lines (no hidden gold metadata).

| preset | k | rerank | gold_present | selection_rate | value_acc |
| --- | --- | --- | --- | --- | --- |
| quick | 2 | none | 1 | 0.1667 | 0.1667 |
| quick | 4 | none | 1 | 0.2917 | 0.2917 |
| quick | 8 | none | 1 | 0.25 | 0.25 |
| quick | 2 | latest_step | 1 | 1 | 1 |
| quick | 4 | latest_step | 1 | 1 | 1 |
| quick | 8 | latest_step | 1 | 1 | 1 |


Report preset (s5q24) run (same table, larger sample):

| preset | k | rerank | gold_present | selection_rate | value_acc |
| --- | --- | --- | --- | --- | --- |
| standard | 2 | none | 1 | 0.55 | 0.55 |
| standard | 4 | none | 1 | 0.3333 | 0.3333 |
| standard | 8 | none | 1 | 0.2583 | 0.2583 |
| standard | 2 | latest_step | 1 | 1 | 0.9667 |
| standard | 4 | latest_step | 1 | 1 | 0.9667 |
| standard | 8 | latest_step | 1 | 1 | 0.9667 |
| standard | 2 | last_occurrence | 1 | 0.2917 | 0.2917 |
| standard | 4 | last_occurrence | 1 | 0.125 | 0.1083 |
| standard | 8 | last_occurrence | 1 | 0.1083 | 0.1083 |
| standard | 2 | prefer_set_latest | 1 | 1 | 0.9667 |
| standard | 4 | prefer_set_latest | 1 | 1 | 0.9667 |
| standard | 8 | prefer_set_latest | 1 | 1 | 0.9667 |

Interpretation: selection under ambiguity is the bottleneck. The LLM-only selector degrades as k grows, while simple deterministic reranking (latest_step or prefer_set_latest) restores near-perfect selection when gold is present. last_occurrence underperforms because shuffled candidates break recency-by-position.

```powershell
.\scripts\run_selector_bench.ps1 -Preset standard -ModelPath "C:\AI\models\your-model.gguf"
.\scripts\run_selector_bench.ps1 -Preset standard -ModelPath "C:\AI\models\your-model.gguf" -UseRerank
```


- k=2: rerank none 0.3167, rerank latest_step 0.6667
- k=4: rerank none 0.1667, rerank latest_step 0.6250
- k=8: rerank none 0.1583, rerank latest_step 0.4167

Decomposition line per run is `gold_present_rate -> selection_rate -> accuracy_when_gold_present -> overall accuracy`.

Formulas people can use:

- `gold_present_rate` = fraction of rows where the gold line is in context.
- `selection_rate` = fraction of gold-present rows that cite the gold line.
- `accuracy_when_gold_present` = accuracy conditioned on gold-present rows.
- `overall_accuracy` ~= `gold_present_rate * accuracy_when_gold_present` (when gold evidence is required; otherwise this is an approximation).
- `missing_gold_gap` ~= `accuracy_when_gold_present * (1 - gold_present_rate)` (loss from missing gold).
- `selection_loss` ~= `(1 - selection_rate) * accuracy_when_gold_present` (when gold is required).

Blunt system rule: if `gold_present_rate` is high but `selection_rate` is low, fix selection/ranking. If `gold_present_rate` is low, fix retrieval/recall. If both are high but accuracy is low, fix answer formatting or value extraction.

How to read the scores (plain English):

- If `gold_present_rate` is low, your retriever is failing (evidence missing).
- If `gold_present_rate` is high but `selection_rate` is low, you have an ordering/selection problem.
- If `selection_rate` is high but `accuracy_when_gold_present` is low, the answerer fails even with correct evidence.
- If `value_acc` is high but `exact_acc` is low, answers are right but citations/formatting are wrong.

Reproduce the k=4 order-bias run:

```powershell
$env:GOLDEVIDENCEBENCH_RETRIEVAL_K = "4"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_ORDER_SEED = "0"
foreach ($order in @("gold_first","gold_middle","gold_last","shuffle")) {
  $env:GOLDEVIDENCEBENCH_RETRIEVAL_ORDER = $order
  $outDir = "runs\ambig_${order}_k4_s3q16"
  goldevidencebench sweep --out $outDir --seeds 3 --episodes 1 --steps 240 --queries 16 `
    --state-modes kv --distractor-profiles standard `
    --adapter goldevidencebench.adapters.retrieval_llama_cpp_adapter:create_adapter --no-derived-queries `
    --no-twins --require-citations --results-json "$outDir\combined.json" `
    --max-book-tokens 400 --distractor-rate 0.7 --clear-rate 0.01 --tail-distractor-steps 80
  python .\scripts\summarize_results.py --in "$outDir\combined.json" --out-json "$outDir\summary.json"
}
```

Query-sandwich variant (toggle only; same run otherwise):

```powershell
$env:GOLDEVIDENCEBENCH_RETRIEVAL_QUERY_SANDWICH = "1"
goldevidencebench sweep --out runs --seeds 1 --episodes 1 --steps 120 --queries 8 `
  --state-modes kv --distractor-profiles standard `
  --adapter goldevidencebench.adapters.retrieval_llama_cpp_adapter:create_adapter --no-derived-queries `
  --no-twins --require-citations --results-json .\runs\combined.json --max-book-tokens 400
```

Adapter contract (hard-validated):

- `value` required.
- `support_ids` must be a list (max 3 by default) and must reference UPDATE IDs in the episode (closed-book uses the book ledger).
- Extra fields are rejected; missing `value` fails fast.
- `adapter_schema_version=1.0` is attached to metrics outputs for compatibility.

Run your adapter:

```powershell
goldevidencebench model --data .\data\goldevidencebench.jsonl --adapter goldevidencebench.adapters.ledger_adapter:create_adapter --protocol closed_book
```

Adapter tuning:

- `goldevidencebench model --max-book-tokens 1200` (passed to adapters that expose `max_book_tokens`).
- Llama adapter logs prompt token counts to stderr for debugging.
- For the llama-cpp adapter, set `GOLDEVIDENCEBENCH_REQUIRE_CITATIONS=0` to force `support_ids` to be an empty list (value-only mode).
- Llama-cpp adapter uses grammar-constrained decoding when available: citations-on runs force exactly one support ID, citations-off runs force an empty list.
- When citations are required, the adapter deterministically selects the latest ledger ID for the tag that matches the predicted value, falling back to the latest ledger entry for the tag.

Both `goldevidencebench run` and `goldevidencebench model` can emit machine-readable metrics via `--results-json` (JSON object or array; overwrites the file each run; intended for plotting accuracy vs tokens/passes). Closed-book with `--protocol both` writes an array.

Practical runs: create a folder per run, sweep multiple seeds/state_modes/distractor_profiles, and keep one `results.json` per run to compare stability (long-context behavior is seed/ordering-sensitive).

Sweeps: `goldevidencebench sweep --out runs --seeds 5 --state-modes kv,counter,set,relational --distractor-profiles standard,adversarial,instruction` (writes one subfolder per combo with data/preds/results).
Use `--no-require-citations` in sweeps to disable citations in generated data and grading.
