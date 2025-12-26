# GoldEvidenceBench

GoldEvidenceBench (CLI: `goldevidencebench`) is a benchmark + harness for long-context state tracking. It generates synthetic "episode logs" with evolving state (kv/counter/set/relational), distractors (including instruction injection), and queries that require answering from the latest state update. It evaluates open-book vs closed-book protocols, enforces citation support IDs with capped-k + F1, checks entailment-from-citations, and uses counterfactual twin episodes to detect shortcut heuristics. It also reports efficiency (tokens/query, passes, wall time) so you can measure capability per compute.

GoldEvidenceBench is a small benchmark + reference codebase for testing whether an LLM can track evolving state across long documents by reading its own "book" artifacts:

- **Chapters**: narrative text (contains distractors and stale summaries)
- **Glossary (tags)**: a lightweight key reference
- **State ledger**: the authoritative state updates (with support IDs)

## TL;DR (layman summary)

GoldEvidenceBench shows whether your AI system can reliably pick the right piece of evidence when several similar candidates exist. It builds long, noisy logs with changing facts, then checks if the model chooses the most recent, correct update and cites it. The key benefit is that it separates "the evidence was available" from "the model chose the right evidence," so you can improve the exact part of your system that is failing (retrieval vs selection vs formatting).

## Headline results (summary)

Selection under ambiguity is the bottleneck. Simple deterministic selection outperforms the LLM as candidate lists grow.

| Finding | Evidence |
| --- | --- |
| Ordering bias is severe | gold_last ? gold_middle/shuffle ? gold_first |
| Query sandwich did not help | selection_rate did not improve; shuffle got worse |
| Pick-then-answer did not help | selection_rate stayed flat or dropped |
| Deterministic reranker helps | rerank latest_step roughly doubles selection at k=2/4/8 |

## Reproduce the headline results (minimal commands)

1) Order bias (k=4, s3q16):

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

2) Reranker k-curve (same_key, shuffle, s5q24):

```powershell
$env:GOLDEVIDENCEBENCH_RETRIEVAL_WRONG_TYPE = "same_key"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_ORDER = "shuffle"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_ORDER_SEED = "0"
$ks = @("2","4","8")
foreach ($rerank in @("none","latest_step")) {
  $env:GOLDEVIDENCEBENCH_RETRIEVAL_RERANK = $rerank
  foreach ($k in $ks) {
    $env:GOLDEVIDENCEBENCH_RETRIEVAL_K = $k
    $outDir = "runs\ab_rerank_${rerank}_k${k}_same_shuffle_s5q24"
    goldevidencebench sweep --out $outDir --seeds 5 --episodes 1 --steps 200 --queries 24 `
      --state-modes kv --distractor-profiles standard `
      --adapter goldevidencebench.adapters.retrieval_llama_cpp_adapter:create_adapter --no-derived-queries `
      --no-twins --require-citations --results-json "$outDir\combined.json" `
      --max-book-tokens 400 --distractor-rate 0.7 --clear-rate 0.01 --tail-distractor-steps 80
    python .\scripts\summarize_results.py --in "$outDir\combined.json" --out-json "$outDir\summary.json"
  }
}
```

The benchmark generates synthetic episodes (updates + distractors + queries), including derived-invariant queries that require computation over the current state, grades model answers (optionally requiring citations/support IDs), and includes baselines (naive scan vs ledger-based reader).

## Defaults (chosen here)

These are the CLI defaults (picked to create long-ish documents with frequent distractors):

- `episodes=20`, `steps=220`, `keys=14`, `queries=12`, `derived_query_rate=0.35`, `chapters=8`, `twins=true`
- `distractor_rate=0.50`, `clear_rate=0.08`
- `distractor_profile=instruction` (adds spec-violating instructions); `instruction_suite` adds quoted/format variants; `adversarial` adds stale-echo distractors
- `state_mode=kv` (switch to `counter`, `set`, or `relational`)
- `require_citations=true` (questions ask for JSON `{value, support_ids}` with max 3)
- Closed-book is the headline score (`goldevidencebench run` defaults to `--protocol closed_book`; open-book is diagnostic)

## Why this benchmark

This benchmark isolates a specific long-context failure mode: **state changes over time** (updates + clears), embedded in a long document with misleading restatements. It's motivated by reports that transformer LLMs can struggle with consistent state tracking across long sequences (see: MIT News, 2025-12-17).

## Install

Python 3.12 is assumed.

```powershell
python -m pip install -e .
```

## Generate a dataset

Writes JSONL with one row per query.

```powershell
goldevidencebench generate --out .\data\goldevidencebench.jsonl --seed 0
```

Each row looks like:

- `id`: query ID
- `document`: the raw **episode log** (updates + distractors)
- `book`: a derived "book" artifact (chapters + glossary + ledger) for convenience/baselines
- `question`: the question text (and output format requirements if citations are enabled)
- `gold`: `{value, support_ids}` where `support_ids` contains the authoritative UPDATE ID that establishes the current value
- `meta`: includes `requires_citation`, the queried `key`, and derived-query fields
- `schema_version`: `"0.1"`
- `state_mode`: `kv|counter|set|relational`

Derived queries add `meta.query_type=derived` with a `derived_op` and optional `derived_manager` (relational reports).

State dynamics:

- `kv` (default): standard key->value overwrites
- `counter`: numeric accumulators (increments)
- `set`: membership add/remove (values are comma-separated lists)
- `relational`: reassignment tasks (e.g., who reports to whom)

## Run baselines

```powershell
goldevidencebench run --data .\data\goldevidencebench.jsonl --baseline naive
goldevidencebench run --data .\data\goldevidencebench.jsonl --baseline ledger
```

Protocols (headline = closed_book):

- `open_book`: baselines read the raw `document` (episode log)
- `closed_book`: baselines read only the derived `book` artifact

`goldevidencebench run` defaults to `--protocol closed_book` (pass `--protocol both` for diagnostics).

Metrics:

- `value_acc`: predicted `value` matches gold
- `cite_f1`: support-ID F1 (only when citations are required; capped at `max_support_k`)
- `support_bloat`: fraction of citation-required answers that use more support IDs than needed (penalized in exact accuracy)
- `entailment`: fraction where the answer is justified by the cited updates only
- `exact_acc`: value match + (if required) support includes gold + entailment-from-citations
- `twin_consistency`: counterfactual twin agreement/disagreement rate (anti-shortcut)
- `twin_flip_rate`: twin pairs where the answer flips when the decisive UPDATE flips (higher is better)
- `instr_acc` / `instr_gap`: accuracy on questions with instruction-injection distractors, and the drop vs. clean questions
- `instr_override_rate`: fraction of instruction-tagged questions that follow injected instructions (lower is better)
- `state_integrity_rate`: fraction of instruction-tagged questions that still answer from the latest true state (higher is better)
- Efficiency curve (printed by `goldevidencebench run`): tokens read, tokens/query, passes over doc, wall-clock seconds (`wall_s`) and per-query (`wall_s_per_q`). Llama-cpp runs also record `prefill_s`/`decode_s` and per-query variants when the low-level perf API is available.

## Quickstart evaluation

Use the PowerShell runner to avoid manual sweeps. It writes results to `runs\combined.json`.

Smoke check (fast, noisy signal):

```powershell
.\scripts\run_bench.ps1 -Preset smoke -ModelPath "C:\AI\models\your-model.gguf"
```

Standard check (still small, more stable):

```powershell
.\scripts\run_bench.ps1 -Preset standard -ModelPath "C:\AI\models\your-model.gguf"
```

By default the runner disables citations (value-only). To require citations, add `-RequireCitations`.
When citations are disabled, `exact_acc` tracks `value_acc`.

Summarize results into CSV/JSON (for papers/plots):

```powershell
python .\scripts\summarize_results.py --in .\runs\combined.json --out-csv .\runs\summary.csv --out-json .\runs\summary.json
```

Collect all run summaries into one CSV (optionally newest per pattern):

```powershell
python .\scripts\collect_runs.py --runs-dir .\runs --out-csv .\runs\summary_all.csv --latest-only
```

```powershell
python .\scripts\summarize_results.py --in .\runs\combined.json --out-csv .\runs\summary.csv --out-json .\runs\summary.json
```

The summary JSON includes overall means plus group means for `value_acc`, `exact_acc`, `cite_f1`, and `entailment`.
If `metrics_raw` are present, the summary includes `overall_raw` and `by_group_raw` for the same metrics.
Use `--out-decomp-csv` to emit a per-run decomposition table (gold_present_rate, selection_rate,
accuracy_when_gold_present, overall accuracy, plus retrieval settings).
Recency bucket summaries (tokens since last update, distractors since update, writes to key) are included when
`preds.jsonl` exists next to each `data.jsonl`. Defaults are `200,400,800,1600` for tokens, `2,4,8,16` for
distractors, and `1,2,4,8` for writes. You can override them:

```powershell
python .\scripts\summarize_results.py --in .\runs\combined.json --out-json .\runs\summary.json `
  --recency-buckets 200,400,800,1600 --distractor-buckets 2,4,8,16 --writes-buckets 1,2,4,8
```

To force longer recency gaps, add `--tail-distractor-steps N` when generating or sweeping. This makes the
final N steps distractor-only (no updates), creating a longer tail after the last update.

## Efficient testing workflow (fast -> slow)

Why long sweeps take hours: total queries roughly equal
`seeds × state_modes × distractor_profiles × episodes × queries` (double if twins are on).
At ~40s/query, 144 queries is ~1h36m, 288 queries is ~3h12m.

Biggest speed lever: keep `--max-book-tokens` small during iteration (400-1200). Larger values
inflate prefill time unless you also raise model `n_ctx`.

Use these presets to iterate quickly:

Smoke (2-5 min): sanity check instruction handling.

```powershell
goldevidencebench sweep --out runs --seeds 1 --episodes 1 --steps 30 --queries 4 `
  --state-modes kv --distractor-profiles instruction `
  --adapter goldevidencebench.adapters.llama_cpp_adapter:create_adapter --no-derived-queries `
  --no-twins --no-require-citations --results-json .\runs\combined.json --max-book-tokens 600
```

Triage (10-20 min): compare kv vs set, standard vs instruction.

```powershell
goldevidencebench sweep --out runs --seeds 1 --episodes 1 --steps 60 --queries 8 `
  --state-modes kv,set --distractor-profiles standard,instruction `
  --adapter goldevidencebench.adapters.llama_cpp_adapter:create_adapter --no-derived-queries `
  --no-twins --no-require-citations --results-json .\runs\combined.json --max-book-tokens 1200
```

Real (hours): full run with citations + twins on for reporting.

```powershell
goldevidencebench sweep --out runs --seeds 3 --episodes 1 --steps 100 --queries 12 `
  --state-modes kv,set --distractor-profiles standard,instruction `
  --adapter goldevidencebench.adapters.llama_cpp_adapter:create_adapter --no-derived-queries `
  --results-json .\runs\combined.json --max-book-tokens 6000
```

PaTH-style curve (accuracy vs steps):

```powershell
goldevidencebench sweep --out runs --seeds 1 --episodes 1 --queries 8 `
  --steps-list 20,40,80,160,320 --state-modes kv --distractor-profiles standard `
  --adapter goldevidencebench.adapters.llama_cpp_adapter:create_adapter --no-derived-queries `
  --no-twins --no-require-citations --results-json .\runs\combined.json --max-book-tokens 600
```

Memory-budget curve (accuracy vs max_book_tokens):

```powershell
goldevidencebench sweep --out runs --seeds 1 --episodes 1 --steps 60 --queries 8 `
  --state-modes kv --distractor-profiles standard `
  --adapter goldevidencebench.adapters.llama_cpp_adapter:create_adapter --no-derived-queries `
  --no-twins --no-require-citations --results-json .\runs\combined.json `
  --max-book-tokens-list 200,400,800,1200
```

## Grade model outputs

Predictions JSONL can be either:

- `{ "id": "...", "value": "...", "support_ids": ["U0007"] }`, or
- `{ "id": "...", "output": "..." }` where `output` contains a JSON object (optionally embedded in text)

```powershell
goldevidencebench grade --data .\data\goldevidencebench.jsonl --pred .\preds.jsonl
```

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
Env var names use the `GOLDEVIDENCEBENCH_` prefix; legacy `TAGBENCH_` variables are still accepted.
Set `GOLDEVIDENCEBENCH_RETRIEVAL_K` to include top-k latest entries for the key (default 1). Set
`GOLDEVIDENCEBENCH_RETRIEVAL_WRONG_TYPE` to `none`, `same_key`, or `other_key` to inject a wrong line for robustness
testing. Use `GOLDEVIDENCEBENCH_RETRIEVAL_INCLUDE_CLEAR=0` to skip CLEAR entries.
Use `GOLDEVIDENCEBENCH_RETRIEVAL_DROP_PROB` (0-1) to probabilistically drop the correct line, and
`GOLDEVIDENCEBENCH_RETRIEVAL_DROP_SEED` to make the drop deterministic by row id.
Use `GOLDEVIDENCEBENCH_RETRIEVAL_ORDER=shuffle|gold_first|gold_middle|gold_last` (and optional
`GOLDEVIDENCEBENCH_RETRIEVAL_ORDER_SEED`) to control ordering and test positional bias under ambiguity.
Set `GOLDEVIDENCEBENCH_RETRIEVAL_QUERY_SANDWICH=1` to repeat the question before and after the
candidate ledger lines (query sandwich mitigation).
Set `GOLDEVIDENCEBENCH_RETRIEVAL_PICK_THEN_ANSWER=1` to force a two-step flow: pick a support_id
first, then answer using only that line.
Set `GOLDEVIDENCEBENCH_RETRIEVAL_RERANK=latest_step` to deterministically choose the newest candidate
before answering (non-LLM selector baseline).

Retrieval order bias (example, k=4, s3q16, kv/standard, gold always present):

- `gold_first`: value/exact/cite_f1 = 0.0
- `gold_middle`: value/exact/cite_f1 = 0.4167
- `shuffle`: value/exact/cite_f1 = 0.4583
- `gold_last`: value/exact/cite_f1 = 1.0

Entailment stays 1.0 and `gold_in_context_rate` is 1.0, so the failures are positional
selection errors (not retrieval or hallucination).

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

Reranker baseline (same_key, k=4, shuffle, s3q16):

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

- k=2: rerank none 0.3167, rerank latest_step 0.6667
- k=4: rerank none 0.1667, rerank latest_step 0.6250
- k=8: rerank none 0.1583, rerank latest_step 0.4167

Decomposition line per run is `gold_present_rate -> selection_rate -> accuracy_when_gold_present -> overall accuracy`.

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

## Anti-cheat / robustness notes

- The episode log contains **UPDATE** lines (authoritative) and **DISTRACTOR** lines (untrusted).
- When citations are enabled, correct answers require returning **support IDs** (update IDs like `U0007`) and passing **entailment-from-citations**.
- By default, every episode also includes a **counterfactual twin** (one UPDATE is flipped); grading reports `twin_consistency` to detect shortcut heuristics.
- UPDATE IDs are non-monotonic (hash-like) to prevent "pick the max ID" shortcuts; ordering comes from the logged `step`.
- Closed-book protocol feeds only the derived book artifact (no episode log).
- Book artifacts are structurally validated (allowed sections + ledger grammar) to prevent hidden log leaks.
- `--distractor-profile instruction` (default) injects spec-violating instructions plus format traps (invalid JSON/extra fields); `adversarial` adds stale-echo distractors (late repeats of old values).
- Instruction profiles also insert "helpful summary" restatements of stale values.
- Instruction-injection resistance is measured via `instr_acc`/`instr_gap`; answers derived from instruction lines are not authoritative.

### Efficiency snapshot (ledger baseline, 1 episode, steps=150, queries=12, distractor_profile=instruction)

Closed-book (headline metric):

| state_mode | exact_acc | tokens/query |
| --- | --- | --- |
| kv | 1.00 | 3,574 |
| counter | 1.00 | 3,537 |
| set | 1.00 | 3,601 |
| relational | 1.00 | 3,473 |

Open-book (diagnostic):

| state_mode | exact_acc | tokens/query |
| --- | --- | --- |
| kv | 1.00 | 1,922 |
| counter | 1.00 | 1,910 |
| set | 1.00 | 1,955 |
| relational | 1.00 | 1,879 |

## Dev

```powershell
python -m pip install -e .[dev]
python -m pytest
python -m ruff check .
```
What's new: GoldEvidenceBench now evaluates closed-book state tracking by default (answer using only a derived "book" artifact, not the raw episode log), adds richer state dynamics (kv|counter|set|relational) and derived-invariant queries, and includes adversarial distractors such as instruction injection, format traps, and stale-echo repeats of outdated values. To reduce loopholes, UPDATE IDs are non-monotonic (hash-like) and each episode includes a counterfactual twin; grading reports both twin_consistency and twin_flip_rate alongside support_bloat. `goldevidencebench run` also prints an efficiency curve (tokens read, tokens/query, passes, wall-clock) so accuracy can be compared against compute cost.
