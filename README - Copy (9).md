# GoldEvidenceBench

GoldEvidenceBench (CLI: `goldevidencebench`) is a benchmark + harness for long-context state tracking. It generates synthetic "episode logs" with evolving state (kv/kv_commentary/counter/set/relational), distractors (including instruction injection), and queries that require answering from the latest state update. It evaluates open-book vs closed-book protocols, enforces citation support IDs with capped-k + F1, checks entailment-from-citations, and uses counterfactual twin episodes to detect shortcut heuristics. It also reports efficiency (tokens/query, passes, wall time) so you can measure capability per compute.

Book artifacts include:

- **Chapters**: narrative text (contains distractors and stale summaries)
- **Glossary (tags)**: a lightweight key reference
- **State ledger**: the authoritative state updates (with support IDs)

## TL;DR (layman summary)

GoldEvidenceBench shows whether your AI system can reliably pick the right piece of evidence when several similar candidates exist. It builds long, noisy logs with changing facts, then checks if the model chooses the most recent, correct update and cites it. The key benefit is that it separates "the evidence was available" from "the model chose the right evidence," so you can improve the exact part of your system that is failing (retrieval vs selection vs formatting).

This helps fix a common real-world failure: the right evidence is retrieved, but the model still picks the wrong snippet. GoldEvidenceBench isolates that selection bottleneck and shows when a simple selector/reranker fixes it.

Example (plain English):

Log:
- Update 1: "Shipping address = 12 Oak St"
- Update 2: "Shipping address = 99 Pine Ave"
- Note: "Customer mentioned they used to live on Oak St"

Question: "Where should we ship the order?"

Correct evidence is Update 2 (99 Pine Ave). The NOTE is contextual but not authoritative. GoldEvidenceBench measures whether the system chooses the correct update and cites it, even when nearby notes mention older facts.

## Quick links
- [Related work](#related-work)
- [Adapters](docs/ADAPTERS.md)
- [Primary flow](#primary-flow-the-done-path)
- [Goal](#goal-a-self-teaching-gym-for-evidence-selection)
- [Mixture of Oracles](#mixture-of-oracles-why-these-metrics-matter)
- [Selector training loop](#selector-training-loop-recommended-workflow)
- [Reference proof](#reference-proof-selector-vs-llm)
- [Deep dive / repro details](#deep-dive--repro-details)
- [Install](#install)

## Primary flow (the done path)

1) Run one command to reproduce the headline:

```powershell
.\scripts\run_reference.ps1 -Preset standard -ModelPath "C:\AI\models\your-model.gguf"
```

Headline metric: closed-book exact_acc with citations on (use value_acc when citations are off for quick iteration).

Preset standard runs with --require-citations.

2) Read the result table in `runs/summary_all.csv` (it matches the "Reference proof" section below).

3) Takeaways: selection under ambiguity fails for LLM-only; a deterministic selector fixes it; learned selectors reduce but do not remove order bias.

Everything else in this README is an extension or deeper dive.

If selection is the bottleneck, run `scripts/run_selector_only.ps1`; if gold_present is low, run the BM25/TF-IDF baselines to confirm retrieval issues.

## Goal: a self-teaching gym for evidence selection

GoldEvidenceBench is built to be a self-teaching gym for a specific class of skills: *track state from messy/long context by choosing the right evidence*.

It works as self-teaching when two things are true:

- You can generate lots of situations (episodes) automatically.
- You have an automatic oracle for what is right (gold line / authoritative update / correct value).

GoldEvidenceBench provides that oracle, so you can train without humans for those behaviors.

What kinds of self-teaching this enables:

- Evidence selection under ambiguity: train a model/module to pick the correct line among plausible candidates.
- Authority gating (NOTE vs UPDATE): train what is allowed to change state.
- Retrieval learning (gold-present): train the retriever/reranker to surface the right line.
- Abstain/ask-for-more when gold is missing: train a don't-guess policy when evidence isn't there.

Self-teaching loop:

1) Generate episodes (increasing difficulty: more distractors, paraphrases, key aliasing, contradictions, NOTES).
2) Run your system (retriever -> selector -> answer).
3) Grade automatically (gold present? selected gold? value correct? citation correct?).
4) Turn mistakes into training data:
   - Selection: prefer gold over chosen distractor (pairwise or classification).
   - Retrieval: query-gold positives + hard negatives (contrastive).
   - Abstain: label insufficient evidence when gold is missing.
5) Retrain, repeat, and push difficulty at the failure boundary.

Where it won't self-teach well: anywhere you don't have a reliable oracle (open-ended writing, fuzzy truth, best idea).

## v2 release notes
- Defaults to closed-book evaluation and supports richer state modes + derived queries.
- Added authority-aware selection (`GOLDEVIDENCEBENCH_RETRIEVAL_AUTHORITY_FILTER=1`) to fix kv_commentary NOTE noise.
- Verified perfect end-to-end exact_acc (citations on) for kv_commentary using the reference system (Qwen 2.5 7B Q5_K_M, retrieval_llama_cpp_adapter, authority filter on; s3q16 + s5q24 grids, k=2/4/8). (see runs/*authfilter*; command below).
- Added compute vs quality figure and multi-model kv comparison.

## V2 takeaway (authority-aware selection)

In kv_commentary, the dominant failure is *authoritativeness*, not selection or attribution. Adding `GOLDEVIDENCEBENCH_RETRIEVAL_AUTHORITY_FILTER=1` (drop NOTE lines before selection) restores perfect end-to-end accuracy in the kv_commentary grids (s3q16 + s5q24, k=2/4/8) for the reference system (Qwen 2.5 7B Q5_K_M, retrieval_llama_cpp_adapter). This is now the recommended default for kv_commentary.
Runs: `runs/kv_commentary_grid_linear_authfilter_k{2,4,8}_s3q16` and `runs/kv_commentary_grid_linear_authfilter_k{2,4,8}_s5q24`.
Clean authority-filter A/B (kv_commentary, s3q16, gold present = 1.0):

| rerank | k | value_acc | exact_acc | selection_rate |
| --- | --- | --- | --- | --- |
| prefer_set_latest | 2 | 0.9375 | 0.9375 | 0.9375 |
| prefer_set_latest | 4 | 0.9375 | 0.9375 | 0.9375 |
| prefer_set_latest | 8 | 0.9375 | 0.9375 | 0.9375 |
| linear | 2 | 1.0000 | 1.0000 | 1.0000 |
| linear | 4 | 1.0000 | 1.0000 | 1.0000 |
| linear | 8 | 1.0000 | 1.0000 | 1.0000 |

Clean authority-filter A/B (kv_commentary, s5q24, gold present = 1.0):

| rerank | k | value_acc | exact_acc | selection_rate |
| --- | --- | --- | --- | --- |
| prefer_set_latest | 2 | 0.8917 | 0.8917 | 0.8917 |
| prefer_set_latest | 4 | 0.8917 | 0.8917 | 0.8917 |
| prefer_set_latest | 8 | 0.8917 | 0.8917 | 0.8917 |
| linear | 2 | 1.0000 | 1.0000 | 1.0000 |
| linear | 4 | 1.0000 | 1.0000 | 1.0000 |
| linear | 8 | 1.0000 | 1.0000 | 1.0000 |


Command (s3q16 example):

```powershell
$env:GOLDEVIDENCEBENCH_RETRIEVAL_AUTHORITY_FILTER = "1"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_RERANK = "linear"
foreach ($k in 2,4,8) {
  $env:GOLDEVIDENCEBENCH_RETRIEVAL_K = "$k"
  $outDir = "runs\kv_commentary_grid_linear_authfilter_k${k}_s3q16"
  goldevidencebench sweep --out $outDir --seeds 3 --episodes 1 --steps 240 --queries 16 `
    --state-modes kv_commentary --distractor-profiles standard `
    --adapter goldevidencebench.adapters.retrieval_llama_cpp_adapter:create_adapter --no-derived-queries `
    --no-twins --require-citations --results-json "$outDir\combined.json" `
    --max-book-tokens 400 --note-rate 0.30
  python .\scripts\summarize_results.py --in "$outDir\combined.json" --out-json "$outDir\summary.json"
}
```
Canonical v2 default: authority filter (hard gate) + `prefer_update_latest` (soft tie-break).

## V3 plan (NOTE robustness + authority spoofing)

**V3-A: Learned NOTE robustness (trusted authority field)**

Train with NOTE candidates present, but label the gold UPDATE as correct using `export_selector_dataset.py --use-gold-support` so the selector learns authority, not just recency.

- Train selector with NOTES present.
- Evaluate with the authority filter OFF.
- Authority is a structured feature (not just text pattern).
- Goal: match linear baseline behavior without hard gating.

**V3-B: Authority spoofing (untrusted authority signal)**

- Add a stress profile where some NOTE lines look like UPDATEs, or UPDATE lines contain ?NOTE:? text, or the authority marker is missing/ambiguous.
- Measure how often the selector is tricked and whether abstain triggers when authority is unclear.

**V3-C: Answer contract (extraction clamp)**

- Set `GOLDEVIDENCEBENCH_RETRIEVAL_DETERMINISTIC_ANSWER=1` to return the value parsed from the selected ledger line (no LLM).
- Use this as an oracle upper bound: if accuracy_when_gold_present rises to selection_rate, the remaining gap was answerer/extraction.

**Architecture guardrail**

- Split authority gating (cheap classifier/rules/signature check) from content selection (reranker).
- Keep authority checks as hard gates so content plausibility cannot override it.

Proof run uses `linear` to match the reference tables; the default for general use is `prefer_update_latest`.


## V3-A NOTE-aware selector (filter OFF) results

Run context: kv_commentary, k=4 same_key, authority filter OFF, gold present = 1.0. (runs/kv_commentary_noteaware_train2_*_k4_s3q16)

| order | gold_support_selected_rate | selected_note_rate | wrong_update_rate | accuracy_when_gold_present | value_acc |
| --- | --- | --- | --- | --- | --- |
| gold_first | 0.938 | 0.000 | 0.062 | 0.938 | 0.938 |
| gold_middle | 0.938 | 0.000 | 0.062 | 0.938 | 0.938 |
| gold_last | 0.938 | 0.000 | 0.062 | 0.938 | 0.938 |
| shuffle | 0.938 | 0.000 | 0.062 | 0.938 | 0.938 |

Interpretation: NOTE attraction is gone (selected_note_rate = 0), and order bias is gone (same values across orders). The remaining error is rare wrong-UPDATE selection (~6.25%). For kv_commentary, prefer gold_support_selected_rate over selection_rate because selection_rate can count NOTE as "gold" when the most recent line is a NOTE.

V3-A NOTE rate sweep (s5q24, k=4):

| note_rate | authority_filter | gold_support_selected_rate | selected_note_rate | wrong_update_rate | value_acc |
| --- | --- | --- | --- | --- | --- |
| 0.2 | off | 0.8500 | 0.0000 | 0.1500 | 0.8500 |
| 0.4 | off | 0.8583 | 0.0333 | 0.1083 | 0.8583 |
| 0.2 | on | 0.8500 | 0.0000 | 0.1500 | 0.8500 |
| 0.4 | on | 0.8917 | 0.0000 | 0.1083 | 0.8917 |

Interpretation: NOTE attraction stays low even without the filter (<= 3.3%), but the main bottleneck is still wrong UPDATE selection (~10-15%). The authority filter helps at higher NOTE rates (0.4) by removing the residual NOTE picks.

New extraction diagnostics (in summary.json / summary_all.csv):

- answer_acc_given_gold_selected (alias: value_acc_when_gold_selected): accuracy when the gold support_id was selected.
- value_is_substring_of_selected_line_rate: how often the predicted value is a substring of the selected ledger line.
- support_consistency_rate: how often the answer cites the same support_id the selector chose.
- gold_support_selected_rate: how often the answer cites the authoritative gold UPDATE (ignores NOTE).
- selected_note_rate: share of predictions that cite a NOTE line.
- selected_wrong_update_rate: share of non-gold selections that are UPDATEs (conditional rate).
- wrong_update_rate: share of all selections that are non-gold UPDATEs (overall rate).

## Monitor research questions (V3 focus)

- When can a learned monitor replace a hard gate? (V3-A: trusted authority field)
- How do monitors fail under spoofing? (V3-B: authority spoofing)

Make the monitor measurable:

- Authority false accept / false reject rates (gate calibration).
- Abstain precision/recall when gold is missing (don't-guess policy).

## Mixture of Oracles (why these metrics matter)

GoldEvidenceBench already behaves like a mixture-of-oracles evaluation, even if you don't call it that.

This buys you something normal evals do not: you can improve one module at a time (retriever vs selector vs authority gate vs abstain) because each metric is a localized training signal.

Oracle vs judge:

- Oracle = deterministic/synthetic ground truth labels produced by the generator (gold line/value/authority).
- Judge = model-based evaluator (optional, noisy) and not required for core metrics.

Each metric is an oracle for a specific contract:

- Retrieval oracle: was gold evidence present? (`gold_present_rate`)
- Selection oracle: did we choose gold when present? (`selection_rate`, `accuracy_when_gold_present`)
- Attribution oracle: does cited evidence entail the claim? (`cite_f1`, entailment)
- Authority oracle: was the chosen line allowed to update state? (NOTE vs UPDATE)
- Robustness oracle: does the decision survive shuffles/confusers? (order-bias, k-curve)
- Abstain oracle: if gold is missing, did the system refuse/escalate? (abstain policy)

Oracles come in two roles:

- Hard gates (non-negotiable): authority + attribution
- Soft scores (tradeoffs): selection confidence + robustness + cost

Counterfactuals (shuffle, confusers) make the oracles harder to game.

Oracle stack (one-line version):

Gates: authority + attribution must pass.
Score: rank by selection + robustness subject to cost.

Decision quality here = satisfying gates + selecting the correct state update under confusers.

Recommended scoring rule: treat authority + attribution as preconditions when reporting accuracy (especially when citations are required).

## What counts as an oracle?

An oracle is any source of truth you can check automatically, like an authoritative ledger line, a signed update, or a ground-truth simulator state.
This is the opposite of open-ended tasks (creative writing, fuzzy truth, best idea), where correctness is subjective and labels are noisy.

## Where this is uniquely useful

GoldEvidenceBench fits domains where the truth is in the log but selection is hard:

- Authoritative event logs: orders/shipments, account or profile state, configuration changes.
- Policy vs commentary workflows: support tickets, medical or billing notes vs updates.
- Pipelines where retrieval succeeds but the model picks the wrong snippet.

## Training signals (what the oracle enables)

The harness produces dense, automatable labels so you can train specific behaviors without humans:

- Selection behavior: pairwise preferences (query + gold) vs (query + chosen distractor).
- Authority behavior: teach NOTE vs UPDATE constraints to prevent commentary from mutating state.
- Attribution behavior: train to cite only evidence that entails the claim.
- Abstain behavior: label insufficient evidence when gold is missing and train refusal/escalation.

Key point: if the oracle is stable, you get repeatable gradients; if the oracle is fuzzy, the signal is noise.

What this yields in practice: reliably correct state updates in messy long context, because you can diagnose the failing contract and train it in isolation instead of hoping end-to-end prompting fixes it.

## What we've learned (and what it's for)

GoldEvidenceBench separates failure modes instead of blending them into one score:

- If `gold_present_rate` is low, retrieval is the bottleneck.
- If `gold_present_rate` is high but `selection_rate` is low, ranking is the bottleneck.
- If both are high but `value_acc` is low, the answerer/prompt/schema is the bottleneck.
- Even with perfect evidence and selection, value accuracy can still drop (formatting/extraction failures).

This aligns with the original long-context/state-tracking motivation: the model must use the latest authoritative update under distractors.

What this helps in practice:

- Selector/reranker training: export labeled (query, candidates, gold) examples, train a selector, and measure selection_rate + accuracy_when_gold_present.
- RAG debugging: BM25/TF-IDF show low gold_present_rate even when selection works, so retrieval needs improvement.
- Authority filtering: kv_commentary shows NOTE noise can be fixed by filtering non-authoritative lines.

What to expect from canonical sweeps:

- LLM-only selection shows order bias under ambiguity (gold_first vs gold_last spread).
- Deterministic selectors remove order sensitivity when evidence is isolated.
- Noisy retrieval drops accuracy in proportion to missing gold, while entailment stays high.

Next steps without feature creep:

1) Freeze a v2 canonical suite: order-bias, authority stress, and one retriever sanity run.
2) Treat selector training as the product loop (export -> train -> evaluate).
3) Add one strong retriever baseline (dense/semantic), then stop.

## Research use (reproducible runs)

Use this if you want publishable, comparable results:

- Freeze a preset (seeds/steps/queries/k/order/drop_prob) and name it in the paper.
- Run the canonical command once per model and keep the output under `runs/reference_v1/`.
- Report the decomposition line: `gold_present_rate -> selection_rate -> accuracy_when_gold_present -> overall accuracy`.
- Record the model path, commit hash, and command used with each run.

Suggested v1 reporting convention:

- Benchmark version: v1.0 (frozen presets + metrics)
- Model: <name/quant>
- Command: <exact command>
- Outputs: `runs/summary_all.csv`, plus the figure/table in the README

## Selector training loop (recommended workflow)

Quick one-command loop (generate -> train -> evaluate):

```powershell
.\scripts\run_selector_training.ps1 -ModelPath "C:\AI\models\your-model.gguf" `
  -StateMode kv_commentary -AuthoritativeOnly -UseAuthorityFilter
```

This writes `runs/selector_training_quick/summary.json` and trains a linear selector under the current defaults.

Latest quick eval (kv_commentary, shuffle, k=4, s2q16):

- Overall: value_acc=0.7188, exact_acc=0.7188, cite_f1=0.7188, entailment=1.0000
- Decomposition: 0.8438 -> 0.8519 -> 0.8519 -> 0.7188

If you want to *improve* a system, treat GoldEvidenceBench as a selector training loop:

1) Export (query, candidates, gold) datasets.
2) Train a selector/reranker on those labels.
3) Evaluate using `selection_rate` and `accuracy_when_gold_present`.

This loop is model-agnostic: you can keep the answerer fixed and only improve selection.

## Selector training (optional but powerful)

Train a tiny linear selector from generated data (no extra dependencies):

```powershell
python .\scripts\export_selector_dataset.py --data .\data\goldevidencebench.jsonl --out .\data\selector_train.jsonl --k 4 --wrong-type same_key --order shuffle
python .\scripts\train_selector_linear.py --data .\data\selector_train.jsonl --out .\models\linear_selector.json

$env:GOLDEVIDENCEBENCH_RETRIEVAL_RERANK="linear"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_LINEAR_MODEL=".\models\linear_selector.json"
```

Use this when you want a learned selector instead of a fixed heuristic.
Authority-aware features used by the linear selector: UPDATE vs NOTE, step distance, position, and key/value overlap.
Authority filter baseline: set `GOLDEVIDENCEBENCH_RETRIEVAL_AUTHORITY_FILTER=1` to drop NOTE lines before selection.
Recommended default for kv_commentary: keep `GOLDEVIDENCEBENCH_RETRIEVAL_AUTHORITY_FILTER=1`.
Example training result (default settings): train_selection_rate 1.0000, test_selection_rate 1.0000.

Observed A/B (s3q16, same settings):

| mode | selection_rate | accuracy_when_gold_present | value_acc |
| --- | --- | --- | --- |
| LLM-only (none) | 0.3125 | 0.2292 | 0.2292 |
| linear selector | 0.5000 | 0.4375 | 0.4375 |

This shows a clear, tangible improvement from training the selector.
Re-run with higher seeds/queries for a more stable estimate.

Run a quick sweep with the trained selector:

```powershell
$env:GOLDEVIDENCEBENCH_RETRIEVAL_RERANK="linear"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_LINEAR_MODEL=".\models\linear_selector.json"

$outDir = "runs\linear_selector_quick"
goldevidencebench sweep --out $outDir --seeds 1 --episodes 1 --steps 80 --queries 8 `
  --state-modes kv --distractor-profiles standard `
  --adapter goldevidencebench.adapters.retrieval_llama_cpp_adapter:create_adapter --no-derived-queries `
  --no-twins --require-citations --results-json "$outDir\combined.json" --max-book-tokens 400
python .\scripts\summarize_results.py --in "$outDir\combined.json" --out-json "$outDir\summary.json"
```

## Headline results (summary)

Selection under ambiguity is the bottleneck. Simple deterministic selection outperforms the LLM as candidate lists grow.

See rows `ambig_*` and `ab_rerank_*` in `runs/summary_all.csv` for the exact numbers.

| Finding | Evidence |
| --- | --- |
| Ordering bias is severe | selection_rate (LLM-only, k=4 same_key): gold_last > gold_middle/shuffle > gold_first |
| Query sandwich did not help | selection_rate did not improve; shuffle got worse |
| Pick-then-answer did not help | selection_rate stayed flat or dropped |
| Deterministic reranker helps | rerank latest_step roughly doubles selection at k=2/4/8 |
| Learned selector still order-sensitive | linear selector: gold_first < gold_middle/last (see generalization sweep below) |

Generalization sweep (linear selector, k=4 same_key, gold present = 1.0):

s3q16 (runs/linear_order_*_s3q16):

| order | selection_rate | accuracy_when_gold_present |
| --- | --- | --- |
| gold_first | 0.417 | 0.417 |
| gold_middle | 0.688 | 0.688 |
| gold_last | 0.688 | 0.688 |
| shuffle | 0.688 | 0.688 |

s5q24 (runs/order_bias_linear_*_k4_same_s5q24):

| order | selection_rate | accuracy_when_gold_present |
| --- | --- | --- |
| gold_first | 0.625 | 0.525 |
| gold_middle | 0.583 | 0.467 |
| gold_last | 0.542 | 0.417 |
| shuffle | 0.583 | 0.442 |

## TF-IDF lexical retriever baseline

This uses cosine similarity over TF-IDF vectors from ledger lines.

TF-IDF result (kv, k=4, s3q16): gold_present_rate 0.0417, selection_rate 1.0, accuracy_when_gold_present 1.0, value_acc 0.0417.
This mirrors BM25: lexical retrieval rarely surfaces the correct update.

```powershell
$env:GOLDEVIDENCEBENCH_RETRIEVAL_RETRIEVER="tfidf"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_K="4"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_RERANK="latest_step"
goldevidencebench sweep --out runs\tfidf_kv_s3q16 --seeds 3 --episodes 1 --steps 240 --queries 16 `
  --state-modes kv --distractor-profiles standard `
  --adapter goldevidencebench.adapters.retrieval_llama_cpp_adapter:create_adapter --no-derived-queries `
  --no-twins --require-citations --results-json runs\tfidf_kv_s3q16\combined.json `
  --max-book-tokens 400 --distractor-rate 0.7 --clear-rate 0.01 --tail-distractor-steps 80
python .\scripts\summarize_results.py --in runs\tfidf_kv_s3q16\combined.json --out-json runs\tfidf_kv_s3q16\summary.json
```

## BM25 baseline (RAG-like retrieval)

Treat each ledger line as a document and retrieve top-k with BM25 before selection.

BM25 result (kv, k=4, s3q16): gold_present_rate 0.0417, selection_rate 1.0, accuracy_when_gold_present 1.0, value_acc 0.0417.
This shows retrieval is the bottleneck: BM25 rarely surfaces the correct update, even though selection works when it does.

```powershell
$env:GOLDEVIDENCEBENCH_RETRIEVAL_RETRIEVER="bm25"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_K="4"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_RERANK="latest_step"
goldevidencebench sweep --out runs\bm25_kv_s3q16 --seeds 3 --episodes 1 --steps 240 --queries 16 `
  --state-modes kv --distractor-profiles standard `
  --adapter goldevidencebench.adapters.retrieval_llama_cpp_adapter:create_adapter --no-derived-queries `
  --no-twins --require-citations --results-json runs\bm25_kv_s3q16\combined.json `
  --max-book-tokens 400 --distractor-rate 0.7 --clear-rate 0.01 --tail-distractor-steps 80
python .\scripts\summarize_results.py --in runs\bm25_kv_s3q16\combined.json --out-json runs\bm25_kv_s3q16\summary.json
```

## Deep dive / repro details

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
- `distractor_rate=0.50`, `clear_rate=0.08`, `note_rate=0.12` (kv_commentary only)
- `distractor_profile=instruction` (adds spec-violating instructions); `instruction_suite` adds quoted/format variants; `adversarial` adds stale-echo distractors
- `state_mode=kv` (switch to `kv_commentary`, `counter`, `set`, or `relational`)
- `require_citations=true` (questions ask for JSON `{value, support_ids}` with max 3)
- Closed-book is the headline score (`goldevidencebench run` defaults to `--protocol closed_book`; open-book is diagnostic)

## Why this benchmark

This benchmark isolates a specific long-context failure mode: **state changes over time** (updates + clears), embedded in a long document with misleading restatements. It's motivated by reports that transformer LLMs can struggle with consistent state tracking across long sequences (see: MIT News, 2025-12-17).

## Attribution & method

This project uses AI-assisted coding and writing, with human review and iteration. The benchmark design, experiments, and results are reproducible; the goal is clarity and scientific usefulness over authorship style.

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
- `state_mode`: `kv|kv_commentary|counter|set|relational`

Derived queries add `meta.query_type=derived` with a `derived_op` and optional `derived_manager` (relational reports).

State dynamics:

- `kv` (default): standard key->value overwrites
- `kv_commentary`: like kv, but inserts non-authoritative NOTE ledger lines (latest_step can be wrong)
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

Speed: what actually dominates runtime

- Prefill usually dominates decode (long book/context). Cut prefill first: keep `--max-book-tokens` small while iterating.
- Use smoke/triage presets during development; save `standard` runs for headline tables.
- Total queries scale as `seeds x state_modes x distractor_profiles x episodes x queries` (twins doubles it).
- If selection is the bottleneck, add a selection-only mode that predicts support_id with minimal output; it can be much faster than full answers.

## Efficient testing workflow (fast -> slow)

Reference system (baseline vs reranker in one command):

Expected output: `runs\summary_all.csv` with rows for selector_quick_none_k2/4/8 and selector_quick_latest_step_k2/4/8.

```powershell
.\scripts\run_reference.ps1 -Preset quick -ModelPath "C:\AI\models\your-model.gguf"
```

Selector+answerer preset (reranker baseline):

```powershell
# no rerank
.\scripts\run_selector_bench.ps1 -Preset quick -ModelPath "C:\AI\models\your-model.gguf"
# with rerank
.\scripts\run_selector_bench.ps1 -Preset quick -ModelPath "C:\AI\models\your-model.gguf" -UseRerank
```

Selector bake-off (quick preset, four rerank modes):

```powershell
.\scripts\run_selector_bakeoff.ps1 -Preset quick -ModelPath "C:\AI\models\your-model.gguf"
```

Expected output: updated `runs\summary_all.csv` with selector_quick_<rerank>_k2/4/8 rows for `none`,
`latest_step`, `last_occurrence`, and `prefer_set_latest`.

Selector-only (fast selection metrics, skip LLM answers):

```powershell
.\scripts\run_selector_only.ps1 -Preset quick -ModelPath "C:\AI\models\your-model.gguf" -Rerank latest_step
```

Use this when tuning selector policies. It skips answer generation and only emits `support_ids`, so the relevant numbers are `gold_present_rate` and `selection_rate` (value accuracy is not meaningful in this mode).

Train a linear selector (no extra deps):

```powershell
python .\scripts\export_selector_dataset.py --data .\data\goldevidencebench.jsonl --out .\data\selector_train.jsonl --k 4 --wrong-type same_key --order shuffle
python .\scripts\train_selector_linear.py --data .\data\selector_train.jsonl --out .\models\linear_selector.json

$env:GOLDEVIDENCEBENCH_RETRIEVAL_RERANK="linear"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_LINEAR_MODEL=".\models\linear_selector.json"
```

The linear selector learns a small scoring function over candidate lines (step, position, op). Use `selection_rate` to compare against `none` or `latest_step`.

Estimate runtime before a sweep:

```powershell
python .\scripts\estimate_runtime.py --from-combined .\runs\combined.json --seeds 3 --episodes 1 --queries 12 --state-modes 2 --distractor-profiles 2 --twins
```

```powershell
python .\scripts\estimate_runtime.py --seeds 3 --episodes 1 --queries 12 --state-modes 2 --distractor-profiles 2 --twins --seconds-per-q 30
```

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

See `docs/ADAPTERS.md` for the full adapter contract, supported adapters, and tuning knobs.

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

These are baseline scores (not the LLM-only selector), shown to illustrate cost/latency, not ambiguity failure.

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


## External anchors

To connect with existing long-context benchmarks, report a small mapping table:

- Order-bias under ambiguity (GoldEvidenceBench) -> positional sensitivity in LongBench / RULER.
- Selection vs gold-present decomposition -> retrieval vs generation split used in RAG evals.

This makes it easy for readers to compare your curves to standard long-context results.

## Related work

See `docs/RELATED.md` for the full link list.
