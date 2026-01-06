# GoldEvidenceBench

GoldEvidenceBench (CLI: `goldevidencebench`) is a benchmark + harness for long-context state tracking. It generates synthetic "episode logs" with evolving state (kv/kv_commentary/counter/set/relational), distractors (including instruction injection), and queries that require answering from the latest state update. It evaluates open-book vs closed-book protocols, enforces citation support IDs with capped-k + F1, checks entailment-from-citations, and uses counterfactual twin episodes to detect shortcut heuristics. It also reports efficiency (tokens/query, passes, wall time) so you can measure capability per compute.

Book artifacts include:

- **Chapters**: narrative text (contains distractors and stale summaries)
- **Glossary (tags)**: a lightweight key reference
- **State ledger**: the authoritative state updates (with support IDs)

## TL;DR (layman summary)

GoldEvidenceBench shows whether your AI system can reliably pick the right piece of evidence when several similar candidates exist. It builds long, noisy logs with changing facts, then checks if the model chooses the most recent, correct update and cites it. The key benefit is that it separates "the evidence was available" from "the model chose the right evidence," so you can improve the exact part of your system that is failing (retrieval vs selection vs formatting).

This helps fix a common real-world failure: the right evidence is retrieved, but the model still picks the wrong snippet. GoldEvidenceBench isolates that selection bottleneck and shows when a simple selector/reranker fixes it.

One-line takeaway: with authority filtering + copy-clamp, kv_commentary is ~1.0 accurate; without the filter, NOTE picks dominate even when extraction is perfect.

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

**V3-D: UPDATE-vs-UPDATE disambiguation + abstain**

Matrix (filter ON/OFF ? spoof_rate ? k):

```powershell
$env:GOLDEVIDENCEBENCH_RETRIEVAL_RERANK = "linear"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_LINEAR_MODEL = ".\models\linear_selector_note_v8.json"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_AUTHORITY_SPOOF_SEED = "0"

foreach ($filter in @('0','1')) {
  if ($filter -eq '1') { $env:GOLDEVIDENCEBENCH_RETRIEVAL_AUTHORITY_FILTER = '1' } else { Remove-Item Env:\GOLDEVIDENCEBENCH_RETRIEVAL_AUTHORITY_FILTER -ErrorAction SilentlyContinue }
  foreach ($spoof in 0.1,0.5,0.8) {
    $env:GOLDEVIDENCEBENCH_RETRIEVAL_AUTHORITY_SPOOF_RATE = "$spoof"
    foreach ($k in 2,4,8) {
      $env:GOLDEVIDENCEBENCH_RETRIEVAL_K = "$k"
      $outDir = "runs\v3d_spoof${spoof}_filter${filter}_k${k}_s3q16"
      goldevidencebench sweep --out $outDir --seeds 3 --episodes 1 --steps 240 --queries 16 `
        --state-modes kv_commentary --distractor-profiles standard `
        --adapter goldevidencebench.adapters.retrieval_llama_cpp_adapter:create_adapter --no-derived-queries `
        --no-twins --require-citations --results-json "$outDir\combined.json" `
        --max-book-tokens 400 --note-rate 0.30
      python .\scripts\summarize_results.py --in "$outDir\combined.json" --out-json "$outDir\summary.json"
    }
  }
}
```

Report: wrong_update_rate, spoof_accept_rate_non_gold, value_acc, and abstain precision/recall (if enabled).

Abstain calibration (drop sweep example, k=4):

```powershell
$env:GOLDEVIDENCEBENCH_RETRIEVAL_ABSTAIN_ON_MISSING = "1"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_DROP_SEED = "0"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_RERANK = "linear"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_K = "4"
foreach ($drop in 0.0,0.2,0.4) {
  $env:GOLDEVIDENCEBENCH_RETRIEVAL_DROP_PROB = "$drop"
  $outDir = "runs\abstain_drop${drop}_k4_s3q16"
  goldevidencebench sweep --out $outDir --seeds 3 --episodes 1 --steps 240 --queries 16 `
    --state-modes kv_commentary --distractor-profiles standard `
    --adapter goldevidencebench.adapters.retrieval_llama_cpp_adapter:create_adapter --no-derived-queries `
    --no-twins --require-citations --results-json "$outDir\combined.json" `
    --max-book-tokens 400 --note-rate 0.30
  python .\scripts\summarize_results.py --in "$outDir\combined.json" --out-json "$outDir\summary.json"
}
```

Use `summary.json` (or `runs/summary_all.csv`) to read `abstain_precision` and `abstain_recall`.

V3-D matrix summary (v8 selector, s3q16):

| filter | spoof_rate | k | value_acc | selected_note_rate | wrong_update_rate | spoof_accept_rate_non_gold |
| --- | --- | --- | --- | --- | --- | --- |
| off | 0.1 | 2/4/8 | 0.6875 | 0.3125 | 0.0000 | 0.0000 |
| off | 0.5 | 2/4/8 | 0.6875 | 0.3125 | 0.0000 | 0.0000 |
| off | 0.8 | 2/4/8 | 0.6875 | 0.3125 | 0.0000 | 0.0000 |
| on | 0.1 | 2/4/8 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| on | 0.5 | 2/4/8 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| on | 0.8 | 2/4/8 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |

Interpretation: spoof exposure rises with spoof_rate/k, but spoofed non-gold selections stay at 0.0; filter OFF fails only via NOTE authority violations, while filter ON is perfect across the grid.

Runs: runs/v3d_spoof{0.1,0.5,0.8}_filter{0,1}_k{2,4,8}_s3q16 (18 runs total).



**V3-A: Learned NOTE robustness (trusted authority field)**

Train with NOTE candidates present, but label the gold UPDATE as correct using `export_selector_dataset.py --use-gold-support` so the selector learns authority, not just recency.

- Train selector with NOTES present.
- Evaluate with the authority filter OFF.
- Authority is a structured feature (not just text pattern).
- Goal: match linear baseline behavior without hard gating.

**V3-B: Authority spoofing (untrusted authority signal)**

- Add a stress profile where some NOTE lines look like UPDATEs, or UPDATE lines contain NOTE text, or the authority marker is missing/ambiguous.
- Measure how often the selector is tricked and whether abstain triggers when authority is unclear.
- Report spoof_accept_rate + gold_support_selected_rate.

Command (s3q16 example, spoof cues at 30%):

```powershell
Remove-Item Env:\GOLDEVIDENCEBENCH_RETRIEVAL_AUTHORITY_FILTER -ErrorAction SilentlyContinue
$env:GOLDEVIDENCEBENCH_RETRIEVAL_AUTHORITY_SPOOF_RATE = "0.3"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_AUTHORITY_SPOOF_SEED = "0"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_RERANK = "linear"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_K = "4"
$outDir = "runs\authority_spoof_0.3_linear_k4_s3q16"
goldevidencebench sweep --out $outDir --seeds 3 --episodes 1 --steps 240 --queries 16 `
  --state-modes kv_commentary --distractor-profiles standard `
  --adapter goldevidencebench.adapters.retrieval_llama_cpp_adapter:create_adapter --no-derived-queries `
  --no-twins --require-citations --results-json "$outDir\combined.json" `
  --max-book-tokens 400 --note-rate 0.30
python .\scripts\summarize_results.py --in "$outDir\combined.json" --out-json "$outDir\summary.json"
```
UPDATE burst stress test (new distractor profile):

`update_burst` injects rapid same-key UPDATE bursts with near-miss values so the latest update is easy to confuse with the previous one. This stresses wrong-UPDATE selection.

Optional: set `GOLDEVIDENCEBENCH_UPDATE_BURST_RATE` (or `--update-burst-rate`) to control burst probability; default is 0.25.

```powershell
$outDir = "runs\update_burst_k4_s3q16"
goldevidencebench sweep --out $outDir --seeds 3 --episodes 1 --steps 240 --queries 16 `
  --state-modes kv --distractor-profiles update_burst `
  --adapter goldevidencebench.adapters.retrieval_llama_cpp_adapter:create_adapter --no-derived-queries `
  --no-twins --require-citations --results-json "$outDir\combined.json" `
  --max-book-tokens 400
python .\scripts\summarize_results.py --in "$outDir\combined.json" --out-json "$outDir\summary.json"
```


Update-burst rate sweep (k=4, steps=240, s5q24; gold_present_rate=1.0):

| burst_rate | selection_rate | wrong_update_rate | value_acc |
| --- | --- | --- | --- |
| 0.10 | 0.9500 | 0.0500 | 0.9500 |
| 0.25 | 0.9000 | 0.1000 | 0.9000 |
| 0.40 | 0.7833 | 0.2167 | 0.7833 |

Runs: runs/update_burst_rate0.10_k4_steps240_s5q24, runs/update_burst_rate0.25_k4_steps240_s5q24, runs/update_burst_rate0.40_k4_steps240_s5q24.

Interpretation: higher burst density increases wrong-UPDATE selection and lowers accuracy; this trend is clearer with s5q24 than the smaller s3q16 checks.

Negative result (update_burst training, k=8, rate=0.40, s5q24):

| selector | selection_rate | wrong_update_rate | value_acc |
| --- | --- | --- | --- |
| baseline linear | 0.8917 | 0.1083 | 0.8917 |
| update_burst-trained | 0.8250 | 0.1750 | 0.8250 |

Runs: runs/update_burst_linear_baseline_k8_rate0.40_s5q24, runs/update_burst_linear_updatebursttrain_k8_rate0.40_s5q24.

Interpretation: hard-negative training on update_burst did not improve wrong-UPDATE selection; it degraded performance in this regime.

Optional linear tie-breaker A/B (k=8, rate=0.40, s5q24; gold_present_rate=1.0):

| tie_break | selection_rate | wrong_update_rate | value_acc |
| --- | --- | --- | --- |
| none | 0.7583 | 0.2417 | 0.7583 |
| latest_step | 0.8417 | 0.1583 | 0.8417 |

Runs: runs/update_burst_linear_tienone_k8_rate0.40_s5q24, runs/update_burst_linear_tielatest_step_k8_rate0.40_s5q24.

Interpretation: in the hardest regime (k=8, rate=0.40), recency tie-breaking materially improves wrong-UPDATE selection.
Wall cutoff (k=8, s5q24): rate=0.33 holds (value_acc=0.9250, wrong_update_rate=0.0750), rate=0.34 drops (value_acc=0.8500, wrong_update_rate=0.1500). Runs: runs/update_burst_linear_k8_rate0.33_s5q24, runs/update_burst_linear_k8_rate0.34_s5q24.

Selector-only wall map (k=8, s5q24; gold_present_rate=1.0):

| rate | rerank | selection_rate | wrong_update_rate |
| --- | --- | --- | --- |
| 0.25 | none | 0.2500 | 0.7500 |
| 0.25 | linear | 0.8583 | 0.1417 |
| 0.25 | latest_step | 1.0000 | 0.0000 |
| 0.30 | none | 0.1667 | 0.8333 |
| 0.30 | linear | 0.9250 | 0.0750 |
| 0.30 | latest_step | 1.0000 | 0.0000 |
| 0.35 | none | 0.0417 | 0.9583 |
| 0.35 | linear | 0.9000 | 0.1000 |
| 0.35 | latest_step | 1.0000 | 0.0000 |
| 0.40 | none | 0.1250 | 0.8750 |
| 0.40 | linear | 0.8333 | 0.1667 |
| 0.40 | latest_step | 1.0000 | 0.0000 |

Runs: runs/update_burst_selonly_none_k8_rate0.25_s5q24, runs/update_burst_selonly_linear_k8_rate0.25_s5q24, runs/update_burst_selonly_latest_step_k8_rate0.25_s5q24,
      runs/update_burst_selonly_none_k8_rate0.30_s5q24, runs/update_burst_selonly_linear_k8_rate0.30_s5q24, runs/update_burst_selonly_latest_step_k8_rate0.30_s5q24,
      runs/update_burst_selonly_none_k8_rate0.35_s5q24, runs/update_burst_selonly_linear_k8_rate0.35_s5q24, runs/update_burst_selonly_latest_step_k8_rate0.35_s5q24,
      runs/update_burst_selonly_none_k8_rate0.40_s5q24, runs/update_burst_selonly_linear_k8_rate0.40_s5q24, runs/update_burst_selonly_latest_step_k8_rate0.40_s5q24.

Interpretation: in selector-only mode, latest_step fully resolves wrong-UPDATE selection under update_burst; linear helps but does not saturate.



Selector-only update_burst (k=8, s3q16, linear):

| rate | selection_rate | wrong_update_rate |
| --- | --- | --- |
| 0.30 | 1.0000 | 0.0000 |
| 0.40 | 1.0000 | 0.0000 |
| 0.50 | 1.0000 | 0.0000 |

Runs: runs/update_burst_linear_k8_rate0.30_s3q16, runs/update_burst_linear_k8_rate0.40_s3q16, runs/update_burst_linear_k8_rate0.50_s3q16.

Interpretation: at the smaller s3q16 setting, linear selection saturates even at high burst rates; the hard regime appears in the larger s5q24 setting and in full-pipeline runs.

Full pipeline update_burst (k=8, rate=0.40, s3q16; gold_present_rate=1.0):

| copy_clamp | selection_rate | wrong_update_rate | accuracy_when_gold_present | value_acc |
| --- | --- | --- | --- | --- |
| 0 | 0.6875 | 0.3125 | 0.0625 | 0.0625 |
| 1 | 0.2500 | 0.7500 | 0.0625 | 0.0625 |

Runs: runs/update_burst_full_linear_copy0_k8_rate0.40_s3q16, runs/update_burst_full_linear_copy1_k8_rate0.40_s3q16.

Interpretation: copy-clamp does not fix update_burst in the full pipeline; selection collapses when clamp=1 and wrong-UPDATE selection dominates either way. Keep clamp off for update_burst; the linear tie-break below also underperforms in this regime, so focus on stronger UPDATE disambiguation features.

Full pipeline update_burst with linear tie-break (k=8, rate=0.40, s3q16; gold_present_rate=1.0):

| selection_rate | wrong_update_rate | accuracy_when_gold_present | value_acc |
| --- | --- | --- | --- |
| 0.4167 | 0.5833 | 0.1250 | 0.1250 |

Run: runs/update_burst_full_linear_tie_k8_rate0.40_s3q16.

Interpretation: the tie-breaker underperforms the baseline in this full-pipeline setting (lower selection_rate, higher wrong_update_rate). It is not a safe default here.

Full pipeline update_burst with prefer_update_latest (k=8, rate=0.40, s3q16; gold_present_rate=1.0):

| selection_rate | wrong_update_rate | accuracy_when_gold_present | value_acc |
| --- | --- | --- | --- |
| 0.6250 | 0.3750 | 0.1250 | 0.1250 |

Run: runs/update_burst_full_prefer_update_latest_k8_rate0.40_s3q16.

Interpretation: prefer_update_latest improves selection vs tie-break, but still leaves substantial wrong-UPDATE errors and low value accuracy. The bottleneck remains UPDATE disambiguation and answer extraction.

Full pipeline update_burst with prefer_update_latest + copy-clamp (k=8, rate=0.40, s3q16; gold_present_rate=1.0):

| selection_rate | wrong_update_rate | accuracy_when_gold_present | value_acc |
| --- | --- | --- | --- |
| 0.4375 | 0.5625 | 0.1250 | 0.1250 |

Run: runs/update_burst_full_prefer_update_latest_copy1_k8_rate0.40_s3q16.

Interpretation: copy-clamp does not improve prefer_update_latest in the update_burst full pipeline; selection drops and wrong-UPDATE errors dominate. Treat copy-clamp as unsafe in this regime.

Full pipeline update_burst A/B (k=8, s3q16; gold_present_rate=1.0):

| burst_rate | rerank | selection_rate | wrong_update_rate | accuracy_when_gold_present | value_acc |
| --- | --- | --- | --- | --- | --- |
| 0.25 | linear | 0.7083 | 0.2917 | 0.0417 | 0.0417 |
| 0.25 | prefer_update_latest | 0.6250 | 0.3750 | 0.0417 | 0.0417 |
| 0.30 | linear | 0.6250 | 0.3750 | 0.1458 | 0.1458 |
| 0.30 | prefer_update_latest | 0.5000 | 0.5000 | 0.1458 | 0.1458 |
| 0.35 | linear | 0.6667 | 0.3333 | 0.0833 | 0.0833 |
| 0.35 | prefer_update_latest | 0.5000 | 0.5000 | 0.0833 | 0.0833 |

Runs: runs/update_burst_full_linear_k8_rate0.25_s3q16, runs/update_burst_full_prefer_update_latest_k8_rate0.25_s3q16, runs/update_burst_full_linear_k8_rate0.30_s3q16, runs/update_burst_full_prefer_update_latest_k8_rate0.30_s3q16, runs/update_burst_full_linear_k8_rate0.35_s3q16, runs/update_burst_full_prefer_update_latest_k8_rate0.35_s3q16.

Interpretation: linear beats prefer_update_latest on selection rate at 0.25 and 0.35, but accuracy_when_gold_present is very low across these runs, so answerer/extraction dominates. If this is unexpected, double-check that no answerer clamps or deterministic modes were left on in the environment.

Deterministic answerer check (k=8, s3q16; gold_present_rate=1.0):

| burst_rate | rerank | selection_rate | wrong_update_rate | accuracy_when_gold_present | value_acc | answer_acc_given_gold_selected |
| --- | --- | --- | --- | --- | --- | --- |
| 0.25 | linear | 0.5208 | 0.4792 | 0.0417 | 0.0417 | 0.0400 |
| 0.25 | prefer_update_latest | 0.5000 | 0.5000 | 0.0417 | 0.0417 | 0.0833 |
| 0.35 | linear | 0.2917 | 0.7083 | 0.0833 | 0.0833 | 0.1429 |
| 0.35 | prefer_update_latest | 0.4375 | 0.5625 | 0.0833 | 0.0833 | 0.0000 |

Runs: runs/update_burst_full_linear_detanswer_k8_rate0.25_s3q16, runs/update_burst_full_prefer_update_latest_detanswer_k8_rate0.25_s3q16, runs/update_burst_full_linear_detanswer_k8_rate0.35_s3q16, runs/update_burst_full_prefer_update_latest_detanswer_k8_rate0.35_s3q16.

Interpretation: deterministic answer did not lift accuracy_when_gold_present; answer_acc_given_gold_selected is still low. That means the deterministic path is either not active in this regime or the selected lines are not parse-stable. Treat this as a diagnostic (validate the env var and parsing rules before interpreting it as a model failure).

Clean sanity check (kv, s1q4, deterministic answerer, clean env): value_acc=1.0, selection_rate=1.0, answer_acc_given_gold_selected=1.0.
Run: runs/detanswer_sanity_kv_s1q4_clean.

Clean update_burst deterministic answerer (k=8, s3q16; clean env):

| rerank | burst_rate | selection_rate | wrong_update_rate | accuracy_when_gold_present | value_acc | answer_acc_given_gold_selected |
| --- | --- | --- | --- | --- | --- | --- |
| linear | 0.25 | 0.9167 | 0.0833 | 0.9167 | 0.9167 | 1.0000 |
| linear | 0.35 | 0.8750 | 0.1250 | 0.8750 | 0.8750 | 1.0000 |
| prefer_update_latest | 0.25 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| prefer_update_latest | 0.35 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |

Runs: runs/update_burst_full_linear_detanswer_k8_rate0.25_s3q16_clean, runs/update_burst_full_linear_detanswer_k8_rate0.35_s3q16_clean, runs/update_burst_full_prefer_update_latest_detanswer_k8_rate0.25_s3q16_clean, runs/update_burst_full_prefer_update_latest_detanswer_k8_rate0.35_s3q16_clean.

Interpretation: with a clean env, deterministic answerer works (answer_acc_given_gold_selected=1.0). Remaining error is selection; prefer_update_latest resolves wrong-update picks in this regime.

Environment hygiene (when to clean):

- Before any deterministic answerer diagnostic (to avoid stale clamps/flags masking results).
- After switching state mode, distractor profile, or reranker model.
- After toggling any retrieval/answerer flags (authority filter, copy-clamp, deterministic answer, abstain, drop_prob, k).
- If results look inconsistent with known baselines.

Clean env command (PowerShell):

```powershell
Remove-Item Env:\GOLDEVIDENCEBENCH_* -ErrorAction SilentlyContinue
```

Selector-only update_burst A/B (k=8, rate=0.40, s3q16; gold_present_rate=1.0):

| rerank | selection_rate | wrong_update_rate | accuracy_when_gold_present | value_acc |
| --- | --- | --- | --- | --- |
| linear | 0.3750 | 0.6250 | 0.1250 | 0.1250 |
| prefer_update_latest | 0.6250 | 0.3750 | 0.1250 | 0.1250 |

Runs: runs/update_burst_selonly_linear_k8_rate0.40_s3q16, runs/update_burst_selonly_prefer_update_latest_k8_rate0.40_s3q16.

Interpretation: even in selector-only mode, prefer_update_latest materially outperforms linear on UPDATE disambiguation. The remaining error is selection (wrong-UPDATE) rather than answer extraction.







Abstain calibration (selection-only, update_burst k=8 rate=0.34, s3q16):

| drop_prob | abstain_rate | abstain_precision | abstain_recall | selection_rate | wrong_update_rate |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.0000 | ? | ? | 0.8750 | 0.1250 |
| 0.2 | 0.1579 | 1.0000 | 1.0000 | 0.7292 | 0.1026 |
| 0.4 | 0.2727 | 1.0000 | 1.0000 | 0.5417 | 0.1333 |

Runs: runs/abstain_update_burst_k8_rate0.34_drop{0,0.2,0.4}_s3q16.

Interpretation: abstain triggers only when the gold line is dropped from candidates, with perfect precision/recall in this sweep; baseline (drop=0) does not abstain.

Full pipeline abstain calibration (update_burst k=8 rate=0.34, s3q16):

| drop_prob | value_acc | abstain_rate | abstain_precision | abstain_recall | selection_rate | wrong_update_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.9167 | 0.0000 | ? | ? | 0.9167 | 0.0833 |
| 0.2 | 0.5208 | 0.3043 | 1.0000 | 1.0000 | 0.4792 | 0.1481 |
| 0.4 | 0.6042 | 0.2727 | 1.0000 | 1.0000 | 0.5625 | 0.1000 |

Runs: runs/abstain_update_burst_k8_rate0.34_drop{0,0.2,0.4}_s3q16_full.

Interpretation: abstain remains perfectly calibrated end-to-end; accuracy drops as expected when gold is removed from the candidate set.




Negative result (recency_v2 feature, k=8, s5q24; gold_present_rate=1.0):

| rate | selector | selection_rate | wrong_update_rate | value_acc |
| --- | --- | --- | --- | --- |
| 0.35 | baseline linear | 0.8917 | 0.1083 | 0.8917 |
| 0.35 | recency_v2 | 0.8333 | 0.1667 | 0.8333 |
| 0.40 | baseline linear | 0.8000 | 0.2000 | 0.8000 |
| 0.40 | recency_v2 | 0.6667 | 0.3333 | 0.6667 |

Runs: runs/update_burst_linear_baseline_k8_rate0.35_s5q24, runs/update_burst_linear_recency_v2_k8_rate0.35_s5q24,
      runs/update_burst_linear_baseline_k8_rate0.40_s5q24, runs/update_burst_linear_recency_v2_k8_rate0.40_s5q24.

Interpretation: the added recency_v2 feature regresses wrong-UPDATE selection at high burst rates; keep the tie-breaker, avoid this training tweak.

Negative result (recency-rank retrain v9, k=8, s5q24; gold_present_rate=1.0):

| rate | selector | selection_rate | wrong_update_rate | value_acc |
| --- | --- | --- | --- | --- |
| 0.33 | baseline linear | 0.9250 | 0.0750 | 0.9250 |
| 0.33 | recency_rank_v9 | 0.2917 | 0.7083 | 0.2917 |
| 0.34 | baseline linear | 0.8500 | 0.1500 | 0.8500 |
| 0.34 | recency_rank_v9 | 0.2500 | 0.7500 | 0.2500 |

Runs: runs/update_burst_linear_k8_rate0.33_s5q24, runs/update_burst_linear_k8_rate0.34_s5q24,
      runs/update_burst_linear_v9_k8_rate0.33_s5q24, runs/update_burst_linear_v9_k8_rate0.34_s5q24.

Interpretation: the recency-rank feature retrain collapsed under update_burst; do not use v9 weights.

Negative result (recency-rank retrain v9b, low LR + wrong-update penalty, k=8, s5q24; gold_present_rate=1.0):

| rate | selector | selection_rate | wrong_update_rate | value_acc |
| --- | --- | --- | --- | --- |
| 0.33 | baseline linear | 0.9250 | 0.0750 | 0.9250 |
| 0.33 | recency_rank_v9b | 0.0833 | 0.9167 | 0.0917 |
| 0.34 | baseline linear | 0.8500 | 0.1500 | 0.8500 |
| 0.34 | recency_rank_v9b | 0.1500 | 0.8500 | 0.1500 |

Runs: runs/update_burst_linear_k8_rate0.33_s5q24, runs/update_burst_linear_k8_rate0.34_s5q24,
      runs/update_burst_linear_v9b_k8_rate0.33_s5q24, runs/update_burst_linear_v9b_k8_rate0.34_s5q24.

Interpretation: low-LR + wrong-update penalty retrain made wrong-UPDATE selection much worse; avoid v9b.






V3-B authority spoofing results (s3q16, k=4, gold_present_rate = 1.0):

| spoof_rate | value_acc | selection_rate | gold_support_selected_rate | selected_note_rate | wrong_update_rate | spoof_accept_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.7708 | 0.6042 | 0.7708 | 0.0833 | 0.1458 | 0.0833 |
| 0.5 | 0.4375 | 0.4375 | 0.4375 | 0.2500 | 0.3125 | 0.2708 |

Runs: runs/authority_spoof_0.1_linear_k4_s3q16, runs/authority_spoof_0.5_linear_k4_s3q16.

Interpretation: higher spoofing raises spoof_accept_rate and wrong_update_rate, and accuracy tracks the resulting selection collapse.

V3-B spoof filter A/B (prefer_update_latest, spoof_rate=0.5, s3q16, k=4):

| authority_filter | value_acc | selection_rate | gold_support_selected_rate | selected_note_rate | wrong_update_rate | spoof_accept_rate |
| --- | --- | --- | --- | --- | --- | --- |
| off | 0.2708 | 0.4167 | 0.2708 | 0.3750 | 0.3542 | 0.3958 |
| on | 0.3125 | 0.3125 | 0.3125 | 0.0000 | 0.6875 | 0.0625 |

Interpretation: the filter eliminates NOTE picks and sharply lowers spoof acceptance; remaining errors are almost entirely wrong UPDATE selection.

Takeaway: with spoofing, authority filtering solves NOTE/spoof confusion but exposes the next bottleneck (wrong UPDATE selection).

NOTE camouflage stress test (new distractor profile):

`note_camouflage` makes NOTE lines look like updates (e.g., "UPDATE: SET ..." or quoted UPDATEs) while keeping them non-authoritative. This tests whether selectors are robust to NOTE camouflage, not just NOTE labels.

```powershell
$outDir = "runs\note_camouflage_k4_s3q16"
goldevidencebench sweep --out $outDir --seeds 3 --episodes 1 --steps 240 --queries 16 `
  --state-modes kv_commentary --distractor-profiles note_camouflage `
  --adapter goldevidencebench.adapters.retrieval_llama_cpp_adapter:create_adapter --no-derived-queries `
  --no-twins --require-citations --results-json "$outDir\combined.json" `
  --max-book-tokens 400 --note-rate 0.30
python .\scripts\summarize_results.py --in "$outDir\combined.json" --out-json "$outDir\summary.json"
```

Selector training under spoofing (optional): export with `--authority-spoof-rate`/`--authority-spoof-seed` and train with `--spoof-penalty` to penalize spoofed candidates. This can be combined with `--hard-negatives` for wrong?UPDATE pressure.

V3-B order generalization (v8: spoofpen+hardneg, filter ON, spoof_rate=0.5, k=4, s3q16):

| order | value_acc | selection_rate | gold_support_selected_rate | wrong_update_rate | spoof_accept_rate | spoof_accept_rate_non_gold |
| --- | --- | --- | --- | --- | --- | --- |
| gold_first | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.3125 | 0.0000 |
| gold_middle | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 |
| gold_last | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.4375 | 0.0000 |
| shuffle | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 |

Runs: runs/authority_spoof_0.5_filter1_linear_spoofpen_hardneg_{gold_first,gold_middle,gold_last,shuffle}_k4_s3q16.

Interpretation: accuracy is order-invariant and spoof exposure does not cause wrong picks (spoof_accept_rate_non_gold = 0).

Note: the remaining bottleneck depends on the regime. With filter ON and the v8 selector, wrong-UPDATE selection is resolved in this setting; without the filter or without hard-negative training, wrong-UPDATE and NOTE authority errors reappear.

Filter-OFF check (v8, spoof_rate=0.5, k=4, s3q16): value_acc=0.6875, gold_support_selected_rate=0.6875, selected_note_rate=0.3125, spoof_accept_rate=0.625, spoof_accept_rate_non_gold=0.0.
Interpretation: without the authority filter, the selector still avoids wrong UPDATEs but accepts NOTE lines; the hard gate remains required.





**V3-C: Answer contract (extraction clamp)**

- Set `GOLDEVIDENCEBENCH_RETRIEVAL_DETERMINISTIC_ANSWER=1` to return the value parsed from the selected ledger line (no LLM).
- Set `GOLDEVIDENCEBENCH_RETRIEVAL_COPY_CLAMP=1` to require the answer value be an exact substring of the selected line; otherwise return null.
- Use this as an oracle upper bound: if accuracy_when_gold_present rises to selection_rate, the remaining gap was answerer/extraction.


V3-C results (kv_commentary, k=4 same_key, s3q16; authority filter ON + copy-clamp ON):

| order | value_acc | selection_rate | selected_note_rate | wrong_update_rate |
| --- | --- | --- | --- | --- |
| gold_first | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| gold_middle | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| gold_last | 0.9583 | 0.9583 | 0.0000 | 0.0417 |
| shuffle | 0.9792 | 0.9792 | 0.0000 | 0.0208 |

Runs: runs/kv_commentary_noteaware_copyclamp_filter_{gold_first,gold_middle,gold_last,shuffle}_k4_s3q16.

Interpretation: authority gating removes NOTE errors and copy-clamp removes extraction drift; remaining error is rare wrong-UPDATE selection.


Contrast (filter OFF + copy-clamp ON; same settings):

| order | value_acc | selection_rate | selected_note_rate | wrong_update_rate |
| --- | --- | --- | --- | --- |
| gold_first | 0.6875 | 1.0000 | 0.3125 | 0.0000 |
| gold_middle | 0.6875 | 1.0000 | 0.3125 | 0.0000 |
| gold_last | 0.6458 | 0.9583 | 0.3125 | 0.0417 |
| shuffle | 0.6875 | 1.0000 | 0.3125 | 0.0000 |

Runs: runs/kv_commentary_noteaware_copyclamp_{gold_first,gold_middle,gold_last,shuffle}_k4_s3q16.

Interpretation: copy-clamp fixes extraction, but NOTE authority errors remain without the filter (selected_note_rate ~31%).




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
- spoof_accept_rate: share of selections that cite a spoofed NOTE/UPDATE line (V3-B).
- spoof_accept_rate_non_gold: share of selections that are spoofed *and* non-gold (direct spoof-caused errors).

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

## Failure modes and mitigation map

The goal is to identify where accuracy drops, measure it with a specific metric, and apply a targeted mitigation. Each failure mode is a real-world risk because it changes which state update gets trusted.

| failure point | trigger / regime | drop-off symptom | mitigation / stitch |
| --- | --- | --- | --- |
| Retrieval miss | lexical retrievers, high distractor rate | low `gold_present_rate` | improve retriever, raise `k`, dense baseline; validate retrieval before selector |
| Authority confusion (NOTE vs UPDATE) | `kv_commentary`, NOTE-heavy logs | `selected_note_rate` > 0 | hard authority filter + note-aware selector (gate before rerank) |
| Wrong-UPDATE selection | `same_key`, `update_burst`, high `k` | rising `wrong_update_rate` | `prefer_update_latest` or recency tie-break; train with same_key hard negatives |
| Order bias | gold order sweeps | spread in `selection_rate` across orders | deterministic selector; shuffle training; tie-break |
| Extraction drift | answerer instability | `answer_acc_given_gold_selected` < 1 or low substring rate | deterministic answerer + copy-clamp; inspect parsing rules |
| Missing gold | gold dropped from candidates | low accuracy with `abstain_rate` near 0 | enable `abstain_on_missing` and calibrate drop sweep |
| Instruction override | instruction distractor profile | elevated `instr_override_rate` / `instr_gap` | citations + authority gate; treat instructions as non-authoritative |

Stitching patterns (composable fixes):

- Strict mode: authority filter -> selector (linear or prefer_update_latest) -> deterministic answer or copy-clamp -> abstain on missing.
- Diagnostic mode: selector-only to isolate retrieval vs selection; then re-run full pipeline.
- Robustness mode: order sweeps + update_burst to find the drop-off wall before changing defaults.

## Current plan (update_burst wall, 2026-01)

Stress regime (linear + step_bucket=10 + k=16) with seeds=3 fails the wrong_update_rate <= 0.10 gate even at
update_burst_rate=0.02. That means this diagnostic regime is beyond the 0.10 gate for any nonzero burst rate.
The earlier 1-seed wall (~0.18-0.19) was variance.

Seeds=3 pin runs used to locate the wall (full pipeline):

- `runs/wall_update_burst_full_linear_bucket10_pin_20260104_200253_s3` (0.175-0.19)
- `runs/wall_update_burst_full_linear_bucket10_pin_20260104_215915_s3_low` (0.10-0.17)
- `runs/wall_update_burst_full_linear_bucket10_pin_20260104_233152_s3_micro` (0.02-0.08)

Micro sweep (seeds=3; full pipeline, linear + step_bucket=10 + k=16):

| update_burst_rate | selection_rate | wrong_update_rate | value_acc |
| --- | --- | --- | --- |
| 0.02 | 0.8750 | 0.1250 | 0.8750 |
| 0.04 | 0.7708 | 0.2292 | 0.7708 |
| 0.06 | 0.8125 | 0.1875 | 0.8125 |
| 0.08 | 0.7500 | 0.2500 | 0.7500 |

Production gate (prefer_update_latest, k=8, full pipeline, seeds=3):

- `runs/update_burst_prefer_update_latest_gate_20260105_025033` (0.25, 0.35, 0.45)
- `runs/update_burst_prefer_update_latest_gate_20260105_133108_full_s3` (0.95)
- `runs/update_burst_prefer_update_latest_gate_20260105_142019_full_s3_099` (0.99)
 - Selector-only quick probe: `runs/update_burst_prefer_update_latest_gate_20260105_152002_quick5` (1.0; value_acc not meaningful)
Stable gate path (frozen for checks): `runs/release_gates/update_burst_prefer_update_latest_k8_rate0.99`.

| update_burst_rate | selection_rate | wrong_update_rate | value_acc |
| --- | --- | --- | --- |
| 0.25 | 1.0000 | 0.0000 | 1.0000 |
| 0.35 | 1.0000 | 0.0000 | 1.0000 |
| 0.45 | 1.0000 | 0.0000 | 1.0000 |
| 0.95 | 1.0000 | 0.0000 | 1.0000 |
| 0.99 | 1.0000 | 0.0000 | 1.0000 |

Selector-only auto-pin run (diagnostic only; value_acc not meaningful): `runs/wall_update_burst_20260104_190551`.

Plan going forward:

1) Keep the linear+bucket10 stress regime as diagnostic only (always beyond the 0.10 gate).
2) Use the prefer_update_latest 0.99 full-pipeline run above as the production gate in `configs/usecase_checks.json`.
3) Wall not found up to 0.99; treat the production default as robust in update_burst at this scale.
4) Keep the UI same_label gate (`runs/ui_same_label_gate.json`) as a release check for UI-adapter readiness.

Canonical wall sweep commands (frozen):

```powershell
# Stress regime (full pipeline, linear + step_bucket=10 + k=16)
.\scripts\run_update_burst_full_linear_bucket10.ps1 `
  -ModelPath "C:\AI\models\your-model.gguf" `
  -OutRoot "runs\wall_update_burst_full_linear_bucket10_20260104_180252" `
  -Rates 0.205,0.209,0.22,0.24

# Pin sweep (same regime, lower rates)
.\scripts\run_update_burst_full_linear_bucket10.ps1 `
  -ModelPath "C:\AI\models\your-model.gguf" `
  -OutRoot "runs\wall_update_burst_full_linear_bucket10_pin_20260104_180252" `
  -Rates 0.18,0.19,0.195,0.20 `
  -FindWall:$true
```

## Computer-use agents (why this benchmark maps)

Computer-use agents share the same selection-under-ambiguity failure mode, but the domains are not identical.
This benchmark reuses the selection/evidence discipline; it does not solve perception or planning.

In a UI, the agent must choose one action among many plausible candidates and update state correctly.
The most damaging error is a wrong action with no detection (silent failure), so the extension should include
a post-action verification contract: did the expected UI state delta occur?

Minimal extension to cover UI agents:

- **Adapter**: convert the accessibility tree/DOM into a candidate list (actions), not pixels.
- **Gold**: the correct action for the step.
- **Distractors**: same-label buttons, popups, overlays, and near-miss targets.
- **Post-action verification**: capture the expected UI delta and mark mismatches for abstain/rollback/alert.
- **Metrics**: `gold_present_rate`, `selection_rate`, UI wrong action rate (same as `wrong_update_rate`),
  post-action verify rate, and abstain precision/recall.

First milestone (highest leverage):

- **same_label profile**: duplicate "Next/Continue/Save" buttons, measure selection vs ambiguity.
- **Wall sweep**: increase duplicates until `wrong_update_rate` crosses a release gate.
- **Gate**: declare "safe up to X ambiguity; abstain beyond that."

If this lands, you get the same decomposition you already use for text logs, but applied to real UI actions:
retrieval vs selection vs abstain, with a concrete, publishable wall.

Adoption hook (CI gates):

- `selection_rate` floor under `same_label`.
- `wrong_update_rate` ceiling under `same_label`.
- post-action verify rate floor under confusers.
- `abstain_precision/recall` under missing-gold sweeps.
- `gold_present_rate` for retriever coverage.

Adapter checklist (minimal):

- Extract action candidates from the accessibility tree/DOM (click, type, select).
- Emit stable `candidate_id` and `action_type` for each candidate.
- Provide gold `candidate_id` per step (authoritative).
- Log distractor metadata (label similarity, bbox overlap, z-order, modal scope).
- Record the expected post-action UI delta and the verification result.
- Support `abstain` when no candidate meets confidence or gold is missing.

Minimal UI candidate schema (example):

```json
{
  "id": "step_0007",
  "candidates": [
    {
      "candidate_id": "btn_save_primary",
      "action_type": "click",
      "label": "Save",
      "role": "button",
      "app_path": "Settings > Profile",
      "bbox": [120, 440, 80, 28],
      "visible": true,
      "enabled": true,
      "modal_scope": "profile_dialog"
    }
  ],
  "gold": {
    "candidate_id": "btn_save_primary"
  }
}
```

Stub assets (UI same_label):

- Config: `configs/ui_same_label.json`
- Fixture: `data/ui_same_label_fixture.jsonl`
- Script: `scripts/run_ui_same_label_stub.ps1`

Run the stub script (validates paths, no adapter yet):

```powershell
.\scripts\run_ui_same_label_stub.ps1
```

This writes `runs/ui_same_label_gate.json` and `runs/ui_same_label_summary.json` (used by the threshold checks).

Validate the fixture directly:

```powershell
python .\scripts\validate_ui_fixture.py --fixture .\data\ui_same_label_fixture.jsonl
```

Score the fixture with simple selection modes:

```powershell
python .\scripts\score_ui_fixture.py --fixture .\data\ui_same_label_fixture.jsonl --mode gold
python .\scripts\score_ui_fixture.py --fixture .\data\ui_same_label_fixture.jsonl --mode first
```

Summarize the fixture:

```powershell
python .\scripts\summarize_ui_fixture.py --fixture .\data\ui_same_label_fixture.jsonl
```

Score with post-action verification (observed deltas):

```powershell
python .\scripts\score_ui_fixture.py --fixture .\data\ui_same_label_fixture.jsonl `
  --observed .\data\ui_same_label_observed_ok.jsonl --mode gold
```

Run the UI adapter stub (fixture-based selection):

```powershell
python .\scripts\run_ui_adapter_stub.py --fixture .\data\ui_same_label_fixture.jsonl
```

Run the CLI command (same adapter, JSON output):

```powershell
goldevidencebench ui-score --fixture .\data\ui_same_label_fixture.jsonl --out .\runs\ui_same_label_metrics.json
```

Summarize via CLI:

```powershell
goldevidencebench ui-summary --fixture .\data\ui_same_label_fixture.jsonl --out .\runs\ui_same_label_summary.json
```

UI same_label wall sweep (vary duplicates; writes JSON/CSV summary):

```powershell
.\scripts\run_ui_same_label_wall.ps1 -Duplicates 1,2,3,4,5 -Steps 6
```

Tip: for the stub adapter, add `-SelectionMode first` to see the failure curve for a naive policy.
This also writes a stable snapshot to `runs/ui_same_label_wall_latest/score.json` for threshold checks.

UI wall finder (writes the ceiling into `configs/usecase_checks.json`):

```powershell
python .\scripts\find_ui_wall.py --runs-dir .\runs\ui_same_label_wall_20260105_153000 `
  --metric metrics.wrong_action_rate --threshold 0.10 --direction gte `
  --update-config .\configs\usecase_checks.json --check-id ui_same_label_wall
```

Generate a new same-label fixture (example):

```powershell
python .\scripts\generate_ui_fixture.py --out .\data\ui_same_label_generated.jsonl `
  --steps 5 --duplicates 3 --labels "Next,Continue,Save" --seed 0
```

Generate via CLI:

```powershell
goldevidencebench ui-generate --out .\data\ui_same_label_generated.jsonl `
  --steps 5 --duplicates 3 --labels "Next,Continue,Save" --seed 0
```

Expected stub results (fixture sanity checks):

- `score_ui_fixture.py --mode gold`: selection_rate=1.0, wrong_action_rate=0.0
- `score_ui_fixture.py --mode first`: selection_rate=0.6667, wrong_action_rate=0.3333
- `score_ui_fixture.py --observed ... --mode gold`: post_action_verify_rate=1.0
- `ui-score` (default adapter): selection_rate=0.6667, wrong_action_rate=0.3333

## Impact / use-case profiles

These are concrete, world-facing uses and the minimum checks that keep them safe. Each profile maps to a failure mode and a short sweep that validates it.

| use case | why it matters | minimum checks |
| --- | --- | --- |
| Authoritative change logs (orders, billing, config) | wrong update silently changes state | `same_key` + order sweep; watch `wrong_update_rate` and `selection_rate` |
| Policy vs commentary pipelines (support/medical notes) | NOTE lines must not mutate state | `kv_commentary` with authority filter ON; `selected_note_rate` ~ 0 |
| Retrieval-heavy RAG | evidence may be missing or buried | `gold_present_rate` from dense/lexical baselines; raise `k` or swap retriever |
| High-risk decisions (compliance, approvals) | unsafe to guess when evidence missing | drop sweep + `abstain_precision/recall`; require citations |
| Instruction injection exposure | prompts embedded in logs override truth | `instruction` profile; track `instr_override_rate` and `instr_gap` |
| Architecture eval (PaTH/RoPE-style) | test if state tracking improves under long context | `PaTH-style` steps curve + `same_key` selection wall |

Recommended presets (quick checks; replace `ModelPath` as needed):

- Authoritative change logs (selection under ambiguity, same_key):

```powershell
.\scripts\run_selector_bench.ps1 -Preset quick -ModelPath "C:\AI\models\your-model.gguf" -UseRerank
```

- Authority gating (NOTE vs UPDATE in kv_commentary):

```powershell
$env:GOLDEVIDENCEBENCH_RETRIEVAL_AUTHORITY_FILTER = "1"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_RERANK = "linear"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_K = "4"
$outDir = "runs\kv_commentary_authfilter_quick"
goldevidencebench sweep --out $outDir --seeds 1 --episodes 1 --steps 80 --queries 8 `
  --state-modes kv_commentary --distractor-profiles standard `
  --adapter goldevidencebench.adapters.retrieval_llama_cpp_adapter:create_adapter --no-derived-queries `
  --no-twins --require-citations --results-json "$outDir\combined.json" `
  --max-book-tokens 400 --note-rate 0.30
python .\scripts\summarize_results.py --in "$outDir\combined.json" --out-json "$outDir\summary.json"
```

- Instruction injection exposure:

```powershell
.\scripts\run_bench.ps1 -Preset standard -ModelPath "C:\AI\models\your-model.gguf" -RequireCitations
```

- Retrieval sanity (dense vs lexical):

```powershell
$env:GOLDEVIDENCEBENCH_RETRIEVAL_RETRIEVER = "dense"
$outDir = "runs\dense_kv_quick"
goldevidencebench sweep --out $outDir --seeds 1 --episodes 1 --steps 80 --queries 8 `
  --state-modes kv --distractor-profiles standard `
  --adapter goldevidencebench.adapters.retrieval_llama_cpp_adapter:create_adapter --no-derived-queries `
  --no-twins --require-citations --results-json "$outDir\combined.json" --max-book-tokens 400
python .\scripts\summarize_results.py --in "$outDir\combined.json" --out-json "$outDir\summary.json"
```

- Abstain calibration (missing gold):

```powershell
$env:GOLDEVIDENCEBENCH_RETRIEVAL_ABSTAIN_ON_MISSING = "1"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_DROP_SEED = "0"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_RERANK = "linear"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_K = "4"
foreach ($drop in 0.0,0.3) {
  $env:GOLDEVIDENCEBENCH_RETRIEVAL_DROP_PROB = "$drop"
  $outDir = "runs\abstain_drop${drop}_k4_quick"
  goldevidencebench sweep --out $outDir --seeds 1 --episodes 1 --steps 80 --queries 8 `
    --state-modes kv_commentary --distractor-profiles standard `
    --adapter goldevidencebench.adapters.retrieval_llama_cpp_adapter:create_adapter --no-derived-queries `
    --no-twins --require-citations --results-json "$outDir\combined.json" `
    --max-book-tokens 400 --note-rate 0.30
  python .\scripts\summarize_results.py --in "$outDir\combined.json" --out-json "$outDir\summary.json"
}
```

Threshold check (after running the presets above and `scripts/run_ui_same_label_stub.ps1`):

```powershell
python .\scripts\check_thresholds.py --config .\configs\usecase_checks.json
```

Release checklist (optional; runs UI same_label stub, canonical update_burst sweeps + thresholds):

```powershell
.\scripts\run_release_check.ps1 -ModelPath "C:\AI\models\your-model.gguf" -RunSweeps
```

UI release checklist (stub + wall sweep, optional config update):

```powershell
.\scripts\run_ui_release_check.ps1 -UpdateConfig
```

The config is intentionally small and uses `warn` vs `error` severity so you can tighten ceilings after you locate the drop-off wall for your regime.

Wall finder (example: wrong-update wall under update_burst):

```powershell
python .\scripts\find_wall.py --runs-dir .\runs `
  --metric retrieval.wrong_update_rate --param update_burst_rate `
  --threshold 0.10 --direction gte --state-mode kv --profile update_burst
```

Optional: write the ceiling back into `configs/usecase_checks.json` once the wall is known:

```powershell
python .\scripts\find_wall.py --runs-dir .\runs `
  --metric retrieval.wrong_update_rate --param update_burst_rate `
  --threshold 0.10 --direction gte --state-mode kv --profile update_burst `
  --update-config .\configs\usecase_checks.json --check-id YOUR_CHECK_ID
```

`YOUR_CHECK_ID` must already exist in the config; the tool will add or update the metric path with a max/min threshold based on `--direction`.

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
Optional linear tie-breaker: set `GOLDEVIDENCEBENCH_RETRIEVAL_LINEAR_TIE_BREAK=latest_step` (and optionally `GOLDEVIDENCEBENCH_RETRIEVAL_LINEAR_TIE_EPS`) to prefer newer steps when scores are close.
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

## Dense (hash) retriever baseline

This is a lightweight, dependency-free dense embedding baseline using a hashing projection over token vectors.
It is not a semantic model, but it provides a compact ?dense? retriever for comparison.

Run (kv, k=4, s3q16):

```powershell
$env:GOLDEVIDENCEBENCH_RETRIEVAL_RETRIEVER="dense"
goldevidencebench sweep --out runs\dense_kv_s3q16 --seeds 3 --episodes 1 --steps 240 --queries 16 `
  --state-modes kv --distractor-profiles standard `
  --adapter goldevidencebench.adapters.retrieval_llama_cpp_adapter:create_adapter --no-derived-queries `
  --no-twins --require-citations --results-json runs\dense_kv_s3q16\combined.json `
  --max-book-tokens 400
python .\scripts\summarize_results.py --in runs\dense_kv_s3q16\combined.json --out-json runs\dense_kv_s3q16\summary.json
```

Dense result (kv, k=4, s3q16): gold_present_rate 1.0000, selection_rate 0.3750, accuracy_when_gold_present 1.0000, value_acc 0.4167.
This shows dense retrieval can surface gold reliably, but selection still fails under ambiguity in this setting (selection_rate < 1).
Run: runs/dense_retriever_k4_s3q16.


Dense result (kv_commentary, k=4, s3q16): gold_present_rate 1.0000, selection_rate 0.4375, value_acc 0.4792.
Run: runs/dense_kv_commentary_k4_s3q16.
Interpretation: gold is present and NOTE is not selected, but selection still collapses under ambiguity; this remains selection-limited.


BM25 result (kv_commentary, k=4, s3q16): gold_present_rate 0.0625, selection_rate 1.0, value_acc 0.0625.
Run: runs/bm25_kv_commentary_k4_s3q16.

TF-IDF result (kv_commentary, k=4, s3q16): gold_present_rate 0.25, selection_rate 0.6667, value_acc 0.1667.
Run: runs/tfidf_kv_commentary_k4_s3q16.

Interpretation: lexical retrieval is the bottleneck in kv_commentary (gold rarely present); when gold is present, selection succeeds.


Cross-model check (Meta-Llama-3.1-8B, update_burst k=8 rate=0.34, s3q16, tie-break on):

| model | value_acc | selection_rate | gold_support_selected_rate | wrong_update_rate |
| --- | --- | --- | --- | --- |
| Meta-Llama-3.1-8B Q4_K_M | 0.1458 | 0.5000 | 0.5000 | 0.0000 |

Run: runs/update_burst_linear_tieeps002_k8_rate0.34_s3q16_llama31.

Interpretation: selection under ambiguity still dominates (gold_support_selected_rate=0.5), but the error mode differs from Qwen; the wall is model-dependent.


Cross-model check (fixed config, update_burst k=8 rate=0.34, s3q16, tie-break on):

| model | value_acc | selection_rate | gold_support_selected_rate | wrong_update_rate | answer_acc_given_gold_selected |
| --- | --- | --- | --- | --- | --- |
| Qwen 2.5 7B Q5_K_M | 0.1458 | 0.6000 | 0.6000 | 0.0000 | 1.0000 |
| Meta-Llama-3.1 8B Q4_K_M | 0.1458 | 0.7000 | 0.7000 | 0.0000 | 1.0000 |
| Mistral 7B Q4_K_M | 0.1875 | 0.8000 | 0.8000 | 0.0000 | 1.0000 |

Runs: runs/update_burst_linear_tieeps002_k8_rate0.34_s3q16_{qwen25,llama31,mistral7}.

Interpretation: extraction is perfect once gold is selected (answer_acc_given_gold_selected=1.0); selection remains the only bottleneck, and its severity is model-dependent.


Selector-only cross-model check (update_burst k=8 rate=0.34, s3q16):

| model | selection_rate | gold_support_selected_rate | wrong_update_rate |
| --- | --- | --- | --- |
| Qwen 2.5 7B Q5_K_M | 0.6000 | 0.6000 | 0.0000 |
| Meta-Llama-3.1 8B Q4_K_M | 0.8000 | 0.8000 | 0.0000 |
| Mistral 7B Q4_K_M | 0.7000 | 0.7000 | 0.0000 |

Runs: runs/update_burst_selonly_k8_rate0.34_s3q16_{qwen25,llama31,mistral7}.

Interpretation: selection remains the dominant failure even when the answerer is removed.



k-sweep (selector-only, Qwen, update_burst rate=0.34, s3q16):

| k | selection_rate | wrong_update_rate |
| --- | --- | --- |
| 8 | 0.6000 | 0.0000 |
| 12 | 0.5714 | 0.0000 |

Runs: runs/update_burst_selonly_k{8,12}_rate0.34_s3q16_qwen.

Interpretation: higher k slightly lowers selection_rate, but does not introduce wrong-UPDATE errors; ambiguity is still the dominant factor.



High-burst sweep (k=8, update_burst, s3q16; tie-break on):

| burst_rate | value_acc | selection_rate | wrong_update_rate |
| --- | --- | --- | --- |
| 0.45 | 0.3125 | 0.5200 | 0.1333 |
| 0.50 | 0.1875 | 0.3077 | 0.0000 |

Runs: runs/update_burst_linear_tieeps002_k8_rate0.45_s3q16, runs/update_burst_linear_tieeps002_k8_rate0.50_s3q16.

Interpretation: above the wall, selection collapses rapidly; rate=0.50 shows severe recall loss even without wrong-UPDATE picks.



High-burst sweep (k=4, update_burst, s3q16; tie-break on):

| burst_rate | value_acc | selection_rate | wrong_update_rate |
| --- | --- | --- | --- |
| 0.45 | 0.1875 | 0.7000 | 0.0000 |
| 0.50 | 0.1667 | 0.3636 | 0.0000 |

Runs: runs/update_burst_linear_tieeps002_k4_rate0.45_s3q16, runs/update_burst_linear_tieeps002_k4_rate0.50_s3q16.

Interpretation: lowering k helps at 0.45 but not at 0.50; ambiguity style dominates at the highest burst rate.



Recency-only baseline (k=8, update_burst rate=0.50, s3q16):

| selector | value_acc | selection_rate | wrong_update_rate |
| --- | --- | --- | --- |
| latest_step | 0.1875 | 0.3077 | 0.0000 |

Run: runs/update_burst_lateststep_k8_rate0.50_s3q16.

Interpretation: even a pure recency selector collapses at extreme burst rates; the ambiguity itself is too high in this regime.



Note camouflage (kv_commentary, k=4, s3q16, tie-break on):

| value_acc | gold_support_selected_rate | selected_note_rate | wrong_update_rate |
| --- | --- | --- | --- |
| 0.1667 | 0.6364 | 0.0000 | 0.0000 |

Run: runs/note_camouflage_linear_k4_s3q16.

Interpretation: NOTE camouflage does not cause NOTE selection (selected_note_rate=0), but gold selection drops?suggesting the camouflage mainly increases UPDATE ambiguity.



Note camouflage suite (kv_commentary, k=4, s3q16, tie-break on):

| value_acc | gold_support_selected_rate | selected_note_rate | wrong_update_rate |
| --- | --- | --- | --- |
| 0.1250 | 0.7500 | 0.0000 | 0.0000 |

Run: runs/note_camouflage_suite_linear_k4_s3q16.

Interpretation: the suite further lowers value accuracy without NOTE attraction; UPDATE ambiguity remains the dominant failure mode.



Note camouflage suite (kv_commentary, k=8, s3q16, tie-break on):

| value_acc | gold_support_selected_rate | selected_note_rate | wrong_update_rate |
| --- | --- | --- | --- |
| 0.3333 | 0.7500 | 0.0000 | 0.0000 |

Run: runs/note_camouflage_suite_linear_k8_s3q16.

Interpretation: increasing k does not reduce gold_support_selected_rate in this run; value accuracy remains low, pointing to answerer/extraction noise in this harder regime.



Note camouflage suite (k=8, no copy-clamp, s3q16):

| value_acc | gold_support_selected_rate | selected_note_rate | wrong_update_rate | value_is_substring_of_selected_line_rate |
| --- | --- | --- | --- | --- |
| 0.3125 | 0.6500 | 0.0000 | 0.0000 | 1.0000 |

Run: runs/note_camouflage_suite_linear_k8_noclamp_s3q16.

Interpretation: no-clamp does not improve value accuracy; substring rate is already 1.0, so the remaining errors are selection (gold_support_selected_rate < 1), not extraction.



Note camouflage suite (k=8, selector-only, s3q16):

| gold_present_rate | selection_rate | accuracy_when_gold_present | gold_support_selected_rate | selected_note_rate | wrong_update_rate |
| --- | --- | --- | --- | --- | --- |
| 0.4167 | 0.4000 | 0.1500 | 0.4000 | 0.0000 | 0.0000 |

Run: runs/note_camouflage_suite_selonly_k8_s3q16.

Interpretation: gold coverage is low (gold_present_rate 0.4167), and selection remains difficult even when gold is present. This isolates a retrieval/coverage bottleneck plus a selection bottleneck under camouflage, independent of answer generation.



Note camouflage suite (k=8, selector-only, clean, s3q16):

| gold_present_rate | selection_rate | accuracy_when_gold_present | gold_support_selected_rate | selected_note_rate | wrong_update_rate |
| --- | --- | --- | --- | --- | --- |
| 1.0000 | 0.5208 | 0.0625 | 0.5208 | 0.0000 | 0.4792 |

Run: runs/note_camouflage_suite_selonly_k8_clean_s3q16.

Interpretation: with drop disabled, gold is always present but selection falls to ~0.52 and wrong_update_rate rises to ~0.48. This isolates wrong-UPDATE selection (not NOTE attraction). In selector-only mode the value field is blank by design, so focus on selection_rate/gold_support_selected_rate rather than value_acc.



Note camouflage suite (k=8, selector-only, clean, s3q16) A/B:

| rerank | selection_rate | gold_support_selected_rate | wrong_update_rate |
| --- | --- | --- | --- |
| prefer_update_latest | 0.5000 | 0.5000 | 0.5000 |
| linear | 0.3333 | 0.3333 | 0.6667 |

Runs: runs/note_camouflage_suite_selonly_prefer_update_latest_k8_clean_s3q16, runs/note_camouflage_suite_selonly_linear_k8_clean_s3q16.

Interpretation: prefer_update_latest outperforms the learned linear selector under NOTE camouflage in this clean selector-only setting; wrong-UPDATE selection remains the dominant failure mode.



Note camouflage suite (k=4, selector-only, clean, s3q16) A/B:

| rerank | selection_rate | gold_support_selected_rate | wrong_update_rate |
| --- | --- | --- | --- |
| prefer_update_latest | 0.6250 | 0.6250 | 0.3750 |
| linear | 0.3958 | 0.3958 | 0.6042 |

Runs: runs/note_camouflage_suite_selonly_prefer_update_latest_k4_clean_s3q16, runs/note_camouflage_suite_selonly_linear_k4_clean_s3q16.

Interpretation: the deterministic prefer_update_latest selector continues to beat the learned linear selector at lower k, so the learned model has not closed the wrong-UPDATE gap under camouflage.



Note camouflage suite (k=4, selector-only, clean, s3q16) A/B with camo-trained linear selector:

| rerank | selection_rate | gold_support_selected_rate | wrong_update_rate |
| --- | --- | --- | --- |
| prefer_update_latest | 0.5625 | 0.5625 | 0.4375 |
| linear (note_camouflage_v2) | 0.5208 | 0.5208 | 0.4792 |

Runs: runs/note_camouflage_suite_selonly_prefer_update_latest_k4_clean_s3q16_camo_v2, runs/note_camouflage_suite_selonly_linear_k4_clean_s3q16_camo_v2.

Interpretation: camo-specific training lifts the linear selector vs the earlier k=4 run (selection_rate 0.3958 -> 0.5208), but it still trails prefer_update_latest and wrong-UPDATE selection remains the dominant error.



Note camouflage suite (k=8, selector-only, clean, s3q16) A/B with camo-trained linear selector:

| rerank | selection_rate | gold_support_selected_rate | wrong_update_rate |
| --- | --- | --- | --- |
| prefer_update_latest | 0.3750 | 0.3750 | 0.6250 |
| linear (note_camouflage_v2) | 0.3958 | 0.3958 | 0.6042 |

Runs: runs/note_camouflage_suite_selonly_prefer_update_latest_k8_clean_s3q16_camo_v2, runs/note_camouflage_suite_selonly_linear_k8_clean_s3q16_camo_v2.

Interpretation: at k=8 the camo-trained linear selector slightly exceeds prefer_update_latest on selection_rate, but wrong-UPDATE errors remain high for both; the regime is still dominated by same-key ambiguity.

Note: for camouflage selector-only sweeps, set `GOLDEVIDENCEBENCH_RETRIEVAL_WRONG_TYPE=same_key` to keep the regime hard. Leaving it at the default `none` makes selection trivial (gold_support_selected_rate can hit ~1.0), which is not comparable to the earlier results.



Note camouflage suite (k=8, selector-only, hard same_key, s3q16) grid:

| rerank | selection_rate | gold_support_selected_rate | selected_note_rate | wrong_update_rate |
| --- | --- | --- | --- | --- |
| latest_step | 1.0000 | 0.7292 | 0.2708 | 0.0000 |
| linear_tie_eps002 | 0.4583 | 0.1875 | 0.5625 | 0.2500 |
| prefer_set_latest | 0.4167 | 0.2500 | 0.4792 | 0.2708 |
| prefer_update_latest | 0.4167 | 0.1667 | 0.7083 | 0.1250 |
| linear_tie_eps005 | 0.3750 | 0.1458 | 0.6250 | 0.2292 |
| linear_base | 0.2708 | 0.0000 | 0.8125 | 0.1875 |

Runs: runs/note_camouflage_suite_selonly_latest_step_k8_hard_s3q16_auto, runs/note_camouflage_suite_selonly_linear_tie_eps002_k8_hard_s3q16_auto, runs/note_camouflage_suite_selonly_prefer_set_latest_k8_hard_s3q16_auto, runs/note_camouflage_suite_selonly_prefer_update_latest_k8_hard_s3q16_auto, runs/note_camouflage_suite_selonly_linear_tie_eps005_k8_hard_s3q16_auto, runs/note_camouflage_suite_selonly_linear_base_k8_hard_s3q16_auto.

Interpretation: with same_key ambiguity forced, note camouflage overwhelms learned selectors (high selected_note_rate) and even prefer_update_latest. latest_step avoids wrong-update picks but still selects NOTE ~27% of the time, so authority gating remains necessary in this regime.



Note camouflage suite (k=8, selector-only, hard same_key, authority filter ON, s3q16):

| rerank | selection_rate | gold_support_selected_rate | selected_note_rate | wrong_update_rate |
| --- | --- | --- | --- | --- |
| linear | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| prefer_update_latest | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| latest_step | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

Runs: runs/note_camouflage_suite_selonly_linear_k8_hard_authfilter_s3q16, runs/note_camouflage_suite_selonly_prefer_update_latest_k8_hard_authfilter_s3q16, runs/note_camouflage_suite_selonly_latest_step_k8_hard_authfilter_s3q16.

Interpretation: the authority filter eliminates NOTE attraction entirely; selection becomes perfect across rerankers in this hard regime. The remaining bottleneck only appears when the authority signal is untrusted or absent.



Authority spoofing (rate=0.5, k=8, hard same_key, selector-only, s3q16):

| authority_filter | selection_rate | gold_support_selected_rate | selected_note_rate | wrong_update_rate | spoof_accept_rate |
| --- | --- | --- | --- | --- | --- |
| off | 0.3958 | 0.2083 | 0.3333 | 0.4583 | 0.3333 |
| on  | 0.4375 | 0.4375 | 0.0000 | 0.5625 | 0.0000 |

Runs: runs/authority_spoof_rate0.5_filter0_linear_k8_hard_s3q16, runs/authority_spoof_rate0.5_filter1_linear_k8_hard_s3q16.

Interpretation: the authority filter eliminates spoof acceptance and NOTE picks, but shifts errors to wrong-UPDATE selection. With spoofing present, the gate is still necessary to prevent non-authoritative updates from being chosen.



Authority spoof sweep (linear, k=8, hard same_key, selector-only, s3q16):

| spoof_rate | authority_filter | selection_rate | gold_support_selected_rate | selected_note_rate | wrong_update_rate | spoof_accept_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | off | 0.6667 | 0.9375 | 0.0000 | 0.0625 | 0.0000 |
| 0.1 | on  | 0.9375 | 0.9375 | 0.0000 | 0.0625 | 0.0000 |
| 0.3 | off | 0.5208 | 0.5417 | 0.1042 | 0.3542 | 0.1042 |
| 0.3 | on  | 0.8125 | 0.8125 | 0.0000 | 0.1875 | 0.0000 |
| 0.5 | off | 0.5625 | 0.6042 | 0.1250 | 0.2708 | 0.1250 |
| 0.5 | on  | 0.5625 | 0.5625 | 0.0000 | 0.4375 | 0.0000 |

Runs: runs/authority_spoof_rate0.1_filter0_linear_k8_hard_s3q16, runs/authority_spoof_rate0.1_filter1_linear_k8_hard_s3q16, runs/authority_spoof_rate0.3_filter0_linear_k8_hard_s3q16, runs/authority_spoof_rate0.3_filter1_linear_k8_hard_s3q16, runs/authority_spoof_rate0.5_filter0_linear_k8_hard_s3q16, runs/authority_spoof_rate0.5_filter1_linear_k8_hard_s3q16.

Interpretation: the gate removes spoof acceptance at all rates. At moderate spoof (0.3), the filter lifts selection substantially while also cutting wrong_update_rate. At high spoof (0.5), the filter still stops spoof/NOTE picks but wrong-UPDATE selection dominates, so the bottleneck shifts to UPDATE disambiguation.



Authority spoof (rate=0.5, filter ON, spoof-trained linear selector, k=8, hard same_key, s3q16):

| selection_rate | gold_support_selected_rate | selected_note_rate | wrong_update_rate | spoof_accept_rate | spoof_accept_rate_non_gold |
| --- | --- | --- | --- | --- | --- |
| 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.6875 | 0.0000 |

Run: runs/authority_spoof_rate0.5_filter1_linear_spoof_v1_k8_hard_s3q16.

Interpretation: spoof-trained selection eliminates wrong-UPDATE errors under the gate, but spoof_accept_rate remains high because spoofing can hit gold lines. The non-gold spoof accept rate is still 0.0, so the selector is not choosing wrong spoofed lines. This does not replace the gate; it shows the selector can remain correct even when spoof tags are present.



Authority spoof (rate=0.5, filter OFF, spoof-trained linear, k=8, hard same_key, s3q16):

| selection_rate | gold_support_selected_rate | selected_note_rate | wrong_update_rate | spoof_accept_rate | spoof_accept_rate_non_gold |
| --- | --- | --- | --- | --- | --- |
| 1.0000 | 0.7292 | 0.2708 | 0.0000 | 0.5625 | 0.0000 |

Run: runs/authority_spoof_rate0.5_filter0_linear_spoof_v1_k8_hard_s3q16.

Interpretation: without the gate, the spoof-trained selector still picks NOTE/spoofed lines ~27% of the time. It does not replace the authority filter.





Authority spoof (rate=0.5, filter ON) tie-break A/B (linear, k=8, hard same_key, s3q16):

| linear_tie_break | selection_rate | gold_support_selected_rate | wrong_update_rate |
| --- | --- | --- | --- |
| none | 0.4375 | 0.4375 | 0.5625 |
| latest_step (eps=0.02) | 0.5625 | 0.5625 | 0.4375 |

Runs: runs/authority_spoof_rate0.5_filter1_linear_tie_none_k8_hard_s3q16, runs/authority_spoof_rate0.5_filter1_linear_tie_latest_step_k8_hard_s3q16.

Interpretation: adding a recency tie-break improves selection and reduces wrong-UPDATE errors under spoofing with the gate on, but the regime is still far from robust.



Authority spoof (filter ON) tie-break sweep (linear, k=8, hard same_key, s3q16):

| spoof_rate | tie_break | selection_rate | gold_support_selected_rate | wrong_update_rate |
| --- | --- | --- | --- | --- |
| 0.3 | none | 0.6250 | 0.6250 | 0.3750 |
| 0.3 | latest_step | 0.5625 | 0.5625 | 0.4375 |
| 0.5 | none | 0.5000 | 0.5000 | 0.5000 |
| 0.5 | latest_step | 0.5000 | 0.5000 | 0.5000 |

Runs: runs/authority_spoof_rate0.3_filter1_linear_tie_none_k8_hard_s3q16, runs/authority_spoof_rate0.3_filter1_linear_tie_latest_step_k8_hard_s3q16, runs/authority_spoof_rate0.5_filter1_linear_tie_none_k8_hard_s3q16, runs/authority_spoof_rate0.5_filter1_linear_tie_latest_step_k8_hard_s3q16.

Interpretation: the tie-break does not consistently help across spoof rates; at 0.3 it slightly hurts, and at 0.5 it makes no difference. This suggests the remaining error is not just recency ties but deeper UPDATE ambiguity under spoofing.



























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
- `distractor_profile=instruction` (adds spec-violating instructions); `instruction_suite` adds quoted/format variants; `adversarial` adds stale-echo distractors; `note_camouflage` makes NOTE lines look like updates (suite adds quoted-update variants); `update_burst` injects rapid same-key UPDATE bursts with near-miss values
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
To find the ambiguity wall when recency is less explicit, set `GOLDEVIDENCEBENCH_RETRIEVAL_STEP_BUCKET`
(e.g., 5 or 10) to coarsen step numbers. Train and evaluate with the same bucket value.

Recency-coarsening wall (kv, k=8, s3q16; `runs/kv_wrong_update_linear_bucket*_k8_s3q16`):

| step_bucket | selection_rate | wrong_update_rate | value_acc |
| --- | --- | --- | --- |
| 1 | 1.0000 | 0.0000 | 1.0000 |
| 5 | 0.8125 | 0.1875 | 0.8125 |
| 10 | 0.8750 | 0.1250 | 0.8750 |
| 20 | 0.5417 | 0.4583 | 0.5417 |

Interpretation: once step information is coarsened (bucket >=5), wrong-UPDATE selection starts; by bucket=20 it dominates.

Step-length sensitivity (bucket=2, k=16, keys=4; `runs/kv_wrong_update_bucket2_k16_keys4_steps*_s3q16`):

| steps | selection_rate | wrong_update_rate | value_acc |
| --- | --- | --- | --- |
| 120 | 0.9583 | 0.0417 | 0.9583 |
| 240 | 0.8333 | 0.1667 | 0.8333 |

Interpretation: with the same bucketing, longer sequences amplify ambiguity and wrong-UPDATE picks.

Automated update_burst wall search (staged sweep + find_wall):

```powershell
.\scripts\run_update_burst_wall.ps1 -ModelPath "C:\AI\models\your-model.gguf"
```

Outputs go to a timestamped subfolder under `runs\wall_update_burst_YYYYMMDD_HHMMSS` by default. Optional knobs:
`-RunsDir "runs\wall_update_burst_custom"`, `-StartStage 2` (skip full pipeline), `-Threshold 0.10`,
`-StopAfterWall:$false`, `-LinearModel ".\models\linear_selector.json"`, `-AutoPin:$true`,
`-PinCount 4`, `-PinDecimals 3`.

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
