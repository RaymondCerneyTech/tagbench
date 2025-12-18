# tagbench

tagbench is a benchmark + harness for long-context state tracking. It generates synthetic “episode logs” with evolving state (kv/counter/set/relational), distractors (including instruction injection), and queries that require answering from the latest state update. It evaluates open-book vs closed-book protocols, enforces citation support IDs with capped-k + F1, checks entailment-from-citations, and uses counterfactual twin episodes to detect shortcut heuristics. It also reports efficiency (tokens/query, passes, wall time) so you can measure capability per compute.

`tagbench` is a small benchmark + reference codebase for testing whether an LLM can track evolving state across long documents by reading its own “book” artifacts:

- **Chapters**: narrative text (contains distractors and stale summaries)
- **Glossary (tags)**: a lightweight key reference
- **State ledger**: the authoritative state updates (with support IDs)

The benchmark generates synthetic episodes (updates + distractors + queries), grades model answers (optionally requiring citations/support IDs), and includes baselines (naive scan vs ledger-based reader).

## Defaults (chosen here)

These are the CLI defaults (picked to create long-ish documents with frequent distractors):

- `episodes=20`, `steps=220`, `keys=14`, `queries=12`, `chapters=8`, `twins=true`
- `distractor_rate=0.50`, `clear_rate=0.08`
- `distractor_profile=instruction` (adds spec-violating instructions); `adversarial` adds stale-echo distractors
- `state_mode=kv` (switch to `counter`, `set`, or `relational`)
- `require_citations=true` (questions ask for JSON `{value, support_ids}` with max 3)
- Closed-book is the headline score (`tagbench run` defaults to `--protocol closed_book`; open-book is diagnostic)

## Why this benchmark

This benchmark isolates a specific long-context failure mode: **state changes over time** (updates + clears), embedded in a long document with misleading restatements. It’s motivated by reports that transformer LLMs can struggle with consistent state tracking across long sequences (see: MIT News, 2025-12-17).

## Install

Python 3.12 is assumed.

```powershell
python -m pip install -e .
```

## Generate a dataset

Writes JSONL with one row per query.

```powershell
tagbench generate --out .\data\tagbench.jsonl --seed 0
```

Each row looks like:

- `id`: query ID
- `document`: the raw **episode log** (updates + distractors)
- `book`: a derived “book” artifact (chapters + glossary + ledger) for convenience/baselines
- `question`: the question text (and output format requirements if citations are enabled)
- `gold`: `{value, support_ids}` where `support_ids` contains the authoritative UPDATE ID that establishes the current value
- `meta`: includes `requires_citation` and the queried `key`
- `schema_version`: `"0.1"`
- `state_mode`: `kv|counter|set|relational`

State dynamics:

- `kv` (default): standard key→value overwrites
- `counter`: numeric accumulators (increments)
- `set`: membership add/remove (values are comma-separated lists)
- `relational`: reassignment tasks (e.g., who reports to whom)

## Run baselines

```powershell
tagbench run --data .\data\tagbench.jsonl --baseline naive
tagbench run --data .\data\tagbench.jsonl --baseline ledger
```

Protocols (headline = closed_book):

- `open_book`: baselines read the raw `document` (episode log)
- `closed_book`: baselines read only the derived `book` artifact

`tagbench run` defaults to `--protocol closed_book` (pass `--protocol both` for diagnostics).

Metrics:

- `value_acc`: predicted `value` matches gold
- `cite_f1`: support-ID F1 (only when citations are required; capped at `max_support_k`)
- `entailment`: fraction where the answer is justified by the cited updates only
- `exact_acc`: value match + (if required) support includes gold + entailment-from-citations
- `twin_consistency`: counterfactual twin agreement/disagreement rate (anti-shortcut)
- `twin_flip_rate`: twin pairs where the answer flips when the decisive UPDATE flips (higher is better)
- `instr_acc` / `instr_gap`: accuracy on questions with instruction-injection distractors, and the drop vs. clean questions
- Efficiency curve (printed by `tagbench run`): tokens read, tokens/query, passes over doc, wall-clock seconds.

## Grade model outputs

Predictions JSONL can be either:

- `{ "id": "...", "value": "...", "support_ids": ["U0007"] }`, or
- `{ "id": "...", "output": "..." }` where `output` contains a JSON object (optionally embedded in text)

```powershell
tagbench grade --data .\data\tagbench.jsonl --pred .\preds.jsonl
```

## Plug in your model (adapter interface)

Implement a tiny adapter (module with `create_adapter()` returning an object that has `.predict(row, protocol="...") -> {"value": ..., "support_ids": [...]}`).

Reference adapter (wraps the ledger baseline): `tagbench.adapters.ledger_adapter:create_adapter`.
Two-phase adapter example (build book once, answer many): `tagbench.adapters.log_to_book_adapter:create_adapter` implements `build_artifact(document, episode_id, protocol)` then answers closed-book.
Closed-book Llama example (uses `llama-cpp-python`, set `TAGBENCH_MODEL` to a GGUF path): `tagbench.adapters.llama_cpp_adapter:create_adapter`.

Adapter contract (hard-validated):

- `value` required.
- `support_ids` must be a list (max 3 by default) and must reference UPDATE IDs in the episode (closed-book uses the book ledger).
- Extra fields are rejected; missing `value` fails fast.
- `adapter_schema_version=1.0` is attached to metrics outputs for compatibility.

Run your adapter:

```powershell
tagbench model --data .\data\tagbench.jsonl --adapter tagbench.adapters.ledger_adapter:create_adapter --protocol closed_book
```

Both `tagbench run` and `tagbench model` can emit machine-readable metrics via `--results-json` (JSON object or array; overwrites the file each run; intended for plotting accuracy vs tokens/passes). Closed-book with `--protocol both` writes an array.

Practical runs: create a folder per run, sweep multiple seeds/state_modes/distractor_profiles, and keep one `results.json` per run to compare stability (long-context behavior is seed/ordering-sensitive).

Sweeps: `tagbench sweep --out runs --seeds 5 --state-modes kv,counter,set,relational --distractor-profiles standard,adversarial,instruction` (writes one subfolder per combo with data/preds/results).

## Anti-cheat / robustness notes

- The episode log contains **UPDATE** lines (authoritative) and **DISTRACTOR** lines (untrusted).
- When citations are enabled, correct answers require returning **support IDs** (update IDs like `U0007`) and passing **entailment-from-citations**.
- By default, every episode also includes a **counterfactual twin** (one UPDATE is flipped); grading reports `twin_consistency` to detect shortcut heuristics.
- UPDATE IDs are non-monotonic (hash-like) to prevent “pick the max ID” shortcuts; ordering comes from the logged `step`.
- Closed-book protocol feeds only the derived book artifact (no episode log).
- `--distractor-profile instruction` (default) injects spec-violating instructions; `adversarial` adds stale-echo distractors (late repeats of old values).
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
What’s new: Tagbench now evaluates closed-book state tracking by default (answer using only a derived “book” artifact, not the raw episode log), adds richer state dynamics (kv|counter|set|relational), and includes adversarial distractors such as instruction injection and stale-echo repeats of outdated values. To reduce loopholes, UPDATE IDs are non-monotonic (hash-like) and each episode includes a counterfactual twin; grading reports both twin_consistency and twin_flip_rate. tagbench run also prints an efficiency curve (tokens read, tokens/query, passes, wall-clock) so accuracy can be compared against compute cost.
