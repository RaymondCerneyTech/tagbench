from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from goldevidencebench.ui_eval import (
    score_post_action_verification,
    score_ui_rows,
    score_ui_sequences,
)
from goldevidencebench.ui_fixture import validate_ui_fixture_path
from goldevidencebench.util import read_jsonl


def _select_candidate_ids(rows: list[dict[str, Any]], mode: str, seed: int) -> list[str | None]:
    rng = random.Random(seed)
    selected_ids: list[str | None] = []
    for row in rows:
        candidates = row.get("candidates", [])
        candidate_ids = [
            c.get("candidate_id")
            for c in candidates
            if isinstance(c, dict) and isinstance(c.get("candidate_id"), str)
        ]
        gold = row.get("gold", {})
        gold_id = gold.get("candidate_id") if isinstance(gold, dict) else None

        if mode == "gold":
            selected_ids.append(gold_id if isinstance(gold_id, str) else None)
        elif mode == "first":
            selected_ids.append(candidate_ids[0] if candidate_ids else None)
        elif mode == "random":
            selected_ids.append(rng.choice(candidate_ids) if candidate_ids else None)
        elif mode == "abstain":
            selected_ids.append(None)
        else:
            raise ValueError(f"unknown mode: {mode}")
    return selected_ids


def _load_observed_deltas(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    observed_rows = list(read_jsonl(path))
    observed_by_id = {}
    for row in observed_rows:
        if not isinstance(row, dict):
            continue
        row_id = row.get("id")
        observed = row.get("observed_delta")
        if isinstance(row_id, str):
            observed_by_id[row_id] = observed
    observed_deltas: list[dict[str, Any] | None] = []
    for row in rows:
        row_id = row.get("id")
        observed_deltas.append(observed_by_id.get(row_id))
    return observed_deltas


def main() -> int:
    parser = argparse.ArgumentParser(description="Score UI fixture selections.")
    parser.add_argument(
        "--fixture",
        default="data/ui_same_label_fixture.jsonl",
        help="Path to the UI fixture JSONL file.",
    )
    parser.add_argument(
        "--mode",
        choices=("gold", "first", "random", "abstain"),
        default="first",
        help="Selection mode for scoring.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for mode=random.")
    parser.add_argument("--out", help="Optional output JSON path.")
    parser.add_argument(
        "--observed",
        help="Optional JSONL with {id, observed_delta} rows for post-action verification.",
    )
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        print(f"Fixture not found: {fixture_path}")
        return 1

    errors = validate_ui_fixture_path(fixture_path)
    if errors:
        print("Fixture validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    rows = list(read_jsonl(fixture_path))
    selected_ids = _select_candidate_ids(rows, args.mode, args.seed)
    metrics = score_ui_rows(rows, selected_ids)
    observed_deltas = None
    if args.observed:
        observed_path = Path(args.observed)
        if not observed_path.exists():
            print(f"Observed deltas not found: {observed_path}")
            return 1
        observed_deltas = _load_observed_deltas(observed_path, rows)
        metrics.update(score_post_action_verification(rows, observed_deltas))
    sequence_metrics = score_ui_sequences(rows, selected_ids, observed_deltas)
    payload = {
        "rows": len(rows),
        "mode": args.mode,
        "seed": args.seed,
        "metrics": metrics,
        "sequence_metrics": sequence_metrics,
    }

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
