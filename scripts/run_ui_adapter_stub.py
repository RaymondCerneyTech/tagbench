from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from goldevidencebench.ui_eval import (
    score_post_action_verification,
    score_ui_rows,
    score_ui_sequences,
)
from goldevidencebench.ui_fixture import validate_ui_fixture_path
from goldevidencebench.util import read_jsonl


def _load_adapter(path: str):
    if ":" not in path:
        raise ValueError("Adapter must be in module:function format.")
    module_name, fn_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, fn_name)
    return factory()


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
    parser = argparse.ArgumentParser(description="Run a UI adapter stub on the UI fixture.")
    parser.add_argument(
        "--fixture",
        default="data/ui_same_label_fixture.jsonl",
        help="Path to the UI fixture JSONL file.",
    )
    parser.add_argument(
        "--adapter",
        default="goldevidencebench.adapters.ui_fixture_adapter:create_adapter",
        help="Adapter path (module:function).",
    )
    parser.add_argument(
        "--observed",
        help="Optional JSONL with {id, observed_delta} rows for post-action verification.",
    )
    parser.add_argument("--out", help="Optional output JSON path.")
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
    adapter = _load_adapter(args.adapter)
    selected_ids: list[str | None] = []
    for row in rows:
        pred = adapter.predict(row, protocol="ui")
        selected_ids.append(pred.get("value"))

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
        "adapter": args.adapter,
        "metrics": metrics,
        "sequence_metrics": sequence_metrics,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
