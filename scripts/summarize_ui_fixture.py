from __future__ import annotations

import argparse
import json
from pathlib import Path

from goldevidencebench.ui_fixture import validate_ui_fixture_path
from goldevidencebench.ui_summary import summarize_ui_rows
from goldevidencebench.util import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize UI fixture rows.")
    parser.add_argument(
        "--fixture",
        default="data/ui_same_label_fixture.jsonl",
        help="Path to the UI fixture JSONL file.",
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
    metrics = summarize_ui_rows(rows)
    payload = {"rows": len(rows), "metrics": metrics}

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
