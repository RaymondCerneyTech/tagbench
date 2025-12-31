from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def _pattern_from_name(name: str) -> str:
    name = re.sub(r"_s\d+q\d+$", "", name)
    name = re.sub(r"_s\d+$", "", name)
    return name


def _load_summary(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _row_from_summary(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    overall = data.get("overall", {})
    retrieval = data.get("retrieval", {})
    return {
        "run_dir": str(path.parent),
        "run_name": path.parent.name,
        "pattern": _pattern_from_name(path.parent.name),
        "mtime": path.stat().st_mtime,
        "rows": data.get("rows"),
        "value_acc": overall.get("value_acc_mean"),
        "exact_acc": overall.get("exact_acc_mean"),
        "cite_f1": overall.get("cite_f1_mean"),
        "entailment": overall.get("entailment_mean"),
        "gold_present_rate": retrieval.get("gold_present_rate"),
        "selection_rate": retrieval.get("selection_rate"),
        "accuracy_when_gold_present": retrieval.get("accuracy_when_gold_present"),
        "drop_rate": retrieval.get("drop_rate"),
        "decomposition_line": retrieval.get("decomposition_line"),
        "state_integrity_rate": overall.get("state_integrity_rate_mean"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect summary.json files into a single CSV.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out-csv", type=Path, default=Path("runs/summary_all.csv"))
    parser.add_argument("--latest-only", action="store_true", help="Keep only the newest run per pattern (suffix _sNqM stripped).")
    args = parser.parse_args()

    summaries = []
    for summary_path in args.runs_dir.rglob("summary.json"):
        data = _load_summary(summary_path)
        if not data:
            continue
        summaries.append(_row_from_summary(summary_path, data))

    if args.latest_only:
        latest: dict[str, dict[str, Any]] = {}
        for row in summaries:
            key = row.get("pattern") or row["run_name"]
            prev = latest.get(key)
            if prev is None or row["mtime"] > prev["mtime"]:
                latest[key] = row
        summaries = list(latest.values())

    if not summaries:
        args.out_csv.write_text("", encoding="utf-8")
        return 0

    fields = sorted(summaries[0].keys())
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
