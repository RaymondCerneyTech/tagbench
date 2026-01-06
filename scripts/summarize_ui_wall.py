from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _parse_duplicates(name: str) -> int | None:
    if not name.startswith("dups"):
        return None
    suffix = name[4:]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _row_from_run(run_dir: Path) -> dict[str, Any] | None:
    config = _load_json(run_dir / "config.json") or {}
    score = _load_json(run_dir / "score.json") or {}
    summary = _load_json(run_dir / "summary.json") or {}

    duplicates = config.get("duplicates")
    if duplicates is None:
        duplicates = _parse_duplicates(run_dir.name)
    if duplicates is None:
        return None

    metrics = score.get("metrics", {}) if isinstance(score.get("metrics"), dict) else {}
    summary_metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}

    return {
        "run": run_dir.name,
        "duplicates": duplicates,
        "steps": config.get("steps"),
        "seed": config.get("seed"),
        "labels": config.get("labels"),
        "adapter": config.get("adapter"),
        "selection_mode": config.get("selection_mode"),
        "selection_seed": config.get("selection_seed"),
        "selection_rate": metrics.get("selection_rate"),
        "wrong_action_rate": metrics.get("wrong_action_rate"),
        "post_action_verify_rate": metrics.get("post_action_verify_rate"),
        "avg_candidates_per_row": summary_metrics.get("avg_candidates_per_row"),
        "avg_max_label_duplicates_per_row": summary_metrics.get("avg_max_label_duplicates_per_row"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        values = []
        for key in headers:
            value = row.get(key)
            if value is None:
                values.append("")
            else:
                text = str(value)
                text = text.replace('"', '""')
                if "," in text or "\n" in text:
                    text = f'"{text}"'
                values.append(text)
        lines.append(",".join(values))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize UI same_label wall sweeps.")
    parser.add_argument("--runs-dir", type=Path, required=True, help="Root directory for UI wall runs.")
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    parser.add_argument("--out-csv", type=Path, help="Optional CSV output path.")
    args = parser.parse_args()

    runs_dir = args.runs_dir
    if not runs_dir.exists():
        print(f"Runs dir not found: {runs_dir}")
        return 1

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        row = _row_from_run(run_dir)
        if row is not None:
            rows.append(row)

    rows = sorted(rows, key=lambda r: (r.get("duplicates") or 0))
    payload = {"runs": len(rows), "rows": rows}

    if args.out:
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.out_csv:
        _write_csv(args.out_csv, rows)

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
