from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UiWallPoint:
    run_dir: Path
    param: float
    metric: float


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _get_path(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _parse_duplicates(name: str) -> float | None:
    if not name.startswith("dups"):
        return None
    suffix = name[4:]
    if not suffix.isdigit():
        return None
    return float(int(suffix))


def _load_points(runs_dir: Path, metric_path: str, score_name: str) -> list[UiWallPoint]:
    points: list[UiWallPoint] = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        score_path = run_dir / score_name
        score = _load_json(score_path)
        if not score:
            continue
        metric = _as_float(_get_path(score, metric_path))
        if metric is None:
            continue
        config = _load_json(run_dir / "config.json") or {}
        param = _as_float(config.get("duplicates"))
        if param is None:
            param = _parse_duplicates(run_dir.name)
        if param is None:
            continue
        points.append(UiWallPoint(run_dir=run_dir, param=param, metric=metric))
    return points


def _find_wall(points: list[UiWallPoint], threshold: float, direction: str) -> tuple[UiWallPoint | None, UiWallPoint | None]:
    ordered = sorted(points, key=lambda p: p.param)
    last_ok: UiWallPoint | None = None
    for point in ordered:
        if direction == "gte" and point.metric >= threshold:
            return last_ok, point
        if direction == "lte" and point.metric <= threshold:
            return last_ok, point
        last_ok = point
    return last_ok, None


def _format_points(points: list[UiWallPoint]) -> str:
    lines = ["duplicates,metric,run_dir"]
    for point in points:
        lines.append(f"{point.param:.4f},{point.metric:.4f},{point.run_dir}")
    return "\n".join(lines)


def _relative_path(path: Path) -> str:
    try:
        rel = path.relative_to(Path.cwd())
    except ValueError:
        rel = path
    return rel.as_posix()


def _update_config(
    *,
    config_path: Path,
    check_id: str,
    summary_path: Path,
    metric_path: str,
    threshold: float,
    direction: str,
    severity: str,
) -> None:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    checks = data.setdefault("checks", [])
    check = next((c for c in checks if str(c.get("id")) == check_id), None)
    if check is None:
        check = {"id": check_id, "metrics": []}
        checks.append(check)
    check.setdefault("severity", severity)
    check["summary_path"] = _relative_path(summary_path)
    metrics = check.setdefault("metrics", [])
    entry = next((m for m in metrics if str(m.get("path")) == metric_path), None)
    if entry is None:
        entry = {"path": metric_path}
        metrics.append(entry)
    if direction == "gte":
        entry["max"] = float(threshold)
    else:
        entry["min"] = float(threshold)
    config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Find a UI same_label wall from UI wall sweeps.")
    parser.add_argument("--runs-dir", type=Path, required=True, help="Root directory for UI wall runs.")
    parser.add_argument("--metric", dest="metric_path", default="metrics.wrong_action_rate")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--direction", choices=["gte", "lte"], required=True)
    parser.add_argument("--score-name", default="score.json", help="Score JSON filename inside each run dir.")
    parser.add_argument("--use-wall", action="store_true", help="Update config to wall point instead of last_ok.")
    parser.add_argument("--update-config", type=Path, default=None)
    parser.add_argument("--check-id", type=str, default=None)
    parser.add_argument("--severity", type=str, default="warn")
    args = parser.parse_args()

    points = _load_points(args.runs_dir, args.metric_path, args.score_name)
    if not points:
        print("No matching UI wall runs found.")
        return 1
    print(_format_points(points))

    last_ok, wall = _find_wall(points, threshold=args.threshold, direction=args.direction)
    if wall:
        print(f"wall_param={wall.param:.4f} wall_metric={wall.metric:.4f} run_dir={wall.run_dir}")
        if last_ok:
            print(f"last_ok_param={last_ok.param:.4f} last_ok_metric={last_ok.metric:.4f} run_dir={last_ok.run_dir}")
    else:
        print("wall_param=None")
        if last_ok:
            print(f"last_ok_param={last_ok.param:.4f} last_ok_metric={last_ok.metric:.4f} run_dir={last_ok.run_dir}")

    if args.update_config:
        if not args.check_id:
            raise SystemExit("--check-id is required with --update-config")
        chosen = wall if args.use_wall else last_ok
        if chosen is None:
            print("No point available to update config.")
            return 1
        _update_config(
            config_path=args.update_config,
            check_id=args.check_id,
            summary_path=chosen.run_dir / args.score_name,
            metric_path=args.metric_path,
            threshold=args.threshold,
            direction=args.direction,
            severity=args.severity,
        )
        print(f"Updated {args.update_config} ({args.check_id} {args.metric_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
