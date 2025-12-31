import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        type=Path,
        required=True,
        help="Run directories containing combined.json.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output image path.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="value_acc",
        help="Metric key from metrics dict to plot.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Compute vs quality",
        help="Plot title.",
    )
    return parser.parse_args()


def _load_row(run_dir: Path, metric: str) -> tuple[str, float, float]:
    combined = run_dir / "combined.json"
    if not combined.exists():
        raise SystemExit(f"Missing {combined}")
    data = json.loads(combined.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit(f"Unexpected combined.json format in {combined}")
    row = data[0]
    eff = row.get("efficiency", {}) or {}
    metrics = row.get("metrics", {}) or {}
    tokens_per_q = float(eff.get("tokens_per_q", 0.0) or 0.0)
    value = float(metrics.get(metric, 0.0) or 0.0)
    label = run_dir.name
    return label, tokens_per_q, value


def main() -> int:
    args = parse_args()
    labels: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    for run_dir in args.runs:
        label, tokens_per_q, value = _load_row(run_dir, args.metric)
        labels.append(label)
        xs.append(tokens_per_q)
        ys.append(value)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.scatter(xs, ys, color="#3b6ea8")
    for label, x, y in zip(labels, xs, ys):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("tokens per query")
    ax.set_ylabel(args.metric)
    ax.set_title(args.title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
