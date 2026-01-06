from __future__ import annotations

import argparse
from pathlib import Path

from goldevidencebench.ui_generate import generate_popup_overlay_fixture, generate_same_label_fixture
from goldevidencebench.util import write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a UI fixture JSONL.")
    parser.add_argument("--out", default="data/ui_same_label_generated.jsonl")
    parser.add_argument(
        "--profile",
        choices=["same_label", "popup_overlay"],
        default="same_label",
        help="Fixture profile to generate.",
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--duplicates", type=int, default=2)
    parser.add_argument("--overlay-duplicates", type=int, default=1)
    parser.add_argument(
        "--labels",
        default="Next,Continue,Save",
        help="Comma-separated labels to sample from.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--app-path-prefix", default="UI Flow")
    args = parser.parse_args()

    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if args.profile == "popup_overlay":
        rows = generate_popup_overlay_fixture(
            steps=args.steps,
            base_duplicates=args.duplicates,
            overlay_duplicates=args.overlay_duplicates,
            labels=labels,
            seed=args.seed,
            app_path_prefix=args.app_path_prefix,
        )
    else:
        rows = generate_same_label_fixture(
            steps=args.steps,
            duplicates=args.duplicates,
            labels=labels,
            seed=args.seed,
            app_path_prefix=args.app_path_prefix,
        )
    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
