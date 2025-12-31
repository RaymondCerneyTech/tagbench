from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from goldevidencebench.baselines import parse_book_ledger
from goldevidencebench.adapters.retrieval_llama_cpp_adapter import (
    _apply_drop_with_rng,
    _apply_order,
    _select_entries_for_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--wrong-type", type=str, default="same_key")
    parser.add_argument("--order", type=str, default="shuffle")
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--drop-prob", type=float, default=0.0)
    parser.add_argument("--drop-seed", type=int, default=0)
    parser.add_argument("--include-clear", action="store_true")
    parser.add_argument("--authoritative-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _filter_entries(
    entries: list[dict[str, object]], include_clear: bool, authoritative_only: bool
) -> list[dict[str, object]]:
    filtered = entries
    if not include_clear:
        filtered = [entry for entry in filtered if entry.get("op") != "CLEAR"]
    if authoritative_only:
        filtered = [entry for entry in filtered if entry.get("op") != "NOTE"]
    return filtered


def main() -> int:
    args = parse_args()
    rows_written = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as out:
        for line in args.data.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if args.limit and rows_written >= args.limit:
                break
            meta = row.get("meta", {}) or {}
            if meta.get("query_type") not in {None, "direct"}:
                continue
            key = meta.get("key")
            if not key:
                continue
            book = row.get("book") or row.get("artifact")
            if not book:
                continue
            entries = parse_book_ledger(book)
            entries = _filter_entries(entries, args.include_clear, args.authoritative_only)
            if not entries:
                continue
            selected, diag, wrong_entry = _select_entries_for_key(
                entries=entries, key=key, k=max(1, args.k), wrong_type=args.wrong_type
            )
            rng = random.Random(args.drop_seed ^ hash(row.get("id", "")))
            selected, dropped = _apply_drop_with_rng(
                selected=selected,
                correct_uid=diag.get("correct_uid"),
                wrong_entry=wrong_entry,
                drop_prob=max(0.0, min(1.0, args.drop_prob)),
                rng=rng,
            )
            order_applied = None
            if args.order == "shuffle" and len(selected) > 1:
                shuffle_rng = random.Random(args.order_seed ^ hash(row.get("id", "")))
                shuffle_rng.shuffle(selected)
                order_applied = "shuffle"
            elif args.order in {"gold_first", "gold_middle", "gold_last"} and selected:
                selected, order_applied = _apply_order(
                    selected=selected,
                    correct_uid=diag.get("correct_uid"),
                    order=args.order,
                )

            record = {
                "id": row.get("id"),
                "episode_id": row.get("episode_id"),
                "question": row.get("question"),
                "key": key,
                "k": args.k,
                "wrong_type": args.wrong_type,
                "drop_prob": args.drop_prob,
                "order": order_applied,
                "correct_uid": diag.get("correct_uid"),
                "gold_present": diag.get("correct_included") and not dropped,
                "candidates": [
                    {
                        "uid": entry.get("uid"),
                        "step": entry.get("step"),
                        "op": entry.get("op"),
                        "key": entry.get("key"),
                        "value": entry.get("value"),
                    }
                    for entry in selected
                ],
            }
            out.write(json.dumps(record) + "\n")
            rows_written += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
