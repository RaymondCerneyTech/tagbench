"""Inspect selector dataset for NOTE-vs-UPDATE confusions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _norm_op(op: Any) -> str:
    return str(op or "").strip().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize NOTE vs UPDATE ambiguity in selector datasets.")
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()

    total = 0
    gold_present = 0
    gold_is_note = 0
    gold_not_top = 0
    top_has_note = 0
    top_note_gold_update = 0
    note_candidates = 0
    all_candidates = 0

    for line in args.data.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        candidates = row.get("candidates") or []
        if not candidates:
            continue
        total += 1

        uid_to_op = {c.get("uid"): _norm_op(c.get("op")) for c in candidates if c.get("uid")}
        gold_uid = row.get("gold_support_uid") or row.get("correct_uid")
        gold_op = uid_to_op.get(gold_uid)
        if gold_uid and gold_uid in uid_to_op:
            gold_present += 1
        if gold_op == "NOTE":
            gold_is_note += 1

        max_step = max(int(c.get("step", 0)) for c in candidates)
        top = [c for c in candidates if int(c.get("step", 0)) == max_step]
        top_uids = {c.get("uid") for c in top}
        if gold_uid and gold_uid in uid_to_op and gold_uid not in top_uids:
            gold_not_top += 1

        top_note = any(_norm_op(c.get("op")) == "NOTE" for c in top)
        if top_note:
            top_has_note += 1
            if gold_op and gold_op != "NOTE":
                top_note_gold_update += 1

        note_candidates += sum(1 for c in candidates if _norm_op(c.get("op")) == "NOTE")
        all_candidates += len(candidates)

    def rate(n: int, d: int) -> float:
        return n / d if d else 0.0

    print(f"total_examples: {total}")
    print(f"gold_present_rate: {rate(gold_present, total):.3f}")
    print(f"gold_is_note_rate: {rate(gold_is_note, total):.3f}")
    print(f"gold_not_top_rate: {rate(gold_not_top, total):.3f}")
    print(f"top_has_note_rate: {rate(top_has_note, total):.3f}")
    print(f"top_note_gold_update_rate: {rate(top_note_gold_update, total):.3f}")
    print(f"note_candidate_rate: {rate(note_candidates, all_candidates):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
