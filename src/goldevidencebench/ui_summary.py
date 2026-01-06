from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def summarize_ui_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows_list = list(rows)
    row_count = len(rows_list)
    if row_count == 0:
        return {
            "rows": 0,
            "candidates_total": 0,
            "avg_candidates_per_row": 0.0,
            "label_counts": {},
            "unique_labels": 0,
            "expected_delta_present_rate": 0.0,
            "unique_candidate_id_rate": 0.0,
            "max_label_duplicates_per_row": 0,
            "avg_max_label_duplicates_per_row": 0.0,
        }

    candidates_total = 0
    expected_delta_present = 0
    unique_candidate_rows = 0
    label_counts: Counter[str] = Counter()
    max_label_duplicates: list[int] = []

    for row in rows_list:
        candidates = row.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        candidates_total += len(candidates)

        labels = [
            candidate.get("label")
            for candidate in candidates
            if isinstance(candidate, dict) and isinstance(candidate.get("label"), str)
        ]
        label_counts.update(labels)
        if labels:
            max_label_duplicates.append(max(Counter(labels).values()))
        else:
            max_label_duplicates.append(0)

        candidate_ids = [
            candidate.get("candidate_id")
            for candidate in candidates
            if isinstance(candidate, dict) and isinstance(candidate.get("candidate_id"), str)
        ]
        if len(candidate_ids) == len(set(candidate_ids)):
            unique_candidate_rows += 1

        if isinstance(row.get("expected_delta"), dict):
            expected_delta_present += 1

    avg_candidates_per_row = candidates_total / row_count
    expected_delta_present_rate = expected_delta_present / row_count
    unique_candidate_id_rate = unique_candidate_rows / row_count
    max_label_duplicates_per_row = max(max_label_duplicates) if max_label_duplicates else 0
    avg_max_label_duplicates_per_row = sum(max_label_duplicates) / row_count

    return {
        "rows": row_count,
        "candidates_total": candidates_total,
        "avg_candidates_per_row": avg_candidates_per_row,
        "label_counts": dict(sorted(label_counts.items())),
        "unique_labels": len(label_counts),
        "expected_delta_present_rate": expected_delta_present_rate,
        "unique_candidate_id_rate": unique_candidate_id_rate,
        "max_label_duplicates_per_row": max_label_duplicates_per_row,
        "avg_max_label_duplicates_per_row": avg_max_label_duplicates_per_row,
    }
