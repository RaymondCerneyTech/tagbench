from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .util import read_jsonl

REQUIRED_STEP_FIELDS = {"id", "candidates", "gold"}
REQUIRED_CANDIDATE_FIELDS = {
    "candidate_id",
    "action_type",
    "label",
    "role",
    "app_path",
    "bbox",
    "visible",
    "enabled",
    "modal_scope",
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_ui_rows(rows: Iterable[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    rows_list = list(rows)
    if not rows_list:
        return ["fixture is empty"]

    for row_index, row in enumerate(rows_list, start=1):
        if not isinstance(row, dict):
            errors.append(f"row {row_index}: expected object")
            continue

        missing_step = REQUIRED_STEP_FIELDS - row.keys()
        if missing_step:
            missing = ", ".join(sorted(missing_step))
            errors.append(f"row {row_index}: missing fields {missing}")
            continue

        step_id = row.get("id")
        if not isinstance(step_id, str) or not step_id.strip():
            errors.append(f"row {row_index}: id must be a non-empty string")

        task_id = row.get("task_id")
        if task_id is not None and (not isinstance(task_id, str) or not task_id.strip()):
            errors.append(f"row {row_index}: task_id must be a non-empty string when present")

        step_index = row.get("step_index")
        if step_index is not None:
            if not _is_int(step_index) or step_index < 1:
                errors.append(
                    f"row {row_index}: step_index must be a positive integer when present"
                )

        candidates = row.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            errors.append(f"row {row_index}: candidates must be a non-empty list")
            candidates = []

        candidate_ids: set[str] = set()
        for cand_index, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                errors.append(
                    f"row {row_index} candidate {cand_index}: expected object"
                )
                continue

            missing_candidate = REQUIRED_CANDIDATE_FIELDS - candidate.keys()
            if missing_candidate:
                missing = ", ".join(sorted(missing_candidate))
                errors.append(
                    f"row {row_index} candidate {cand_index}: missing fields {missing}"
                )
                continue

            candidate_id = candidate.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                errors.append(
                    f"row {row_index} candidate {cand_index}: candidate_id must be a non-empty string"
                )
            else:
                if candidate_id in candidate_ids:
                    errors.append(
                        f"row {row_index} candidate {cand_index}: duplicate candidate_id {candidate_id}"
                    )
                candidate_ids.add(candidate_id)

            for field in ("action_type", "label", "role", "app_path"):
                value = candidate.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(
                        f"row {row_index} candidate {cand_index}: {field} must be a non-empty string"
                    )

            bbox = candidate.get("bbox")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or not all(_is_number(v) for v in bbox)
            ):
                errors.append(
                    f"row {row_index} candidate {cand_index}: bbox must be a list of four numbers"
                )

            for field in ("visible", "enabled"):
                value = candidate.get(field)
                if not isinstance(value, bool):
                    errors.append(
                        f"row {row_index} candidate {cand_index}: {field} must be a boolean"
                    )

            modal_scope = candidate.get("modal_scope")
            if modal_scope is not None and (
                not isinstance(modal_scope, str) or not modal_scope.strip()
            ):
                errors.append(
                    f"row {row_index} candidate {cand_index}: modal_scope must be null or a non-empty string"
                )

        gold = row.get("gold")
        if not isinstance(gold, dict):
            errors.append(f"row {row_index}: gold must be an object")
        else:
            gold_id = gold.get("candidate_id")
            if not isinstance(gold_id, str) or not gold_id.strip():
                errors.append(f"row {row_index}: gold.candidate_id must be a non-empty string")
            elif gold_id not in candidate_ids:
                errors.append(
                    f"row {row_index}: gold.candidate_id {gold_id} not found in candidates"
                )

        expected_delta = row.get("expected_delta")
        if expected_delta is not None and not isinstance(expected_delta, dict):
            errors.append(f"row {row_index}: expected_delta must be an object when present")

    return errors


def validate_ui_fixture_path(path: str | Path) -> list[str]:
    rows = read_jsonl(path)
    return validate_ui_rows(rows)
