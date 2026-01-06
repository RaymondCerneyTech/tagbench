from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def score_ui_rows(
    rows: Iterable[dict[str, Any]], selected_ids: list[str | None]
) -> dict[str, float]:
    rows_list = list(rows)
    total = len(rows_list)
    if len(selected_ids) != total:
        raise ValueError("selected_ids length must match rows length")
    if total == 0:
        return {
            "gold_present_rate": 0.0,
            "selection_rate": 0.0,
            "wrong_action_rate": 0.0,
            "abstain_rate": 0.0,
            "accuracy_when_gold_present": 0.0,
        }

    gold_present_count = 0
    correct_count = 0
    wrong_count = 0
    abstain_count = 0

    for row, selected in zip(rows_list, selected_ids, strict=True):
        candidates = row.get("candidates", [])
        candidate_ids = {c.get("candidate_id") for c in candidates if isinstance(c, dict)}
        gold = row.get("gold", {})
        gold_id = gold.get("candidate_id") if isinstance(gold, dict) else None

        if gold_id in candidate_ids:
            gold_present_count += 1

        if selected is None:
            abstain_count += 1
            continue

        if selected == gold_id:
            correct_count += 1
        else:
            wrong_count += 1

    accuracy_when_gold_present = (
        (correct_count / gold_present_count) if gold_present_count else 0.0
    )

    return {
        "gold_present_rate": gold_present_count / total,
        "selection_rate": correct_count / total,
        "wrong_action_rate": wrong_count / total,
        "abstain_rate": abstain_count / total,
        "accuracy_when_gold_present": accuracy_when_gold_present,
    }


def score_post_action_verification(
    rows: Iterable[dict[str, Any]], observed_deltas: list[dict[str, Any] | None]
) -> dict[str, float]:
    rows_list = list(rows)
    total = len(rows_list)
    if len(observed_deltas) != total:
        raise ValueError("observed_deltas length must match rows length")
    if total == 0:
        return {"post_action_verify_rate": 0.0}

    expected_total = 0
    verified = 0

    for row, observed in zip(rows_list, observed_deltas, strict=True):
        expected = row.get("expected_delta")
        if expected is None:
            continue
        if not isinstance(expected, dict):
            continue
        expected_total += 1
        if not isinstance(observed, dict):
            continue
        if all(observed.get(key) == value for key, value in expected.items()):
            verified += 1

    rate = (verified / expected_total) if expected_total else 0.0
    return {"post_action_verify_rate": rate}


def score_ui_sequences(
    rows: Iterable[dict[str, Any]],
    selected_ids: list[str | None],
    observed_deltas: list[dict[str, Any] | None] | None = None,
) -> dict[str, float | int | None]:
    rows_list = list(rows)
    total = len(rows_list)
    if len(selected_ids) != total:
        raise ValueError("selected_ids length must match rows length")
    if observed_deltas is not None and len(observed_deltas) != total:
        raise ValueError("observed_deltas length must match rows length")
    if total == 0:
        return {
            "tasks_total": 0,
            "task_pass_rate": 0.0,
            "task_wrong_action_rate": 0.0,
            "task_post_action_verify_mean": None,
            "task_abstain_rate_mean": 0.0,
            "task_len_mean": 0.0,
        }

    tasks: dict[str, list[int]] = {}
    for idx, row in enumerate(rows_list):
        task_id = row.get("task_id")
        if isinstance(task_id, str) and task_id.strip():
            key = task_id
        else:
            key = f"__row_{idx:04d}"
        tasks.setdefault(key, []).append(idx)

    task_passes = 0
    task_wrong_actions = 0
    task_abstain_rates: list[float] = []
    task_lengths: list[int] = []
    task_verify_rates: list[float] = []

    for indices in tasks.values():
        task_lengths.append(len(indices))
        wrong_action = False
        all_selected = True
        abstain_count = 0
        expected_total = 0
        verified = 0

        for idx in indices:
            row = rows_list[idx]
            selected = selected_ids[idx]
            candidates = row.get("candidates", [])
            candidate_ids = {
                c.get("candidate_id") for c in candidates if isinstance(c, dict)
            }
            gold = row.get("gold", {})
            gold_id = gold.get("candidate_id") if isinstance(gold, dict) else None

            if gold_id not in candidate_ids:
                all_selected = False
            if selected is None:
                abstain_count += 1
                all_selected = False
            elif selected != gold_id:
                wrong_action = True
                all_selected = False

            if observed_deltas is not None:
                expected = row.get("expected_delta")
                if isinstance(expected, dict):
                    expected_total += 1
                    observed = observed_deltas[idx]
                    if isinstance(observed, dict) and all(
                        observed.get(key) == value for key, value in expected.items()
                    ):
                        verified += 1

        if wrong_action:
            task_wrong_actions += 1
        task_abstain_rates.append(abstain_count / len(indices))

        verify_rate: float | None
        if observed_deltas is not None and expected_total:
            verify_rate = verified / expected_total
            task_verify_rates.append(verify_rate)
        else:
            verify_rate = None

        pass_condition = (
            (not wrong_action)
            and all_selected
            and (verify_rate == 1.0 or (verify_rate is None and expected_total == 0))
        )
        if pass_condition:
            task_passes += 1

    tasks_total = len(tasks)
    task_post_action_verify_mean = (
        sum(task_verify_rates) / len(task_verify_rates) if task_verify_rates else None
    )

    return {
        "tasks_total": tasks_total,
        "task_pass_rate": task_passes / tasks_total,
        "task_wrong_action_rate": task_wrong_actions / tasks_total,
        "task_post_action_verify_mean": task_post_action_verify_mean,
        "task_abstain_rate_mean": sum(task_abstain_rates) / tasks_total,
        "task_len_mean": sum(task_lengths) / tasks_total,
    }
