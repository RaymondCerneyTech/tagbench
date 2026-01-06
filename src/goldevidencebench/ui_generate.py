from __future__ import annotations

import random
from typing import Any


def generate_same_label_fixture(
    *,
    steps: int,
    duplicates: int,
    labels: list[str],
    seed: int = 0,
    app_path_prefix: str = "UI Flow",
    action_type: str = "click",
    role: str = "button",
) -> list[dict[str, Any]]:
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if duplicates <= 0:
        raise ValueError("duplicates must be > 0")
    if not labels:
        raise ValueError("labels must be non-empty")

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    task_id = f"task_{seed:04d}"

    for step_index in range(steps):
        label = rng.choice(labels)
        candidates = []
        for dup_index in range(duplicates):
            candidate_id = f"btn_{label.lower()}_{dup_index + 1}"
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "action_type": action_type,
                    "label": label,
                    "role": role,
                    "app_path": f"{app_path_prefix} > Step {step_index + 1}",
                    "bbox": [120 + dup_index * 90, 220 + step_index * 120, 80, 28],
                    "visible": True,
                    "enabled": True,
                    "modal_scope": None,
                }
            )

        gold_id = candidates[-1]["candidate_id"]
        rows.append(
            {
                "id": f"step_{step_index + 1:04d}",
                "task_id": task_id,
                "step_index": step_index + 1,
                "candidates": candidates,
                "gold": {"candidate_id": gold_id},
                "expected_delta": {"event": f"{label.lower()}_{step_index + 1}"},
            }
        )

    return rows


def generate_popup_overlay_fixture(
    *,
    steps: int,
    base_duplicates: int,
    overlay_duplicates: int,
    labels: list[str],
    seed: int = 0,
    app_path_prefix: str = "UI Flow",
    action_type: str = "click",
    role: str = "button",
    overlay_scope: str = "popup",
) -> list[dict[str, Any]]:
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if base_duplicates <= 0:
        raise ValueError("base_duplicates must be > 0")
    if overlay_duplicates <= 0:
        raise ValueError("overlay_duplicates must be > 0")
    if not labels:
        raise ValueError("labels must be non-empty")

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    task_id = f"task_{seed:04d}"

    for step_index in range(steps):
        label = rng.choice(labels)
        candidates: list[dict[str, Any]] = []
        base_app_path = f"{app_path_prefix} > Step {step_index + 1}"
        overlay_app_path = f"{base_app_path} > Popup"

        for dup_index in range(base_duplicates):
            candidate_id = f"btn_{label.lower()}_{dup_index + 1}"
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "action_type": action_type,
                    "label": label,
                    "role": role,
                    "app_path": base_app_path,
                    "bbox": [120 + dup_index * 90, 240 + step_index * 120, 80, 28],
                    "visible": True,
                    "enabled": True,
                    "modal_scope": None,
                }
            )

        for overlay_index in range(overlay_duplicates):
            candidate_id = f"popup_{label.lower()}_{overlay_index + 1}"
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "action_type": action_type,
                    "label": label,
                    "role": role,
                    "app_path": overlay_app_path,
                    "bbox": [150 + overlay_index * 90, 200 + step_index * 120, 80, 28],
                    "visible": True,
                    "enabled": True,
                    "modal_scope": overlay_scope,
                    "overlay": True,
                    "z_index": 1000,
                }
            )

        gold_id = candidates[base_duplicates - 1]["candidate_id"]
        rows.append(
            {
                "id": f"step_{step_index + 1:04d}",
                "task_id": task_id,
                "step_index": step_index + 1,
                "candidates": candidates,
                "gold": {"candidate_id": gold_id},
                "expected_delta": {"event": f"{label.lower()}_{step_index + 1}"},
            }
        )

    return rows
