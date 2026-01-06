from goldevidencebench.ui_fixture import validate_ui_rows
from goldevidencebench.ui_generate import (
    generate_popup_overlay_fixture,
    generate_same_label_fixture,
)


def test_generate_same_label_fixture() -> None:
    rows = generate_same_label_fixture(
        steps=2,
        duplicates=3,
        labels=["Next"],
        seed=0,
        app_path_prefix="Test",
    )
    assert len(rows) == 2
    for idx, row in enumerate(rows, start=1):
        assert len(row["candidates"]) == 3
        assert row["gold"]["candidate_id"] == row["candidates"][-1]["candidate_id"]
        assert row["task_id"] == "task_0000"
        assert row["step_index"] == idx
    errors = validate_ui_rows(rows)
    assert errors == []


def test_generate_popup_overlay_fixture() -> None:
    rows = generate_popup_overlay_fixture(
        steps=2,
        base_duplicates=2,
        overlay_duplicates=1,
        labels=["Save"],
        seed=0,
        app_path_prefix="Test",
    )
    assert len(rows) == 2
    for idx, row in enumerate(rows, start=1):
        candidates = row["candidates"]
        assert len(candidates) == 3
        assert row["gold"]["candidate_id"].startswith("btn_")
        assert any(candidate.get("modal_scope") == "popup" for candidate in candidates)
        assert row["task_id"] == "task_0000"
        assert row["step_index"] == idx
    errors = validate_ui_rows(rows)
    assert errors == []
