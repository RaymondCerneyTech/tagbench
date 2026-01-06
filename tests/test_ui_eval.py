from goldevidencebench.ui_eval import (
    score_post_action_verification,
    score_ui_rows,
    score_ui_sequences,
)


def test_score_ui_rows() -> None:
    rows = [
        {
            "id": "step_0001",
            "candidates": [
                {"candidate_id": "btn_a"},
                {"candidate_id": "btn_b"},
            ],
            "gold": {"candidate_id": "btn_b"},
        },
        {
            "id": "step_0002",
            "candidates": [
                {"candidate_id": "btn_c"},
            ],
            "gold": {"candidate_id": "btn_c"},
        },
    ]
    selected = ["btn_b", None]
    metrics = score_ui_rows(rows, selected)
    assert metrics["gold_present_rate"] == 1.0
    assert metrics["selection_rate"] == 0.5
    assert metrics["wrong_action_rate"] == 0.0
    assert metrics["abstain_rate"] == 0.5
    assert metrics["accuracy_when_gold_present"] == 0.5


def test_score_post_action_verification() -> None:
    rows = [
        {"id": "step_0001", "expected_delta": {"page": "checkout_shipping"}},
        {"id": "step_0002", "expected_delta": {"toast": "Profile saved"}},
    ]
    observed = [{"page": "checkout_shipping"}, {"toast": "Wrong"}]
    metrics = score_post_action_verification(rows, observed)
    assert metrics["post_action_verify_rate"] == 0.5


def test_score_ui_sequences_single_task_passes() -> None:
    rows = [
        {
            "id": "step_0001",
            "task_id": "task_a",
            "candidates": [{"candidate_id": "btn_a"}, {"candidate_id": "btn_b"}],
            "gold": {"candidate_id": "btn_b"},
        },
        {
            "id": "step_0002",
            "task_id": "task_a",
            "candidates": [{"candidate_id": "btn_c"}],
            "gold": {"candidate_id": "btn_c"},
        },
    ]
    selected = ["btn_b", "btn_c"]
    metrics = score_ui_sequences(rows, selected)
    assert metrics["tasks_total"] == 1
    assert metrics["task_pass_rate"] == 1.0
    assert metrics["task_wrong_action_rate"] == 0.0


def test_score_ui_sequences_wrong_action_fails_task() -> None:
    rows = [
        {
            "id": "step_0001",
            "task_id": "task_a",
            "candidates": [{"candidate_id": "btn_a"}],
            "gold": {"candidate_id": "btn_a"},
        },
        {
            "id": "step_0002",
            "task_id": "task_a",
            "candidates": [{"candidate_id": "btn_b"}],
            "gold": {"candidate_id": "btn_b"},
        },
    ]
    selected = ["btn_a", "btn_x"]
    metrics = score_ui_sequences(rows, selected)
    assert metrics["task_pass_rate"] == 0.0
    assert metrics["task_wrong_action_rate"] == 1.0


def test_ui_fixture_adapter_selects_gold(monkeypatch) -> None:
    monkeypatch.setenv("GOLDEVIDENCEBENCH_UI_SELECTION_MODE", "gold")
    from goldevidencebench.adapters.ui_fixture_adapter import UIFixtureAdapter

    adapter = UIFixtureAdapter()
    row = {
        "id": "step_0001",
        "candidates": [{"candidate_id": "btn_a"}, {"candidate_id": "btn_b"}],
        "gold": {"candidate_id": "btn_b"},
    }
    pred = adapter.predict(row, protocol="ui")
    assert pred["value"] == "btn_b"


def test_ui_fixture_adapter_random_is_deterministic(monkeypatch) -> None:
    monkeypatch.setenv("GOLDEVIDENCEBENCH_UI_SELECTION_MODE", "random")
    monkeypatch.setenv("GOLDEVIDENCEBENCH_UI_SELECTION_SEED", "7")
    from goldevidencebench.adapters.ui_fixture_adapter import UIFixtureAdapter

    adapter = UIFixtureAdapter()
    row = {
        "id": "step_0002",
        "candidates": [{"candidate_id": "btn_a"}, {"candidate_id": "btn_b"}],
        "gold": {"candidate_id": "btn_b"},
    }
    first = adapter.predict(row, protocol="ui")["value"]
    second = adapter.predict(row, protocol="ui")["value"]
    assert first == second
