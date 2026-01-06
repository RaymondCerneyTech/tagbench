from goldevidencebench.ui_summary import summarize_ui_rows


def test_summarize_ui_rows() -> None:
    rows = [
        {
            "id": "step_0001",
            "candidates": [
                {"candidate_id": "btn_a", "label": "Next"},
                {"candidate_id": "btn_b", "label": "Next"},
            ],
            "gold": {"candidate_id": "btn_b"},
            "expected_delta": {"page": "checkout_shipping"},
        },
        {
            "id": "step_0002",
            "candidates": [
                {"candidate_id": "btn_c", "label": "Save"},
            ],
            "gold": {"candidate_id": "btn_c"},
        },
    ]
    metrics = summarize_ui_rows(rows)
    assert metrics["rows"] == 2
    assert metrics["candidates_total"] == 3
    assert metrics["unique_labels"] == 2
    assert metrics["expected_delta_present_rate"] == 0.5
    assert metrics["max_label_duplicates_per_row"] == 2
