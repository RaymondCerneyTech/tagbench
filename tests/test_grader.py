from __future__ import annotations

from tagbench.grade import grade_rows


def test_grader_citation_required_affects_exact() -> None:
    data = [
        {
            "id": "Q1",
            "gold": {"value": "amber-0001", "support_id": "U000009", "support_ids": ["U000009"]},
            "meta": {"requires_citation": True, "key": "tag.00"},
            "document": "# Ep\n\n## Episode Log\n- [U000009] UPDATE step=1 SET tag.00 = amber-0001\n",
        }
    ]

    pred_ok_value_wrong_cite = {"Q1": {"value": "amber-0001", "support_ids": ["U000001"]}}
    res = grade_rows(data_rows=data, pred_by_id=pred_ok_value_wrong_cite, citations="auto", entailment_check=False)
    assert res.value_acc == 1.0
    assert res.citation_f1 == 0.0
    assert res.exact_acc == 0.0

    pred_ok = {"Q1": {"value": "amber-0001", "support_ids": ["U000009"]}}
    res = grade_rows(data_rows=data, pred_by_id=pred_ok, citations="auto")
    assert res.exact_acc == 1.0


def test_support_truncation_and_entailment() -> None:
    data = [
        {
            "id": "Q1",
            "gold": {"value": "amber-0001", "support_ids": ["U0A1A1A"]},
            "meta": {"requires_citation": True, "key": "tag.00"},
            "document": "# Ep\n\n## Episode Log\n- [U0A1A1A] UPDATE step=1 SET tag.00 = amber-0001\n- [U0B2B2B] UPDATE step=2 SET tag.01 = teal-9999\n",
        }
    ]

    too_many = {"Q1": {"value": "amber-0001", "support_ids": ["U0A1A1A", "U0B2B2B", "U0C3C3C", "U0D4D4D"]}}
    res = grade_rows(data_rows=data, pred_by_id=too_many, citations="auto", max_support_k=3)
    assert res.citation_precision == 1 / 3  # truncated to first 3 support IDs

    wrong_entail = {"Q1": {"value": "amber-0001", "support_ids": ["U0B2B2B"]}}
    res = grade_rows(data_rows=data, pred_by_id=wrong_entail, citations="auto", entailment_check=True)
    assert res.entailment_rate == 0.0
