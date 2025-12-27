from __future__ import annotations

from goldevidencebench.baselines import iter_predictions, predict_ledger_row
from goldevidencebench.generate import EpisodeConfig, generate_dataset
from goldevidencebench.grade import grade_rows


def test_ledger_baseline_beats_naive_with_distractors() -> None:
    cfg = EpisodeConfig(
        steps=200,
        keys=5,
        queries=40,
        chapters=5,
        distractor_rate=1.0,  # always inject a wrong "tag.xx = value" after each step
        clear_rate=0.0,
        require_citations=True,
        twins=False,
        distractor_profile="standard",
    )
    data = generate_dataset(seed=7, episodes=1, cfg=cfg)

    naive_preds = list(iter_predictions(data, baseline="naive", protocol="open_book"))
    ledger_preds = list(iter_predictions(data, baseline="ledger", protocol="open_book"))

    naive_res = grade_rows(data_rows=data, pred_by_id={p["id"]: p for p in naive_preds}, citations="auto")
    ledger_res = grade_rows(data_rows=data, pred_by_id={p["id"]: p for p in ledger_preds}, citations="auto")

    assert ledger_res.exact_acc == 1.0
    assert ledger_res.value_acc == 1.0
    assert naive_res.value_acc < 1.0


def test_closed_book_protocol_uses_book_artifact() -> None:
    cfg = EpisodeConfig(
        steps=120,
        keys=6,
        queries=30,
        chapters=4,
        distractor_rate=0.8,
        require_citations=True,
        twins=False,
        distractor_profile="standard",
    )
    data = generate_dataset(seed=11, episodes=1, cfg=cfg)

    ledger_closed = list(iter_predictions(data, baseline="ledger", protocol="closed_book"))
    res = grade_rows(data_rows=data, pred_by_id={p["id"]: p for p in ledger_closed}, citations="auto")
    assert res.exact_acc == 1.0

def test_commentary_note_does_not_override_state() -> None:
    row = {
        "id": "Q0001",
        "document": (
            "# Ep\n\n"
            "## Episode Log\n"
            "- [U000001] UPDATE step=1 SET tag.00 = amber-0001\n"
            "- [U000002] UPDATE step=2 NOTE tag.00 = crimson-0002\n"
        ),
        "book": "",
        "gold": {"value": "amber-0001", "support_ids": ["U000001"]},
        "meta": {"key": "tag.00", "state_mode": "kv_commentary", "query_type": "direct"},
    }
    pred = predict_ledger_row(row, protocol="open_book")
    assert pred["value"] == "amber-0001"
    assert pred["support_ids"] == ["U000001"]

