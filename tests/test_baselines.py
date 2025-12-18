from __future__ import annotations

from tagbench.baselines import iter_predictions
from tagbench.generate import EpisodeConfig, generate_dataset
from tagbench.grade import grade_rows


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
