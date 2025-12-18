from __future__ import annotations

import pytest

from tagbench.adapters.ledger_adapter import create_adapter
from tagbench.generate import EpisodeConfig, generate_dataset
from tagbench.model_runner import run_adapter, validate_adapter_output


def test_adapter_closed_book_receives_no_document() -> None:
    cfg = EpisodeConfig(steps=12, keys=3, queries=3, twins=False, distractor_profile="standard")
    data = generate_dataset(seed=1, episodes=1, cfg=cfg)
    adapter = create_adapter()

    preds = run_adapter(data_rows=data, adapter=adapter, protocol="closed_book")
    assert preds.tokens > 0
    for row in data:
        assert row["document"]  # original has doc
    # if adapter leaked doc, validate would have failed due to closed_book guard in predict_ledger_row
    assert len(preds.predictions) == len(data)


def test_adapter_output_validation_rejects_bad_support_ids() -> None:
    cfg = EpisodeConfig(steps=6, keys=2, queries=2, twins=False, distractor_profile="standard")
    data = generate_dataset(seed=2, episodes=1, cfg=cfg)
    row = data[0]

    with pytest.raises(ValueError):
        validate_adapter_output(row=row, raw={"value": "foo", "support_ids": ["UBADBAD"]}, protocol="open_book", max_support_k=3)


def test_adapter_output_validation_rejects_missing_value() -> None:
    cfg = EpisodeConfig(steps=6, keys=2, queries=2, twins=False, distractor_profile="standard")
    data = generate_dataset(seed=3, episodes=1, cfg=cfg)
    row = data[0]

    with pytest.raises(ValueError):
        validate_adapter_output(row=row, raw={"support_ids": []}, protocol="open_book", max_support_k=3)

