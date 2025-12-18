from __future__ import annotations

from tagbench.adapters.log_to_book_adapter import LogToBookAdapter
from tagbench.generate import EpisodeConfig, generate_dataset
from tagbench.model_runner import run_adapter


def test_artifact_stats_present_for_build_adapter() -> None:
    cfg = EpisodeConfig(steps=20, keys=3, queries=4, twins=False, distractor_profile="standard")
    data = generate_dataset(seed=0, episodes=1, cfg=cfg)
    adapter = LogToBookAdapter()
    res = run_adapter(data_rows=data, adapter=adapter, protocol="closed_book")
    assert res.artifact_stats
    stat = res.artifact_stats[0]
    assert stat["has_artifact"] is True
    assert stat["ledger_entries"] > 0
