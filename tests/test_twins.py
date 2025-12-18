from __future__ import annotations

from tagbench.generate import EpisodeConfig, generate_dataset
from tagbench.grade import grade_rows


def test_twins_consistency_metric() -> None:
    cfg = EpisodeConfig(
        steps=80,
        keys=8,
        queries=10,
        chapters=3,
        distractor_rate=0.5,
        require_citations=True,
        twins=True,
        distractor_profile="standard",
    )
    data = generate_dataset(seed=5, episodes=3, cfg=cfg)

    perfect = {r["id"]: {"value": r["gold"]["value"], "support_ids": r["gold"]["support_ids"]} for r in data}
    res = grade_rows(data_rows=data, pred_by_id=perfect, citations="auto")
    assert res.exact_acc == 1.0
    assert res.twin_consistency == 1.0
    assert res.twin_flip_rate == 1.0

    # Reward-hacking style failure: output the base variant's answer for both variants.
    base_value_by_pair: dict[tuple[str, int], str | None] = {}
    for r in data:
        tg = r["meta"].get("twin_group")
        qi = int(r["meta"].get("q_index"))
        if r["meta"].get("twin_variant") == "base":
            base_value_by_pair[(tg, qi)] = r["gold"]["value"]

    hacked = {}
    for r in data:
        tg = r["meta"].get("twin_group")
        qi = int(r["meta"].get("q_index"))
        hacked[r["id"]] = {"value": base_value_by_pair[(tg, qi)], "support_ids": []}

    res = grade_rows(data_rows=data, pred_by_id=hacked, citations="auto", entailment_check=False)
    assert res.twin_consistency is not None
    assert res.twin_consistency < 1.0
    assert res.twin_flip_rate is not None
    assert res.twin_flip_rate < 1.0
