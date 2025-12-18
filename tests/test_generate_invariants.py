from __future__ import annotations

from tagbench.baselines import parse_updates
from tagbench.generate import EpisodeConfig, generate_dataset


def test_generate_ledger_has_steps_and_support_ids_match() -> None:
    cfg = EpisodeConfig(
        steps=50,
        keys=6,
        queries=10,
        chapters=3,
        distractor_rate=0.3,
        clear_rate=0.1,
        twins=False,
        distractor_profile="standard",
        state_mode="kv",
    )
    rows = generate_dataset(seed=123, episodes=1, cfg=cfg)
    assert len(rows) == cfg.queries

    doc = rows[0]["document"]
    updates = parse_updates(doc)
    assert len(updates) == cfg.steps

    for row in rows:
        key = row["meta"]["key"]
        gold = row["gold"]
        last_uid = None
        last_value = None
        for e in updates:
            if e["key"] == key:
                last_uid = e["uid"]
                last_value = e["value"]

        assert gold["support_id"] == last_uid
        assert gold["value"] == last_value
