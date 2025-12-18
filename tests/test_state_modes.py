from __future__ import annotations

from tagbench.baselines import parse_updates
from tagbench.generate import EpisodeConfig, generate_dataset


def test_counter_state_mode_produces_numeric_values() -> None:
    cfg = EpisodeConfig(steps=30, keys=3, queries=5, state_mode="counter", twins=False, distractor_profile="standard")
    rows = generate_dataset(seed=21, episodes=1, cfg=cfg)
    updates = parse_updates(rows[0]["document"])
    assert len(updates) == cfg.steps
    for u in updates:
        if u["value"] is None:
            continue
        assert u["value"].isdigit()


def test_set_state_mode_formats_membership() -> None:
    cfg = EpisodeConfig(steps=30, keys=3, queries=5, state_mode="set", twins=False, distractor_profile="standard")
    rows = generate_dataset(seed=22, episodes=1, cfg=cfg)
    updates = parse_updates(rows[0]["document"])
    assert len(updates) == cfg.steps
    # Values may be None (clears) or comma-joined members
    assert any("," in (u["value"] or "") for u in updates)
