from __future__ import annotations

from tagbench.baselines import parse_book_ledger
from tagbench.generate import EpisodeConfig, generate_dataset


def test_book_artifact_does_not_embed_update_lines_verbatim() -> None:
    cfg = EpisodeConfig(steps=40, keys=4, queries=8, distractor_rate=0.4, twins=False, distractor_profile="standard")
    rows = generate_dataset(seed=3, episodes=1, cfg=cfg)
    doc_lines = {line for line in rows[0]["document"].splitlines() if line.startswith("- [") and "UPDATE" in line}
    book_lines = set(rows[0]["book"].splitlines())

    # No UPDATE lines from the episode log should appear verbatim in the book artifact.
    assert doc_lines.isdisjoint(book_lines)

    ledger = parse_book_ledger(rows[0]["book"])
    assert ledger  # sanity: still parsed


def test_closed_book_ignores_document() -> None:
    cfg = EpisodeConfig(steps=30, keys=3, queries=5, distractor_rate=0.2, twins=False, distractor_profile="standard")
    rows = generate_dataset(seed=8, episodes=1, cfg=cfg)

    # Poison the document with a fake late update; closed_book baseline must ignore it.
    poisoned = []
    for r in rows:
        poisoned.append({**r, "document": r["document"] + "\n- [UBADBAD] UPDATE step=999 SET tag.00 = poison\n"})

    from tagbench.baselines import iter_predictions
    preds = list(iter_predictions(poisoned, baseline="ledger", protocol="closed_book"))
    gold = {r["id"]: r["gold"]["value"] for r in rows}

    for p in preds:
        assert p["value"] == gold[p["id"]]
