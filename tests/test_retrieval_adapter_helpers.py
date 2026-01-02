from __future__ import annotations

import random

from goldevidencebench.adapters.retrieval_llama_cpp_adapter import (
    _LINEAR_FEATURE_ORDER,
    _apply_authority_spoof,
    _apply_drop_with_rng,
    _apply_order,
    _build_min_book,
    _filter_authoritative,
    _select_entries_bm25,
    _select_entries_tfidf,
    _latest_entry_for_key,
    _rerank_last_occurrence,
    _rerank_latest_step,
    _rerank_linear,
    _rerank_prefer_set_latest,
    _rerank_prefer_update_latest,
    _select_entries_for_key,
    LinearSelectorModel,
)
from goldevidencebench.baselines import parse_book_ledger
from goldevidencebench.generate import EpisodeConfig, generate_dataset


def test_latest_entry_for_key_picks_last_step() -> None:
    cfg = EpisodeConfig(
        steps=25,
        keys=4,
        queries=4,
        derived_query_rate=0.0,
        chapters=2,
        distractor_rate=0.3,
        clear_rate=0.1,
        twins=False,
        distractor_profile="standard",
        state_mode="kv",
    )
    rows = generate_dataset(seed=10, episodes=1, cfg=cfg)
    book = rows[0]["book"]
    key = rows[0]["meta"]["key"]
    entry = _latest_entry_for_key(book, key)
    assert entry is not None
    for other in parse_book_ledger(book):
        if other["key"] == key:
            assert entry["step"] >= other["step"]


def test_build_min_book_contains_single_entry() -> None:
    cfg = EpisodeConfig(steps=10, keys=3, queries=3, twins=False, distractor_profile="standard")
    rows = generate_dataset(seed=2, episodes=1, cfg=cfg)
    book = rows[0]["book"]
    key = rows[0]["meta"]["key"]
    entry = _latest_entry_for_key(book, key)
    assert entry is not None
    mini = _build_min_book(entry=entry, key=key, episode_id="E0001")
    ledger = parse_book_ledger(mini)
    assert len(ledger) == 1
    assert ledger[0]["uid"] == entry["uid"]

def test_filter_authoritative_drops_notes() -> None:
    entries = [
        {"uid": "U000001", "step": 1, "op": "NOTE"},
        {"uid": "U000002", "step": 2, "op": "SET"},
    ]
    filtered = _filter_authoritative(entries)
    assert len(filtered) == 1
    assert filtered[0]["uid"] == "U000002"

    only_notes = [
        {"uid": "U000010", "step": 5, "op": "NOTE"},
        {"uid": "U000011", "step": 3, "op": "NOTE"},
    ]
    filtered = _filter_authoritative(only_notes)
    assert len(filtered) == 2



def test_select_entries_for_key_with_wrong_line() -> None:
    cfg = EpisodeConfig(steps=20, keys=3, queries=3, twins=False, distractor_profile="standard")
    rows = generate_dataset(seed=3, episodes=1, cfg=cfg)
    book = rows[0]["book"]
    key = rows[0]["meta"]["key"]
    entries = parse_book_ledger(book)
    selected, diag, _wrong = _select_entries_for_key(entries=entries, key=key, k=1, wrong_type="other_key")
    assert selected
    assert diag["selected_count"] == len(selected)
    assert diag["correct_included"] in {True, False}

def test_select_entries_bm25_prefers_query_match() -> None:
    entries = [
        {"uid": "U000001", "step": 1, "op": "SET", "key": "A", "value": "alpha"},
        {"uid": "U000002", "step": 2, "op": "SET", "key": "A", "value": "beta"},
        {"uid": "U000003", "step": 3, "op": "SET", "key": "B", "value": "gamma"},
    ]
    selected, diag, _ = _select_entries_bm25(entries=entries, question="beta", key="A", k=2)
    assert diag["correct_uid"] == "U000002"
    assert diag["correct_included"] is True
    assert diag["correct_rank"] == 1
    assert any(entry["uid"] == "U000002" for entry in selected)

def test_select_entries_tfidf_prefers_query_match() -> None:
    entries = [
        {"uid": "U000001", "step": 1, "op": "SET", "key": "A", "value": "alpha"},
        {"uid": "U000002", "step": 2, "op": "SET", "key": "A", "value": "beta"},
        {"uid": "U000003", "step": 3, "op": "SET", "key": "B", "value": "gamma"},
    ]
    selected, diag, _ = _select_entries_tfidf(entries=entries, question="beta", key="A", k=2)
    assert diag["correct_uid"] == "U000002"
    assert diag["correct_included"] is True
    assert diag["correct_rank"] == 1
    assert any(entry["uid"] == "U000002" for entry in selected)




def test_apply_authority_spoof_flips_ops() -> None:
    entries = [
        {"uid": "U000001", "step": 1, "op": "NOTE"},
        {"uid": "U000002", "step": 2, "op": "SET"},
        {"uid": "U000003", "step": 3, "op": "CLEAR"},
    ]
    rng = random.Random(123)
    spoofed, count = _apply_authority_spoof(selected=entries, rate=1.0, rng=rng)
    assert count == 3
    assert spoofed[0]["op"] == "SET"
    assert spoofed[1]["op"] == "NOTE"
    assert spoofed[2]["op"] == "NOTE"


def test_apply_drop_with_rng_removes_correct() -> None:
    selected = [{"uid": "U000001"}, {"uid": "U000002"}]
    wrong = {"uid": "U000099"}
    rng = random.Random(123)
    dropped, did_drop = _apply_drop_with_rng(
        selected=selected,
        correct_uid="U000001",
        wrong_entry=wrong,
        drop_prob=1.0,
        rng=rng,
    )
    assert did_drop is True
    assert all(e["uid"] != "U000001" for e in dropped)


def test_order_gold_last_moves_correct_to_end() -> None:
    selected = [{"uid": "U000001"}, {"uid": "U000002"}, {"uid": "U000003"}]
    ordered, applied = _apply_order(selected=selected, correct_uid="U000002", order="gold_last")
    assert applied == "gold_last"
    assert ordered[-1]["uid"] == "U000002"


def test_order_gold_middle_inserts_in_center() -> None:
    selected = [
        {"uid": "U000001"},
        {"uid": "U000002"},
        {"uid": "U000003"},
        {"uid": "U000004"},
        {"uid": "U000005"},
    ]
    ordered, applied = _apply_order(selected=selected, correct_uid="U000003", order="gold_middle")
    assert applied == "gold_middle"
    assert ordered[len(ordered) // 2]["uid"] == "U000003"


def test_rerank_last_occurrence_picks_tail() -> None:
    entries = [
        {"uid": "U000001", "step": 2},
        {"uid": "U000002", "step": 10},
        {"uid": "U000003", "step": 5},
    ]
    chosen = _rerank_last_occurrence(entries)
    assert chosen is not None
    assert chosen["uid"] == "U000003"


def test_rerank_prefer_set_latest_prefers_set() -> None:
    entries = [
        {"uid": "U000001", "step": 2, "op": "ADD"},
        {"uid": "U000002", "step": 3, "op": "SET"},
        {"uid": "U000003", "step": 10, "op": "ADD"},
    ]
    chosen = _rerank_prefer_set_latest(entries)
    assert chosen is not None
    assert chosen["uid"] == "U000002"

def test_rerank_prefer_update_latest_ignores_note() -> None:
    entries = [
        {"uid": "U000001", "step": 5, "op": "NOTE"},
        {"uid": "U000002", "step": 4, "op": "SET"},
    ]
    chosen = _rerank_prefer_update_latest(entries)
    assert chosen is not None
    assert chosen["uid"] == "U000002"

    only_notes = [
        {"uid": "U000010", "step": 7, "op": "NOTE"},
        {"uid": "U000011", "step": 2, "op": "NOTE"},
    ]
    chosen = _rerank_prefer_update_latest(only_notes)
    assert chosen is not None
    assert chosen["uid"] == "U000010"



def test_rerank_latest_step_picks_max_step() -> None:
    entries = [
        {"uid": "U000001", "step": 2},
        {"uid": "U000002", "step": 10},
        {"uid": "U000003", "step": 5},
    ]
    chosen = _rerank_latest_step(entries)
    assert chosen is not None
    assert chosen["uid"] == "U000002"


def test_rerank_linear_prefers_step() -> None:
    entries = [
        {"uid": "U000001", "step": 2, "op": "ADD"},
        {"uid": "U000002", "step": 10, "op": "SET"},
        {"uid": "U000003", "step": 5, "op": "ADD"},
    ]
    model = LinearSelectorModel(
        feature_order=_LINEAR_FEATURE_ORDER,
        weights=[0.0] * len(_LINEAR_FEATURE_ORDER),
    )
    weights = list(model.weights)
    weights[_LINEAR_FEATURE_ORDER.index("step_norm")] = 1.0
    model = LinearSelectorModel(feature_order=_LINEAR_FEATURE_ORDER, weights=weights)
    chosen = _rerank_linear(entries, model, question="what is the tag", key="T0")
    assert chosen is not None
    assert chosen["uid"] == "U000002"
