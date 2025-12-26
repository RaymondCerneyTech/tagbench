from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
import random

from goldevidencebench.baselines import parse_book_ledger
from goldevidencebench.book import LedgerEntry, render_book
from goldevidencebench.adapters.llama_prompt import extract_ledger, truncate_tokens, build_prompt
from goldevidencebench.util import get_env


@dataclass(frozen=True)
class RetrievalConfig:
    include_clear: bool = True
    k: int = 1
    wrong_type: str = "none"  # none|same_key|other_key
    drop_prob: float = 0.0
    drop_seed: int = 0
    order: str = "shuffle"  # shuffle|gold_first|gold_middle|gold_last
    order_seed: int = 0
    query_sandwich: bool = False
    pick_then_answer: bool = False
    rerank_mode: str = "none"  # none|latest_step


def _sorted_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda e: int(e.get("step", -1)), reverse=True)


def _latest_entry_for_key(book: str, key: str) -> dict[str, Any] | None:
    entries = parse_book_ledger(book)
    key_entries = [e for e in entries if e.get("key") == key]
    if not key_entries:
        return None
    return _sorted_entries(key_entries)[0]


def _select_entries_for_key(
    *, entries: list[dict[str, Any]], key: str, k: int, wrong_type: str
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    key_entries = [e for e in entries if e.get("key") == key]
    sorted_key = _sorted_entries(key_entries)
    correct = sorted_key[0] if sorted_key else None
    selected = sorted_key[: max(1, k)]
    wrong_entry = None
    if wrong_type == "same_key":
        if len(sorted_key) > len(selected):
            wrong_entry = sorted_key[len(selected)]
    elif wrong_type == "other_key":
        other_entries = _sorted_entries([e for e in entries if e.get("key") != key])
        if other_entries:
            wrong_entry = other_entries[0]
    if wrong_entry and wrong_entry not in selected:
        selected.append(wrong_entry)
    wrong_info = wrong_entry
    selected_sorted = sorted(selected, key=lambda e: int(e.get("step", -1)))
    correct_rank = None
    if correct and correct in selected:
        ranked = _sorted_entries(selected)
        correct_rank = 1 + ranked.index(correct)
    diag = {
        "k": k,
        "wrong_type": wrong_type,
        "correct_uid": correct.get("uid") if correct else None,
        "correct_included": correct in selected if correct else False,
        "correct_rank": correct_rank,
        "selected_count": len(selected),
    }
    return selected_sorted, diag, wrong_info


def _apply_drop_with_rng(
    *,
    selected: list[dict[str, Any]],
    correct_uid: str | None,
    wrong_entry: dict[str, Any] | None,
    drop_prob: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], bool]:
    if not correct_uid or drop_prob <= 0.0:
        return selected, False
    if rng.random() >= drop_prob:
        return selected, False
    remaining = [e for e in selected if e.get("uid") != correct_uid]
    if not remaining and wrong_entry is not None:
        remaining = [wrong_entry]
    return remaining, True


def _apply_order(
    *, selected: list[dict[str, Any]], correct_uid: str | None, order: str
) -> tuple[list[dict[str, Any]], str | None]:
    if order not in {"gold_first", "gold_middle", "gold_last"}:
        return selected, None
    if not correct_uid or not selected:
        return selected, None
    rest = [e for e in selected if e.get("uid") != correct_uid]
    gold = [e for e in selected if e.get("uid") == correct_uid]
    if not gold:
        return selected, None
    if order == "gold_first":
        return gold + rest, "gold_first"
    if order == "gold_last":
        return rest + gold, "gold_last"
    mid = len(rest) // 2
    return rest[:mid] + gold + rest[mid:], "gold_middle"


def _rerank_latest_step(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None
    return max(entries, key=lambda e: int(e.get("step", -1)))


def _build_min_book(*, entry: dict[str, Any], key: str, episode_id: str) -> str:
    ledger = [
        LedgerEntry(
            uid=entry["uid"],
            step=entry["step"],
            op=entry["op"],
            key=entry["key"],
            value=entry["value"],
        )
    ]
    glossary = {key: f"Synthetic tag {key} used for state-tracking."}
    return render_book(
        title=f"GoldEvidenceBench Retrieval {episode_id}",
        chapters=[""],
        glossary=glossary,
        ledger=ledger,
    )


def _build_multi_book(*, entries: list[dict[str, Any]], episode_id: str) -> str:
    ledger = [
        LedgerEntry(
            uid=entry["uid"],
            step=entry["step"],
            op=entry["op"],
            key=entry["key"],
            value=entry["value"],
        )
        for entry in entries
    ]
    keys = sorted({entry["key"] for entry in entries})
    glossary = {k: f"Synthetic tag {k} used for state-tracking." for k in keys}
    return render_book(
        title=f"GoldEvidenceBench Retrieval {episode_id}",
        chapters=[""],
        glossary=glossary,
        ledger=ledger,
    )


def _selection_question(question: str, key: str) -> str:
    return (
        f"{question}\n"
        f"Select the single correct support_id for {key} from the ledger above. "
        "Return value null."
    )


def _norm_support_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    s = str(value).strip()
    return [s] if s else []


class RetrievalLlamaCppAdapter:
    """
    Retrieval-first adapter:
    - Select latest ledger entry for the question key.
    - Build a minimal book with only that entry.
    - Answer using the standard Llama adapter on the tiny book.
    """

    def __init__(
        self,
        *,
        model_path: str | None = None,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        max_book_tokens: int = 1600,
    ) -> None:
        include_clear_env = get_env("RETRIEVAL_INCLUDE_CLEAR", "1").strip().lower()
        k_env = get_env("RETRIEVAL_K", "1").strip()
        wrong_type = get_env("RETRIEVAL_WRONG_TYPE", "none").strip().lower()
        drop_env = get_env("RETRIEVAL_DROP_PROB", "0").strip()
        drop_seed_env = get_env("RETRIEVAL_DROP_SEED", "0").strip()
        order_env = get_env("RETRIEVAL_ORDER", "shuffle").strip().lower()
        order_seed_env = get_env("RETRIEVAL_ORDER_SEED", "0").strip()
        sandwich_env = get_env("RETRIEVAL_QUERY_SANDWICH", "0").strip().lower()
        pick_env = get_env("RETRIEVAL_PICK_THEN_ANSWER", "0").strip().lower()
        rerank_env = get_env("RETRIEVAL_RERANK", "none").strip().lower()
        try:
            k_val = int(k_env)
        except ValueError:
            k_val = 1
        try:
            drop_prob = float(drop_env)
        except ValueError:
            drop_prob = 0.0
        try:
            drop_seed = int(drop_seed_env)
        except ValueError:
            drop_seed = 0
        if order_env not in {"shuffle", "gold_first", "gold_middle", "gold_last"}:
            order_env = "shuffle"
        try:
            order_seed = int(order_seed_env)
        except ValueError:
            order_seed = 0
        self.cfg = RetrievalConfig(
            include_clear=include_clear_env not in {"0", "false", "no"},
            k=max(1, k_val),
            wrong_type=wrong_type,
            drop_prob=max(0.0, min(1.0, drop_prob)),
            drop_seed=drop_seed,
            order=order_env,
            order_seed=order_seed,
            query_sandwich=sandwich_env in {"1", "true", "yes"},
            pick_then_answer=pick_env in {"1", "true", "yes"},
            rerank_mode=rerank_env if rerank_env in {"none", "latest_step"} else "none",
        )
        from goldevidencebench.adapters.llama_cpp_adapter import LlamaCppAdapter

        self._answerer = LlamaCppAdapter(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            max_book_tokens=max_book_tokens,
            query_sandwich=self.cfg.query_sandwich,
        )
        self._last_diag: dict[str, Any] | None = None

    @property
    def max_book_tokens(self) -> int:
        return self._answerer.max_book_tokens

    @max_book_tokens.setter
    def max_book_tokens(self, value: int) -> None:
        self._answerer.max_book_tokens = value

    def predict(self, row: dict[str, Any], *, protocol: str = "open_book") -> dict[str, Any]:
        if protocol != "closed_book":
            raise ValueError("RetrievalLlamaCppAdapter supports closed_book only.")
        book = row.get("book") or row.get("artifact")
        if not book:
            raise ValueError("book/artifact required for closed_book inference.")
        key = row.get("meta", {}).get("key")
        query_type = row.get("meta", {}).get("query_type")
        if query_type and query_type != "direct":
            return self._answerer.predict(row, protocol=protocol)
        if not key:
            return self._answerer.predict(row, protocol=protocol)
        entries = parse_book_ledger(book)
        if not entries:
            return self._answerer.predict(row, protocol=protocol)
        selected, diag, wrong_entry = _select_entries_for_key(
            entries=entries, key=key, k=self.cfg.k, wrong_type=self.cfg.wrong_type
        )
        rng = random.Random(self.cfg.drop_seed ^ hash(row.get("id", "")))
        selected, dropped = _apply_drop_with_rng(
            selected=selected,
            correct_uid=diag.get("correct_uid"),
            wrong_entry=wrong_entry,
            drop_prob=self.cfg.drop_prob,
            rng=rng,
        )
        order_applied = None
        if self.cfg.order == "shuffle" and len(selected) > 1:
            shuffle_rng = random.Random(self.cfg.order_seed ^ hash(row.get("id", "")))
            shuffle_rng.shuffle(selected)
            order_applied = "shuffle"
        elif self.cfg.order in {"gold_first", "gold_middle", "gold_last"} and selected:
            selected, order_applied = _apply_order(
                selected=selected, correct_uid=diag.get("correct_uid"), order=self.cfg.order
            )
        self._last_diag = {
            "id": row.get("id"),
            "key": key,
            **diag,
            "drop_prob": self.cfg.drop_prob,
            "dropped_correct": dropped,
            "order": order_applied,
        }
        if not selected:
            return self._answerer.predict(row, protocol=protocol)
        if len(selected) == 1:
            entry = selected[0]
            if entry.get("op") == "CLEAR" and not self.cfg.include_clear:
                return self._answerer.predict(row, protocol=protocol)
            mini_book = _build_min_book(entry=entry, key=key, episode_id=row.get("episode_id", "E0000"))
        else:
            mini_book = _build_multi_book(entries=selected, episode_id=row.get("episode_id", "E0000"))
        if self.cfg.rerank_mode == "latest_step":
            chosen = _rerank_latest_step(selected)
            self._last_diag = {
                **(self._last_diag or {}),
                "rerank_mode": "latest_step",
                "reranked_uid": chosen.get("uid") if chosen else None,
            }
            if chosen:
                mini_book = _build_min_book(
                    entry=chosen,
                    key=chosen["key"],
                    episode_id=row.get("episode_id", "E0000"),
                )
        elif self.cfg.pick_then_answer and len(selected) > 1:
            ledger = extract_ledger(mini_book)
            ledger = truncate_tokens(
                ledger,
                self.max_book_tokens,
                tokenize=getattr(self._answerer.llm, "tokenize", None),
                detokenize=getattr(self._answerer.llm, "detokenize", None),
            )
            pick_prompt = build_prompt(
                ledger=ledger,
                question=_selection_question(row["question"], key),
                require_citations=True,
                query_sandwich=self.cfg.query_sandwich,
            )
            picked = self._answerer.predict_raw_from_prompt(
                prompt=pick_prompt, require_citations=True
            )
            picked_ids = _norm_support_list(
                (picked or {}).get("support_ids") or (picked or {}).get("support_id")
            )
            chosen = picked_ids[0] if picked_ids else None
            self._last_diag = {
                **(self._last_diag or {}),
                "pick_then_answer": True,
                "picked_support_id": chosen,
            }
            if chosen:
                chosen_entry = next((e for e in selected if e.get("uid") == chosen), None)
                if chosen_entry:
                    mini_book = _build_min_book(
                        entry=chosen_entry,
                        key=chosen_entry["key"],
                        episode_id=row.get("episode_id", "E0000"),
                    )
        row_for_adapter = {**row, "book": mini_book}
        return self._answerer.predict(row_for_adapter, protocol=protocol)

    def take_perf(self) -> dict[str, Any] | None:
        return self._answerer.take_perf()

    def take_raw(self) -> dict[str, Any] | None:
        return self._answerer.take_raw()

    def take_diag(self) -> dict[str, Any] | None:
        diag = self._last_diag
        self._last_diag = None
        return diag


def create_adapter():
    return RetrievalLlamaCppAdapter()
