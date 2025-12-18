from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

_ID = r"U[A-F0-9]{6}"
_UPDATE_SET_RE = re.compile(
    rf"^- \[(?P<uid>{_ID})\] UPDATE step=(?P<step>\d+) SET (?P<key>tag\.\d{{2}}) = (?P<value>.+)$"
)
_UPDATE_CLEAR_RE = re.compile(
    rf"^- \[(?P<uid>{_ID})\] UPDATE step=(?P<step>\d+) CLEAR (?P<key>tag\.\d{{2}})$"
)
_BOOK_SET_RE = re.compile(
    rf"^- \[(?P<uid>{_ID})\] step=(?P<step>\d+) SET (?P<key>tag\.\d{{2}}) = (?P<value>.+)$"
)
_BOOK_CLEAR_RE = re.compile(rf"^- \[(?P<uid>{_ID})\] step=(?P<step>\d+) CLEAR (?P<key>tag\.\d{{2}})$")


def _iter_lines(document: str) -> Iterator[str]:
    for line in document.splitlines():
        yield line.rstrip("\n")


def parse_updates(document: str) -> list[dict[str, Any]]:
    in_log = False
    entries: list[dict[str, Any]] = []
    for line in _iter_lines(document):
        if line.strip() == "## Episode Log":
            in_log = True
            continue
        if not in_log:
            continue

        m = _UPDATE_SET_RE.match(line)
        if m:
            entries.append(
                {
                    "uid": m.group("uid"),
                    "step": int(m.group("step")),
                    "op": "SET",
                    "key": m.group("key").strip(),
                    "value": m.group("value").strip(),
                }
            )
            continue
        m = _UPDATE_CLEAR_RE.match(line)
        if m:
            entries.append(
                {
                    "uid": m.group("uid"),
                    "step": int(m.group("step")),
                    "op": "CLEAR",
                    "key": m.group("key").strip(),
                    "value": None,
                }
            )
            continue
    return entries


def parse_book_ledger(book: str) -> list[dict[str, Any]]:
    in_ledger = False
    entries: list[dict[str, Any]] = []
    for line in _iter_lines(book):
        if line.strip() == "## State Ledger":
            in_ledger = True
            continue
        if not in_ledger:
            continue
        if line.startswith("## "):
            break

        m = _BOOK_SET_RE.match(line)
        if m:
            entries.append(
                {
                    "uid": m.group("uid"),
                    "step": int(m.group("step")),
                    "op": "SET",
                    "key": m.group("key").strip(),
                    "value": m.group("value").strip(),
                }
            )
            continue
        m = _BOOK_CLEAR_RE.match(line)
        if m:
            entries.append(
                {
                    "uid": m.group("uid"),
                    "step": int(m.group("step")),
                    "op": "CLEAR",
                    "key": m.group("key").strip(),
                    "value": None,
                }
            )
            continue

    return entries


def predict_ledger_row(row: dict[str, Any], *, protocol: str = "open_book") -> dict[str, Any]:
    key = row["meta"]["key"]
    last_value: str | None = None
    last_uid: str | None = None
    if protocol == "open_book":
        entries = parse_updates(row["document"])
    elif protocol == "closed_book":
        # In closed_book mode, the document must not be available.
        assert row.get("document") in (None, ""), "closed_book must not receive the episode log"
        entries = parse_book_ledger(row["book"])
    else:
        raise ValueError("protocol must be open_book or closed_book")

    for e in entries:
        if e["key"] != key:
            continue
        last_uid = e["uid"]
        last_value = e["value"]
    return {
        "id": row["id"],
        "value": last_value,
        "support_id": last_uid,
        "support_ids": [last_uid] if last_uid else [],
    }


@dataclass(frozen=True)
class NaiveScanConfig:
    # If True, treat any "key = value" as an update (including distractors).
    include_distractors: bool = True


_ANY_ASSIGNMENT_RE = re.compile(r"(?P<key>tag\.\d{2})\s*=\s*(?P<value>[a-z]+-\d{4})")
_UNKNOWN_RE = re.compile(r"(?P<key>tag\.\d{2})\s*=\s*UNKNOWN\b")


def predict_naive_row(row: dict[str, Any], *, cfg: NaiveScanConfig | None = None) -> dict[str, Any]:
    cfg = cfg or NaiveScanConfig()
    key = row["meta"]["key"]

    last_value: str | None = None
    last_support: str | None = None
    for line in _iter_lines(row["document"]):
        if line.strip() == "## State Ledger":
            break
        if not cfg.include_distractors and "DISTRACTOR:" in line:
            continue
        if key not in line:
            continue
        # support_id heuristic: if a ledger ID is on the same line, capture it.
        m_uid = re.search(r"\[(U[A-F0-9]{6})\]", line)
        m = _ANY_ASSIGNMENT_RE.search(line)
        if m and m.group("key") == key:
            last_value = m.group("value")
            last_support = m_uid.group(1) if m_uid else last_support
            continue
        m = _UNKNOWN_RE.search(line)
        if m and m.group("key") == key:
            last_value = None
            last_support = m_uid.group(1) if m_uid else last_support

    return {
        "id": row["id"],
        "value": last_value,
        "support_id": last_support,
        "support_ids": [last_support] if last_support else [],
    }


def iter_predictions(
    data_rows: Iterable[dict[str, Any]], *, baseline: str, protocol: str = "open_book"
) -> Iterator[dict[str, Any]]:
    if baseline == "ledger":
        for row in data_rows:
            if protocol == "open_book":
                yield predict_ledger_row(row, protocol=protocol)
            else:
                # remove document to enforce no-leak closed-book
                row_copy = {**row, "document": None}
                yield predict_ledger_row(row_copy, protocol=protocol)
    elif baseline == "naive":
        for row in data_rows:
            if protocol == "open_book":
                yield predict_naive_row(row)
            elif protocol == "closed_book":
                # Evaluate naive scanning on the derived book artifact (chapters only; stops at ledger).
                yield predict_naive_row({**row, "document": row["book"]})
            else:
                raise ValueError("protocol must be open_book or closed_book")
    else:
        raise ValueError(f"Unknown baseline: {baseline!r}")


def parse_model_json_answer(text: str) -> dict[str, Any]:
    """
    Best-effort parser for model outputs when prompts request JSON.
    Accepts:
    - raw JSON object
    - JSON embedded in surrounding text (first {...} span)
    """
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {}
