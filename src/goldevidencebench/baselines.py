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
_UPDATE_NOTE_RE = re.compile(
    rf"^- \[(?P<uid>{_ID})\] UPDATE step=(?P<step>\d+) NOTE (?P<key>tag\.\d{{2}}) = (?P<value>.+)$"
)
_BOOK_SET_RE = re.compile(
    rf"^- \[(?P<uid>{_ID})\] step=(?P<step>\d+) SET (?P<key>tag\.\d{{2}}) = (?P<value>.+)$"
)
_BOOK_CLEAR_RE = re.compile(rf"^- \[(?P<uid>{_ID})\] step=(?P<step>\d+) CLEAR (?P<key>tag\.\d{{2}})$")
_BOOK_NOTE_RE = re.compile(
    rf"^- \[(?P<uid>{_ID})\] step=(?P<step>\d+) NOTE (?P<key>tag\.\d{{2}}) = (?P<value>.+)$"
)
_BOOK_TITLE_RE = re.compile(r"^# .+")
_BOOK_GLOSSARY_RE = re.compile(r"^- (tag\.\d{2}): .+")
_BOOK_CHAPTER_RE = re.compile(r"^## Chapter \d+$")
_BOOK_RULE_RE = re.compile(r"^- .+")


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
        m = _UPDATE_NOTE_RE.match(line)
        if m:
            entries.append(
                {
                    "uid": m.group("uid"),
                    "step": int(m.group("step")),
                    "op": "NOTE",
                    "key": m.group("key").strip(),
                    "value": m.group("value").strip(),
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
        m = _BOOK_NOTE_RE.match(line)
        if m:
            entries.append(
                {
                    "uid": m.group("uid"),
                    "step": int(m.group("step")),
                    "op": "NOTE",
                    "key": m.group("key").strip(),
                    "value": m.group("value").strip(),
                }
            )
            continue

    return entries


def validate_book_artifact(book: str) -> dict[str, Any]:
    """
    Structural validation for book artifacts. Ensures allowed sections and line grammars.
    """
    lines = list(_iter_lines(book))
    errors: list[str] = []
    idx = 0

    def skip_blank(i: int) -> int:
        while i < len(lines) and lines[i].strip() == "":
            i += 1
        return i

    idx = skip_blank(idx)
    if idx >= len(lines) or not _BOOK_TITLE_RE.match(lines[idx]):
        errors.append("missing_or_invalid_title")
        return {"ok": False, "errors": errors}
    idx += 1

    idx = skip_blank(idx)
    if idx >= len(lines) or lines[idx].strip() != "## Reading Rules":
        errors.append("missing_reading_rules")
        return {"ok": False, "errors": errors}
    idx += 1
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("## "):
            break
        if line.strip() and not _BOOK_RULE_RE.match(line):
            errors.append("invalid_reading_rule")
            break
        idx += 1

    idx = skip_blank(idx)
    if idx >= len(lines) or lines[idx].strip() != "## Glossary (Tags)":
        errors.append("missing_glossary")
        return {"ok": False, "errors": errors}
    idx += 1
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("## "):
            break
        if line.strip() and not _BOOK_GLOSSARY_RE.match(line):
            errors.append("invalid_glossary_line")
            break
        idx += 1

    idx = skip_blank(idx)
    saw_chapter = False
    while idx < len(lines):
        line = lines[idx]
        if line.strip() == "## State Ledger":
            break
        if _BOOK_CHAPTER_RE.match(line):
            saw_chapter = True
            idx += 1
            continue
        if line.strip() == "## Episode Log":
            errors.append("episode_log_leak")
            break
        if _UPDATE_SET_RE.match(line) or _UPDATE_CLEAR_RE.match(line) or _UPDATE_NOTE_RE.match(line):
            errors.append("update_line_leak")
            break
        idx += 1

    if not saw_chapter:
        errors.append("missing_chapters")

    idx = skip_blank(idx)
    if idx >= len(lines) or lines[idx].strip() != "## State Ledger":
        errors.append("missing_state_ledger")
        return {"ok": False, "errors": errors}
    idx += 1
    while idx < len(lines):
        line = lines[idx]
        if line.strip() == "":
            idx += 1
            continue
        if line.startswith("## "):
            errors.append("unexpected_section_after_ledger")
            break
        if not (_BOOK_SET_RE.match(line) or _BOOK_CLEAR_RE.match(line) or _BOOK_NOTE_RE.match(line)):
            errors.append("invalid_ledger_line")
            break
        idx += 1

    return {"ok": not errors, "errors": errors}


def _parse_value(state_mode: str, raw: str | None) -> Any:
    if raw is None:
        return None
    if state_mode == "counter":
        try:
            return int(raw)
        except ValueError:
            return None
    if state_mode == "set":
        return set(raw.split(",")) if raw else set()
    return raw


def _format_value(state_mode: str, value: Any) -> str | None:
    if value is None:
        return None
    if state_mode == "set":
        if not value:
            return None
        return ",".join(sorted(value))
    return str(value)


def _apply_updates(
    state_mode: str, entries: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, str | None], dict[str, str]]:
    state: dict[str, Any] = {}
    last_support: dict[str, str | None] = {}
    last_op: dict[str, str] = {}
    for e in entries:
        key = e["key"]
        if e["op"] == "SET":
            state[key] = _parse_value(state_mode, e["value"])
            last_support[key] = e["uid"]
            last_op[key] = e["op"]
        elif e["op"] == "CLEAR":
            state[key] = set() if state_mode == "set" else None
            last_support[key] = e["uid"]
            last_op[key] = e["op"]
        else:
            continue
    return state, last_support, last_op


def _answer_from_state(
    row: dict[str, Any],
    state: dict[str, Any],
    last_support: dict[str, str | None],
    last_op: dict[str, str],
) -> dict[str, Any]:
    state_mode = row.get("meta", {}).get("state_mode", "kv")
    key = row.get("meta", {}).get("key")
    query_type = row.get("meta", {}).get("query_type", "direct")
    derived_op = row.get("meta", {}).get("derived_op")

    value: str | None = None
    support_ids: list[str] = []

    if query_type == "derived" and derived_op:
        if derived_op == "color":
            raw = _format_value(state_mode, state.get(key))
            value = raw.split("-", 1)[0] if raw else None
            if last_support.get(key):
                support_ids = [last_support[key]]  # type: ignore[index]
        elif derived_op == "parity":
            raw = state.get(key)
            if raw is not None:
                value = "even" if int(raw) % 2 == 0 else "odd"
                if last_support.get(key):
                    support_ids = [last_support[key]]  # type: ignore[index]
        elif derived_op == "count":
            raw = state.get(key)
            if last_support.get(key):
                support_ids = [last_support[key]]  # type: ignore[index]
            if raw is None or last_op.get(key) == "CLEAR":
                value = None
            else:
                value = str(len(raw)) if isinstance(raw, set) else None
        elif derived_op == "reports":
            manager = row.get("meta", {}).get("derived_manager")
            if manager:
                report_keys = sorted([k for k, v in state.items() if v == manager])
                if report_keys:
                    value = ",".join(report_keys)
                    support_ids = [last_support[k] for k in report_keys if last_support.get(k)]
    else:
        value = _format_value(state_mode, state.get(key))
        if last_support.get(key):
            support_ids = [last_support[key]]  # type: ignore[index]

    support_id = support_ids[0] if len(support_ids) == 1 else None
    return {
        "value": value,
        "support_id": support_id,
        "support_ids": support_ids,
    }


def predict_ledger_row(row: dict[str, Any], *, protocol: str = "open_book") -> dict[str, Any]:
    if protocol == "open_book":
        entries = parse_updates(row["document"])
    elif protocol == "closed_book":
        # In closed_book mode, the document must not be available.
        assert row.get("document") in (None, ""), "closed_book must not receive the episode log"
        entries = parse_book_ledger(row["book"])
    else:
        raise ValueError("protocol must be open_book or closed_book")
    state_mode = row.get("meta", {}).get("state_mode", "kv")
    state, last_support, last_op = _apply_updates(state_mode, entries)
    answer = _answer_from_state(row, state, last_support, last_op)
    return {
        "id": row["id"],
        "value": answer["value"],
        "support_id": answer["support_id"],
        "support_ids": answer["support_ids"],
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

    pred = {
        "id": row["id"],
        "value": last_value,
        "support_id": last_support,
        "support_ids": [last_support] if last_support else [],
    }
    # Apply derived transforms on top of the naive direct scan.
    if row.get("meta", {}).get("query_type") == "derived":
        state_mode = row.get("meta", {}).get("state_mode", "kv")
        derived_op = row.get("meta", {}).get("derived_op")
        if derived_op == "color" and pred["value"]:
            pred["value"] = str(pred["value"]).split("-", 1)[0]
        elif derived_op == "parity" and pred["value"]:
            try:
                pred["value"] = "even" if int(pred["value"]) % 2 == 0 else "odd"
            except ValueError:
                pred["value"] = None
        elif derived_op == "count" and pred["value"]:
            if state_mode == "set":
                parts = str(pred["value"]).split(",") if pred["value"] else []
                pred["value"] = str(len([p for p in parts if p]))
            else:
                pred["value"] = None
        elif derived_op == "reports":
            pred["value"] = None
            pred["support_id"] = None
            pred["support_ids"] = []
    return pred


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
