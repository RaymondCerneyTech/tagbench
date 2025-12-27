from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GradeResult:
    n: int
    value_acc: float
    citation_precision: float | None
    citation_recall: float | None
    citation_f1: float | None
    support_bloat: float | None
    exact_acc: float
    twin_consistency: float | None
    twin_flip_rate: float | None
    entailment_rate: float | None
    instruction_acc: float | None
    clean_acc: float | None
    instruction_gap: float | None
    instr_override_rate: float | None
    state_integrity_rate: float | None


def _norm_value(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return str(v).strip() or None


def _norm_support(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return str(v).strip() or None


def _norm_support_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        out: list[str] = []
        for x in v:
            s = _norm_support(x)
            if s is not None:
                out.append(s)
        return out
    s = _norm_support(v)
    return [s] if s is not None else []


def _prf1(*, pred: list[str], gold: list[str]) -> tuple[float, float, float]:
    ps = set(pred)
    gs = set(gold)
    tp = len(ps & gs)
    prec = tp / len(ps) if ps else 0.0
    rec = tp / len(gs) if gs else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return prec, rec, f1


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


def _implied_from_citations(
    *, row: dict[str, Any], cited_entries: list[dict[str, Any]]
) -> str | None:
    state_mode = row.get("meta", {}).get("state_mode", "kv")
    query_type = row.get("meta", {}).get("query_type", "direct")
    derived_op = row.get("meta", {}).get("derived_op")
    key = row.get("meta", {}).get("key")

    def last_value_for_key() -> tuple[Any, str | None]:
        last_step = -1
        implied: Any = None
        last_op: str | None = None
        for e in sorted(cited_entries, key=lambda x: x["step"]):
            if e["key"] != key:
                continue
            if e["op"] == "NOTE":
                continue
            if e["step"] >= last_step:
                last_step = e["step"]
                implied = _parse_value(state_mode, e["value"])
                last_op = e["op"]
        return implied, last_op

    if query_type == "derived" and derived_op == "reports":
        manager = row.get("meta", {}).get("derived_manager")
        if not manager:
            return None
        report_keys = sorted([e["key"] for e in cited_entries if e["value"] == manager])
        if not report_keys:
            return None
        return ",".join(report_keys)

    raw_value, last_op = last_value_for_key()
    if query_type == "derived" and derived_op == "color":
        if raw_value is None:
            return None
        return str(raw_value).split("-", 1)[0]
    if query_type == "derived" and derived_op == "parity":
        if raw_value is None:
            return None
        return "even" if int(raw_value) % 2 == 0 else "odd"
    if query_type == "derived" and derived_op == "count":
        if raw_value is None:
            return None
        if last_op == "CLEAR":
            return None
        if isinstance(raw_value, set):
            return str(len(raw_value))
        return None
    return _format_value(state_mode, raw_value)


def _twin_consistency(data_rows: list[dict[str, Any]], pred_by_id: dict[str, dict[str, Any]]) -> float | None:
    pairs: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in data_rows:
        tg = row.get("meta", {}).get("twin_group")
        qi = row.get("meta", {}).get("q_index")
        if tg is None or qi is None:
            continue
        pairs.setdefault((str(tg), int(qi)), []).append(row)

    total = 0
    ok = 0
    for (_tg, _qi), rows in pairs.items():
        if len(rows) != 2:
            continue
        r1, r2 = rows
        g1 = _norm_value(r1["gold"].get("value"))
        g2 = _norm_value(r2["gold"].get("value"))
        p1 = _norm_value(pred_by_id.get(r1["id"], {}).get("value"))
        p2 = _norm_value(pred_by_id.get(r2["id"], {}).get("value"))

        # If gold is equal across variants, predictions should match; if gold differs, predictions should differ.
        want_equal = g1 == g2
        got_equal = p1 == p2
        total += 1
        if want_equal == got_equal:
            ok += 1

    return (ok / total) if total else None


def _twin_flip_rate(data_rows: list[dict[str, Any]], pred_by_id: dict[str, dict[str, Any]]) -> float | None:
    pairs: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in data_rows:
        tg = row.get("meta", {}).get("twin_group")
        qi = row.get("meta", {}).get("q_index")
        if tg is None or qi is None:
            continue
        pairs.setdefault((str(tg), int(qi)), []).append(row)

    total = 0
    ok = 0
    for (_tg, _qi), rows in pairs.items():
        if len(rows) != 2:
            continue
        base = next((r for r in rows if r["meta"].get("twin_variant") == "base"), rows[0])
        twin = next((r for r in rows if r["meta"].get("twin_variant") == "twin"), rows[-1])

        gb = _norm_value(base["gold"].get("value"))
        gt = _norm_value(twin["gold"].get("value"))
        pb = _norm_value(pred_by_id.get(base["id"], {}).get("value"))
        pt = _norm_value(pred_by_id.get(twin["id"], {}).get("value"))

        if gb == gt:
            continue
        total += 1
        if pb != pt:
            ok += 1

    return (ok / total) if total else None


def grade_rows(
    *,
    data_rows: list[dict[str, Any]],
    pred_by_id: dict[str, dict[str, Any]],
    citations: str = "auto",  # "auto" | "on" | "off"
    support_metric: str = "f1",  # "f1" | "exact"
    max_support_k: int = 3,
    entailment_check: bool = True,
) -> GradeResult:
    if citations not in {"auto", "on", "off"}:
        raise ValueError("citations must be one of: auto, on, off")
    if support_metric not in {"f1", "exact"}:
        raise ValueError("support_metric must be one of: f1, exact")
    if max_support_k < 1:
        raise ValueError("max_support_k must be >= 1")

    n = 0
    value_ok = 0
    cite_prec_sum = 0.0
    cite_rec_sum = 0.0
    cite_f1_sum = 0.0
    cite_total = 0
    bloat_total = 0
    bloat_count = 0
    entail_total = 0
    entail_ok = 0
    exact_ok = 0
    instr_total = 0
    instr_ok = 0
    clean_total = 0
    clean_ok = 0
    instr_override_total = 0
    instr_override_count = 0
    instr_integrity_total = 0
    instr_integrity_ok = 0

    for row in data_rows:
        rid = row["id"]
        pred = pred_by_id.get(rid, {})
        gold = row["gold"]

        pv = _norm_value(pred.get("value"))
        gv = _norm_value(gold.get("value"))
        pred_supports = _norm_support_list(pred.get("support_ids"))
        if not pred_supports:
            pred_supports = _norm_support_list(pred.get("support_id"))
        gold_supports = _norm_support_list(gold.get("support_ids"))
        if not gold_supports:
            gold_supports = _norm_support_list(gold.get("support_id"))

        n += 1
        is_value_ok = pv == gv
        if is_value_ok:
            value_ok += 1

        require = row["meta"].get("requires_citation", False) if citations == "auto" else citations == "on"
        is_cite_ok = True
        is_entails = True
        if require:
            cite_total += 1
            pred_supports_scored = pred_supports[:max_support_k]
            prec, rec, f1 = _prf1(pred=pred_supports_scored, gold=gold_supports)
            cite_prec_sum += prec
            cite_rec_sum += rec
            cite_f1_sum += f1

            is_bloat = bool(gold_supports) and len(pred_supports_scored) > len(gold_supports)
            bloat_total += 1
            if is_bloat:
                bloat_count += 1

            if support_metric == "exact":
                is_cite_ok = set(pred_supports_scored) == set(gold_supports)
            else:
                # "f1" mode: exact requires recall=1 (gold support included) and not exceeding max_k.
                is_cite_ok = set(gold_supports).issubset(set(pred_supports_scored))
            if is_bloat:
                is_cite_ok = False

            if entailment_check:
                # "Entailment from citations": using only cited authoritative updates, the answer must follow.
                from goldevidencebench.baselines import parse_updates

                uid_to_entry = {e["uid"]: e for e in parse_updates(row["document"])}
                cited_entries = [uid_to_entry.get(uid) for uid in pred_supports_scored]
                if any(e is None for e in cited_entries):
                    is_entails = False
                else:
                    implied = _implied_from_citations(
                        row=row,
                        cited_entries=[e for e in cited_entries if e is not None],
                    )
                    is_entails = _norm_value(implied) == pv
                entail_total += 1
                entail_ok += 1 if is_entails else 0

        if is_value_ok and (is_cite_ok if require else True) and (is_entails if require else True):
            exact_ok += 1
            if row["meta"].get("has_instruction"):
                instr_ok += 1
            else:
                clean_ok += 1
        if row["meta"].get("has_instruction"):
            instr_total += 1
        else:
            clean_total += 1

        if row["meta"].get("has_instruction"):
            instr_override_total += 1
            instr_integrity_total += 1
            instr_value = _norm_value(row["meta"].get("instruction_value"))
            if instr_value is not None and pv == instr_value:
                instr_override_count += 1
            if pv == gv:
                instr_integrity_ok += 1

    return GradeResult(
        n=n,
        value_acc=value_ok / n if n else 0.0,
        citation_precision=(cite_prec_sum / cite_total) if cite_total else None,
        citation_recall=(cite_rec_sum / cite_total) if cite_total else None,
        citation_f1=(cite_f1_sum / cite_total) if cite_total else None,
        support_bloat=(bloat_count / bloat_total) if bloat_total else None,
        exact_acc=exact_ok / n if n else 0.0,
        twin_consistency=_twin_consistency(data_rows, pred_by_id),
        twin_flip_rate=_twin_flip_rate(data_rows, pred_by_id),
        entailment_rate=(entail_ok / entail_total) if entail_total else None,
        instruction_acc=(instr_ok / instr_total) if instr_total else None,
        clean_acc=(clean_ok / clean_total) if clean_total else None,
        instruction_gap=((clean_ok / clean_total) - (instr_ok / instr_total)) if instr_total and clean_total else None,
        instr_override_rate=(instr_override_count / instr_override_total) if instr_override_total else None,
        state_integrity_rate=(instr_integrity_ok / instr_integrity_total) if instr_integrity_total else None,
    )
