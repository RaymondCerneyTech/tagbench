from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from goldevidencebench import grade as grade_mod
from goldevidencebench.baselines import parse_model_json_answer, parse_updates
from goldevidencebench.util import read_jsonl


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["adapter_schema_version"] = row.get("adapter_schema_version")
    out["baseline"] = row.get("baseline") or row.get("adapter")
    out["protocol"] = row.get("protocol")
    out["seed"] = row.get("seed")
    out["state_mode"] = row.get("state_mode")
    out["distractor_profile"] = row.get("distractor_profile")

    data = row.get("data", {})
    out["data_path"] = data.get("path")
    out["n"] = data.get("n")

    cfg = row.get("config", {})
    for key in (
        "seeds",
        "episodes",
        "steps",
        "keys",
        "queries",
        "derived_query_rate",
        "chapters",
        "distractor_rate",
        "tail_distractor_steps",
        "clear_rate",
        "require_citations",
        "twins",
        "state_modes",
        "distractor_profiles",
        "no_derived_queries",
        "no_require_citations",
        "citations",
        "support_metric",
        "max_support_k",
        "entailment_check",
        "max_book_tokens",
    ):
        if key in cfg:
            out[key] = cfg.get(key)

    env = row.get("env", {})
    out["GOLDEVIDENCEBENCH_MODEL"] = env.get("GOLDEVIDENCEBENCH_MODEL")
    out["GOLDEVIDENCEBENCH_REQUIRE_CITATIONS"] = env.get("GOLDEVIDENCEBENCH_REQUIRE_CITATIONS")

    metrics = row.get("metrics", {})
    for key in (
        "value_acc",
        "exact_acc",
        "cite_f1",
        "cite_p",
        "cite_r",
        "support_bloat",
        "entailment",
        "twin_consistency",
        "twin_flip_rate",
        "instruction_acc",
        "instruction_gap",
        "instr_override_rate",
        "state_integrity_rate",
    ):
        if key in metrics:
            out[key] = metrics.get(key)
    metrics_raw = row.get("metrics_raw") or {}
    for key in (
        "value_acc",
        "exact_acc",
        "cite_f1",
        "cite_p",
        "cite_r",
        "support_bloat",
        "entailment",
        "twin_consistency",
        "twin_flip_rate",
        "instruction_acc",
        "instruction_gap",
        "instr_override_rate",
        "state_integrity_rate",
    ):
        if key in metrics_raw:
            out[f"raw_{key}"] = metrics_raw.get(key)
    return out


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _norm_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s if s else None
    return str(value).strip() or None


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


def _bucket_label(value: int, edges: list[int]) -> str:
    if not edges:
        return "all"
    prev = 0
    for edge in edges:
        if value < edge:
            return f"{prev}-{edge}"
        prev = edge
    return f"{edges[-1]}+"


def _parse_edges(raw: str | None, default: str) -> list[int]:
    text = raw if raw is not None else default
    return [int(s) for s in text.split(",") if s.strip()]


def _compute_decomposition(
    *,
    data_rows: list[dict[str, Any]],
    pred_by_id: dict[str, dict[str, Any]],
    retrieval_stats: list[dict[str, Any]],
) -> dict[str, Any] | None:
    retrieval_by_id: dict[str, dict[str, Any]] = {}
    for stat in retrieval_stats:
        rid = stat.get("id")
        if rid:
            retrieval_by_id[rid] = stat
    total = len(retrieval_stats)
    if total == 0:
        return None
    included = 0
    value_ok = 0
    selection_total = 0
    selection_ok = 0
    for row in data_rows:
        rid = row.get("id")
        if not rid:
            continue
        pred = pred_by_id.get(rid)
        if pred is None:
            continue
        diag = retrieval_by_id.get(rid)
        if not diag or diag.get("correct_included") is not True:
            continue
        included += 1
        gold_value = _norm_value(row.get("gold", {}).get("value"))
        pred_value = _norm_value(pred.get("value"))
        if gold_value == pred_value:
            value_ok += 1
        gold_supports = _norm_support_list(
            row.get("gold", {}).get("support_ids") or row.get("gold", {}).get("support_id")
        )
        correct_uid = diag.get("correct_uid") or (gold_supports[0] if gold_supports else None)
        if correct_uid:
            selection_total += 1
            pred_supports = _norm_support_list(pred.get("support_ids") or pred.get("support_id"))
            if correct_uid in pred_supports:
                selection_ok += 1
    gold_present_rate = included / total if total else 0.0
    acc_when = (value_ok / included) if included else 0.0
    selection_rate = (selection_ok / selection_total) if selection_total else None
    return {
        "gold_present_rate": gold_present_rate,
        "accuracy_when_gold_present": acc_when,
        "selection_rate": selection_rate,
    }


def _pred_index(pred_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in pred_rows:
        rid = r.get("id")
        if not rid:
            continue
        if "value" in r or "support_id" in r or "support_ids" in r:
            out[rid] = {
                "value": r.get("value"),
                "support_id": r.get("support_id"),
                "support_ids": r.get("support_ids"),
            }
            continue
        text = r.get("output") or r.get("text") or r.get("completion")
        if isinstance(text, str):
            parsed = parse_model_json_answer(text)
            out[rid] = {
                "value": parsed.get("value"),
                "support_id": parsed.get("support_id"),
                "support_ids": parsed.get("support_ids"),
            }
    return out


def _score_rows(
    *,
    data_rows: list[dict[str, Any]],
    pred_by_id: dict[str, dict[str, Any]],
    citations: str,
    support_metric: str,
    max_support_k: int,
    entailment_check: bool,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in data_rows:
        rid = row["id"]
        pred = pred_by_id.get(rid, {})
        gold = row["gold"]

        pv = grade_mod._norm_value(pred.get("value"))
        gv = grade_mod._norm_value(gold.get("value"))
        pred_supports = grade_mod._norm_support_list(pred.get("support_ids"))
        if not pred_supports:
            pred_supports = grade_mod._norm_support_list(pred.get("support_id"))
        gold_supports = grade_mod._norm_support_list(gold.get("support_ids"))
        if not gold_supports:
            gold_supports = grade_mod._norm_support_list(gold.get("support_id"))

        require = row["meta"].get("requires_citation", False) if citations == "auto" else citations == "on"
        is_cite_ok = True
        is_entails = True
        prec = None
        rec = None
        if require:
            pred_supports_scored = pred_supports[:max_support_k]
            prec, rec, _f1 = grade_mod._prf1(pred=pred_supports_scored, gold=gold_supports)
            is_bloat = bool(gold_supports) and len(pred_supports_scored) > len(gold_supports)
            if support_metric == "exact":
                is_cite_ok = set(pred_supports_scored) == set(gold_supports)
            else:
                is_cite_ok = set(gold_supports).issubset(set(pred_supports_scored))
            if is_bloat:
                is_cite_ok = False

            if entailment_check:
                uid_to_entry = {e["uid"]: e for e in parse_updates(row["document"])}
                cited_entries = [uid_to_entry.get(uid) for uid in pred_supports_scored]
                if any(e is None for e in cited_entries):
                    is_entails = False
                else:
                    implied = grade_mod._implied_from_citations(
                        row=row,
                        cited_entries=[e for e in cited_entries if e is not None],
                    )
                    is_entails = grade_mod._norm_value(implied) == pv
            else:
                is_entails = True

        value_ok = pv == gv
        exact_ok = value_ok and (is_cite_ok if require else True) and (is_entails if require else True)
        scored.append({"row": row, "value_ok": value_ok, "exact_ok": exact_ok, "prec": prec, "rec": rec})
    return scored


def _summarize_recency(rows: list[dict[str, Any]], edges: list[int]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for entry in rows:
        row = entry["row"]
        val = row.get("meta", {}).get("tokens_since_update")
        if val is None:
            continue
        try:
            val = int(val)
        except (TypeError, ValueError):
            continue
        label = _bucket_label(val, edges)
        buckets[label]["value_ok"].append(int(entry["value_ok"]))
        buckets[label]["exact_ok"].append(int(entry["exact_ok"]))

    out = []
    for label in sorted(buckets.keys(), key=lambda x: (x.endswith("+"), x)):
        vals = buckets[label]
        n = len(vals["value_ok"])
        out.append(
            {
                "bucket": label,
                "n": n,
                "value_acc": sum(vals["value_ok"]) / n if n else 0.0,
                "exact_acc": sum(vals["exact_ok"]) / n if n else 0.0,
            }
        )
    return out


def _summarize_bucket(
    rows: list[dict[str, Any]],
    *,
    field: str,
    edges: list[int],
) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for entry in rows:
        row = entry["row"]
        val = row.get("meta", {}).get(field)
        if val is None:
            continue
        try:
            val = int(val)
        except (TypeError, ValueError):
            continue
        label = _bucket_label(val, edges)
        buckets[label]["value_ok"].append(int(entry["value_ok"]))
        buckets[label]["exact_ok"].append(int(entry["exact_ok"]))

    out = []
    for label in sorted(buckets.keys(), key=lambda x: (x.endswith("+"), x)):
        vals = buckets[label]
        n = len(vals["value_ok"])
        out.append(
            {
                "bucket": label,
                "n": n,
                "value_acc": sum(vals["value_ok"]) / n if n else 0.0,
                "exact_acc": sum(vals["exact_ok"]) / n if n else 0.0,
            }
        )
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    grouped_raw: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    overall: dict[str, list[float]] = defaultdict(list)
    overall_raw: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        metrics = row.get("metrics", {})
        key = (row.get("state_mode") or "", row.get("distractor_profile") or "")
        for metric_key in ("value_acc", "exact_acc", "cite_f1", "entailment", "instr_override_rate", "state_integrity_rate"):
            value = metrics.get(metric_key)
            if value is None:
                continue
            grouped[key][metric_key].append(float(value))
            overall[metric_key].append(float(value))
        metrics_raw = row.get("metrics_raw") or {}
        for metric_key in ("value_acc", "exact_acc", "cite_f1", "entailment", "instr_override_rate", "state_integrity_rate"):
            value = metrics_raw.get(metric_key)
            if value is None:
                continue
            grouped_raw[key][metric_key].append(float(value))
            overall_raw[metric_key].append(float(value))

    averages = []
    for k, metric_map in sorted(grouped.items()):
        entry = {
            "state_mode": k[0],
            "distractor_profile": k[1],
            "n": max((len(v) for v in metric_map.values()), default=0),
        }
        for metric_key, values in metric_map.items():
            entry[f"{metric_key}_mean"] = _mean(values)
        averages.append(entry)

    averages_raw = []
    for k, metric_map in sorted(grouped_raw.items()):
        entry = {
            "state_mode": k[0],
            "distractor_profile": k[1],
            "n": max((len(v) for v in metric_map.values()), default=0),
        }
        for metric_key, values in metric_map.items():
            entry[f"{metric_key}_mean"] = _mean(values)
        averages_raw.append(entry)

    overall_means = {f"{k}_mean": _mean(v) for k, v in overall.items()}
    overall_raw_means = {f"{k}_mean": _mean(v) for k, v in overall_raw.items()}
    summary = {
        "rows": len(rows),
        "overall": overall_means,
        "overall_raw": overall_raw_means,
        "by_group": averages,
        "by_group_raw": averages_raw,
    }
    retrieval_stats = []
    for row in rows:
        stats = row.get("retrieval_stats")
        if isinstance(stats, list):
            retrieval_stats.extend([s for s in stats if isinstance(s, dict)])
    if retrieval_stats:
        included = [1.0 for s in retrieval_stats if s.get("correct_included") is True]
        total = len(retrieval_stats)
        ranks = [s.get("correct_rank") for s in retrieval_stats if s.get("correct_rank") is not None]
        dropped = [1.0 for s in retrieval_stats if s.get("dropped_correct") is True]
        summary["retrieval"] = {
            "n": total,
            "gold_in_context_rate": (sum(included) / total) if total else 0.0,
            "correct_rank_mean": (sum(ranks) / len(ranks)) if ranks else None,
            "drop_rate": (sum(dropped) / total) if total else 0.0,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize GoldEvidenceBench results JSON into CSV/JSON.")
    parser.add_argument("--in", dest="input_path", type=Path, default=Path("runs/combined.json"))
    parser.add_argument("--out-csv", dest="out_csv", type=Path, default=Path("runs/summary.csv"))
    parser.add_argument("--out-json", dest="out_json", type=Path, default=Path("runs/summary.json"))
    parser.add_argument(
        "--out-decomp-csv",
        dest="out_decomp_csv",
        type=Path,
        default=None,
        help="Optional path for decomposition CSV output (one row per run).",
    )
    parser.add_argument(
        "--recency-buckets",
        type=str,
        default=None,
        help="Comma-separated token buckets for tokens_since_update (e.g., 200,400,800,1600).",
    )
    parser.add_argument(
        "--distractor-buckets",
        type=str,
        default=None,
        help="Comma-separated buckets for distractors_since_update (e.g., 2,4,8,16).",
    )
    parser.add_argument(
        "--writes-buckets",
        type=str,
        default=None,
        help="Comma-separated buckets for writes_to_key (e.g., 1,2,4,8).",
    )
    args = parser.parse_args()

    rows = json.loads(args.input_path.read_text(encoding="utf-8"))
    flat = [_flatten(row) for row in rows]
    summary = summarize(rows)
    recency_edges = _parse_edges(args.recency_buckets, "200,400,800,1600")
    distractor_edges = _parse_edges(args.distractor_buckets, "2,4,8,16")
    writes_edges = _parse_edges(args.writes_buckets, "1,2,4,8")

    gold_present_total = 0
    gold_present_value_ok = 0
    selection_total = 0
    selection_ok = 0
    decomp_rows: list[dict[str, Any]] = []

    recency_rows = []
    for row in rows:
        data_path = row.get("data", {}).get("path")
        if not data_path:
            continue
        preds_path = Path(data_path).parent / "preds.jsonl"
        if not preds_path.exists():
            continue
        data_rows = list(read_jsonl(data_path))
        preds = list(read_jsonl(preds_path))
        pred_by_id = _pred_index(preds)
        retrieval_by_id: dict[str, dict[str, Any]] = {}
        stats = row.get("retrieval_stats")
        if isinstance(stats, list):
            decomp = _compute_decomposition(
                data_rows=data_rows, pred_by_id=pred_by_id, retrieval_stats=stats
            )
            if decomp:
                first = stats[0] if stats else {}
                metrics = row.get("metrics", {})
                decomp_rows.append(
                    {
                        "baseline": row.get("baseline") or row.get("adapter"),
                        "seed": row.get("seed"),
                        "state_mode": row.get("state_mode"),
                        "distractor_profile": row.get("distractor_profile"),
                        "steps": row.get("steps"),
                        "queries": row.get("data", {}).get("n"),
                        "max_book_tokens": row.get("config", {}).get("max_book_tokens"),
                        "retrieval_k": first.get("k"),
                        "retrieval_wrong_type": first.get("wrong_type"),
                        "retrieval_order": first.get("order"),
                        "retrieval_drop_prob": first.get("drop_prob"),
                        "retrieval_rerank": first.get("rerank_mode"),
                        "pick_then_answer": first.get("pick_then_answer"),
                        "gold_present_rate": decomp["gold_present_rate"],
                        "selection_rate": decomp["selection_rate"],
                        "accuracy_when_gold_present": decomp["accuracy_when_gold_present"],
                        "overall_value_acc": metrics.get("value_acc"),
                        "overall_exact_acc": metrics.get("exact_acc"),
                        "overall_cite_f1": metrics.get("cite_f1"),
                        "overall_entailment": metrics.get("entailment"),
                    }
                )
        if isinstance(stats, list):
            for stat in stats:
                if not isinstance(stat, dict):
                    continue
                rid = stat.get("id")
                if not rid:
                    continue
                retrieval_by_id[rid] = stat
        cfg = row.get("config", {})
        require_citations = cfg.get("require_citations", True)
        citations = "auto" if require_citations else "off"
        support_metric = cfg.get("support_metric", "f1")
        max_support_k = int(cfg.get("max_support_k", 3))
        entailment_check = bool(cfg.get("entailment_check", True))
        for data_row in data_rows:
            pred = pred_by_id.get(data_row.get("id"))
            if pred is None:
                continue
            diag = retrieval_by_id.get(data_row.get("id"))
            if not diag or diag.get("correct_included") is not True:
                continue
            gold_present_total += 1
            gold_value = _norm_value(data_row.get("gold", {}).get("value"))
            pred_value = _norm_value(pred.get("value"))
            if gold_value == pred_value:
                gold_present_value_ok += 1
            gold_supports = _norm_support_list(
                data_row.get("gold", {}).get("support_ids") or data_row.get("gold", {}).get("support_id")
            )
            correct_uid = diag.get("correct_uid") or (gold_supports[0] if gold_supports else None)
            if correct_uid:
                selection_total += 1
                pred_supports = _norm_support_list(pred.get("support_ids") or pred.get("support_id"))
                if correct_uid in pred_supports:
                    selection_ok += 1
        recency_rows.extend(
            _score_rows(
                data_rows=data_rows,
                pred_by_id=pred_by_id,
                citations=citations,
                support_metric=support_metric,
                max_support_k=max_support_k,
                entailment_check=entailment_check,
            )
        )

    if gold_present_total:
        retrieval_summary = summary.setdefault("retrieval", {})
        retrieval_summary["gold_present_count"] = gold_present_total
        retrieval_summary["accuracy_when_gold_present"] = gold_present_value_ok / gold_present_total
        retrieval_summary["selection_rate"] = (selection_ok / selection_total) if selection_total else None
        if "gold_in_context_rate" in retrieval_summary:
            retrieval_summary["gold_present_rate"] = retrieval_summary["gold_in_context_rate"]
        overall_acc = summary.get("overall", {}).get("value_acc_mean")
        gold_rate = retrieval_summary.get("gold_present_rate")
        sel_rate = retrieval_summary.get("selection_rate")
        acc_when = retrieval_summary.get("accuracy_when_gold_present")
        if None not in (gold_rate, sel_rate, acc_when, overall_acc):
            retrieval_summary["decomposition_line"] = (
                f"{gold_rate:.4f} -> {sel_rate:.4f} -> {acc_when:.4f} -> {overall_acc:.4f}"
            )

    if recency_rows:
        summary["recency"] = {
            "tokens_since_update": _summarize_recency(recency_rows, recency_edges),
            "distractors_since_update": _summarize_bucket(
                recency_rows, field="distractors_since_update", edges=distractor_edges
            ),
            "writes_to_key": _summarize_bucket(recency_rows, field="writes_to_key", edges=writes_edges),
        }

    if flat:
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(flat[0].keys()))
            writer.writeheader()
            writer.writerows(flat)
    if args.out_decomp_csv and decomp_rows:
        with args.out_decomp_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(decomp_rows[0].keys()))
            writer.writeheader()
            writer.writerows(decomp_rows)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_csv} and {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
