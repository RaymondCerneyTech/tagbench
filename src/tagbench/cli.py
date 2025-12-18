from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from tagbench.baselines import iter_predictions, parse_model_json_answer
from tagbench.generate import EpisodeConfig, generate_dataset
from tagbench.grade import grade_rows
from tagbench.model_runner import load_adapter, run_adapter
from tagbench.util import read_jsonl, write_jsonl


def _estimate_tokens(text: str) -> int:
    return len(text.split())


def _print_report(res, prefix: str = "", eff: dict[str, Any] | None = None) -> None:
    prefix_str = f"{prefix}" if prefix else ""
    print(f"{prefix_str}n={res.n} value_acc={res.value_acc:.3f} exact_acc={res.exact_acc:.3f}", end="")
    if res.citation_f1 is not None:
        print(
            f" cite_f1={res.citation_f1:.3f} cite_p={res.citation_precision:.3f} cite_r={res.citation_recall:.3f}",
            end="",
        )
    if res.entailment_rate is not None:
        print(f" entailment={res.entailment_rate:.3f}", end="")
    if res.twin_consistency is not None:
        print(f" twin_consistency={res.twin_consistency:.3f}", end="")
    if res.twin_flip_rate is not None:
        print(f" twin_flip_rate={res.twin_flip_rate:.3f}", end="")
    if res.instruction_acc is not None:
        print(f" instr_acc={res.instruction_acc:.3f}", end="")
    if res.instruction_gap is not None:
        print(f" instr_gap={res.instruction_gap:.3f}", end="")
    if eff:
        print(
            f" tokens={eff['tokens']} (~{eff['tokens_per_q']:.1f}/q) passes={eff['passes']} wall_s={eff['wall']:.2f}",
            end="",
        )
    print()


def _cmd_generate(ns: argparse.Namespace) -> int:
    cfg = EpisodeConfig(
        steps=ns.steps,
        keys=ns.keys,
        queries=ns.queries,
        distractor_rate=ns.distractor_rate,
        clear_rate=ns.clear_rate,
        chapters=ns.chapters,
        require_citations=ns.require_citations,
        twins=ns.twins,
        distractor_profile=ns.distractor_profile,
        state_mode=ns.state_mode,
    )
    rows = generate_dataset(seed=ns.seed, episodes=ns.episodes, cfg=cfg)
    write_jsonl(ns.out, rows)
    print(f"Wrote {len(rows)} rows to {ns.out}")
    return 0


def _cmd_baseline(ns: argparse.Namespace) -> int:
    rows = list(read_jsonl(ns.data))
    preds = list(iter_predictions(rows, baseline=ns.baseline, protocol=ns.protocol))
    write_jsonl(ns.out, preds)
    print(f"Wrote {len(preds)} predictions to {ns.out}")
    return 0


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
        # Accept common model-output wrappers
        text = r.get("output") or r.get("text") or r.get("completion")
        if isinstance(text, str):
            parsed = parse_model_json_answer(text)
            out[rid] = {
                "value": parsed.get("value"),
                "support_id": parsed.get("support_id"),
                "support_ids": parsed.get("support_ids"),
            }
    return out


def _cmd_grade(ns: argparse.Namespace) -> int:
    data_rows = list(read_jsonl(ns.data))
    pred_rows = list(read_jsonl(ns.pred))
    pred_by_id = _pred_index(pred_rows)
    res = grade_rows(
        data_rows=data_rows,
        pred_by_id=pred_by_id,
        citations=ns.citations,
        support_metric=ns.support_metric,
        max_support_k=ns.max_support_k,
        entailment_check=ns.entailment_check,
    )
    _print_report(res, prefix="")
    return 0


def _cmd_model(ns: argparse.Namespace) -> int:
    data_rows = list(read_jsonl(ns.data))
    adapter = load_adapter(ns.adapter)
    res_model = run_adapter(
        data_rows=data_rows,
        adapter=adapter,
        protocol=ns.protocol,
        max_support_k=ns.max_support_k,
    )
    res = grade_rows(
        data_rows=data_rows,
        pred_by_id=_pred_index(res_model.predictions),
        citations=ns.citations,
        support_metric=ns.support_metric,
        max_support_k=ns.max_support_k,
        entailment_check=ns.entailment_check,
    )
    eff = {"tokens": res_model.tokens, "tokens_per_q": res_model.tokens_per_q, "passes": res_model.passes, "wall": 0.0}
    _print_report(res, prefix=f"adapter={ns.adapter} protocol={ns.protocol} ", eff=eff)
    if ns.out:
        write_jsonl(ns.out, res_model.predictions)
    if ns.results_json:
        results_payload = {
            "adapter_schema_version": "1.0",
            "adapter": ns.adapter,
            "protocol": ns.protocol,
            "metrics": {
                "value_acc": res.value_acc,
                "exact_acc": res.exact_acc,
                "cite_f1": res.citation_f1,
                "cite_p": res.citation_precision,
                "cite_r": res.citation_recall,
                "entailment": res.entailment_rate,
                "twin_consistency": res.twin_consistency,
                "twin_flip_rate": res.twin_flip_rate,
                "instruction_acc": res.instruction_acc,
                "instruction_gap": res.instruction_gap,
            },
            "efficiency": eff,
        }
        Path(ns.results_json).write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    return 0


def _cmd_run(ns: argparse.Namespace) -> int:
    data_rows = list(read_jsonl(ns.data))
    protocols = ["open_book", "closed_book"] if ns.protocol == "both" else [ns.protocol]
    results_payloads = []
    for protocol in protocols:
        start = time.perf_counter()
        preds = list(iter_predictions(data_rows, baseline=ns.baseline, protocol=protocol))
        wall = time.perf_counter() - start
        tokens = sum(_estimate_tokens(r["document"] if protocol == "open_book" else r["book"]) for r in data_rows)
        eff = {
            "tokens": tokens,
            "tokens_per_q": tokens / len(data_rows) if data_rows else 0.0,
            "passes": 1,
            "wall": wall,
        }
        res = grade_rows(
            data_rows=data_rows,
            pred_by_id=_pred_index(preds),
            citations=ns.citations,
            support_metric=ns.support_metric,
            max_support_k=ns.max_support_k,
            entailment_check=ns.entailment_check,
        )
        _print_report(res, prefix=f"baseline={ns.baseline} protocol={protocol} ", eff=eff)
        payload = {
            "adapter_schema_version": "1.0",
            "baseline": ns.baseline,
            "protocol": protocol,
            "metrics": {
                "value_acc": res.value_acc,
                "exact_acc": res.exact_acc,
                "cite_f1": res.citation_f1,
                "cite_p": res.citation_precision,
                "cite_r": res.citation_recall,
                "entailment": res.entailment_rate,
                "twin_consistency": res.twin_consistency,
                "twin_flip_rate": res.twin_flip_rate,
                "instruction_acc": res.instruction_acc,
                "instruction_gap": res.instruction_gap,
            },
            "efficiency": eff,
        }
        results_payloads.append(payload)

        if ns.out:
            out_path = ns.out
            if ns.protocol == "both":
                out_path = Path(str(ns.out).replace(".jsonl", f".{protocol}.jsonl"))
            write_jsonl(out_path, preds)

    if ns.results_json:
        Path(ns.results_json).write_text(json.dumps(results_payloads, indent=2), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tagbench")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate a synthetic TagBench dataset (JSONL).")
    g.add_argument("--out", required=True, type=Path, help="Output JSONL path.")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--episodes", type=int, default=20)
    g.add_argument("--steps", type=int, default=220)
    g.add_argument("--keys", type=int, default=14)
    g.add_argument("--queries", type=int, default=12)
    g.add_argument("--chapters", type=int, default=8)
    g.add_argument("--distractor-rate", type=float, default=0.50)
    g.add_argument("--clear-rate", type=float, default=0.08)
    g.add_argument(
        "--distractor-profile",
        choices=["easy", "standard", "adversarial", "instruction"],
        default="instruction",
        help="Adversarial adds stale-echo distractors; instruction injects spec-violating lines.",
    )
    g.add_argument(
        "--state-mode",
        choices=["kv", "counter", "set", "relational"],
        default="kv",
        help="State dynamics to generate.",
    )
    g.add_argument(
        "--twins",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, generate a counterfactual twin for each episode (anti-shortcut metric).",
    )
    g.add_argument(
        "--require-citations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, questions require ledger support IDs.",
    )
    g.set_defaults(func=_cmd_generate)

    b = sub.add_parser("baseline", help="Run a baseline predictor and write predictions JSONL.")
    b.add_argument("--data", required=True, type=Path)
    b.add_argument("--baseline", choices=["naive", "ledger"], default="ledger")
    b.add_argument("--protocol", choices=["open_book", "closed_book"], default="closed_book")
    b.add_argument("--out", required=True, type=Path)
    b.set_defaults(func=_cmd_baseline)

    gr = sub.add_parser("grade", help="Grade predictions against a dataset.")
    gr.add_argument("--data", required=True, type=Path)
    gr.add_argument("--pred", required=True, type=Path)
    gr.add_argument("--citations", choices=["auto", "on", "off"], default="auto")
    gr.add_argument("--support-metric", choices=["f1", "exact"], default="f1")
    gr.add_argument("--max-support-k", type=int, default=3)
    gr.add_argument(
        "--entailment-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, require the answer to follow from cited UPDATE IDs only.",
    )
    gr.set_defaults(func=_cmd_grade)

    r = sub.add_parser("run", help="Run a baseline and print metrics (optionally write predictions).")
    r.add_argument("--data", required=True, type=Path)
    r.add_argument("--baseline", choices=["naive", "ledger"], default="ledger")
    r.add_argument("--citations", choices=["auto", "on", "off"], default="auto")
    r.add_argument("--support-metric", choices=["f1", "exact"], default="f1")
    r.add_argument("--max-support-k", type=int, default=3)
    r.add_argument(
        "--entailment-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, require the answer to follow from cited UPDATE IDs only.",
    )
    r.add_argument("--protocol", choices=["open_book", "closed_book", "both"], default="closed_book")
    r.add_argument("--out", type=Path, default=None)
    r.add_argument("--results-json", type=Path, default=None, help="Optional machine-readable metrics output.")
    r.set_defaults(func=_cmd_run)

    m = sub.add_parser("model", help="Run a custom adapter (LLM or tool) and grade it.")
    m.add_argument("--data", required=True, type=Path)
    m.add_argument("--adapter", required=True, help="Adapter spec module:factory (factory returns .predict(row)).")
    m.add_argument("--protocol", choices=["open_book", "closed_book"], default="closed_book")
    m.add_argument("--citations", choices=["auto", "on", "off"], default="auto")
    m.add_argument("--support-metric", choices=["f1", "exact"], default="f1")
    m.add_argument("--max-support-k", type=int, default=3)
    m.add_argument(
        "--entailment-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, require the answer to follow from cited UPDATE IDs only.",
    )
    m.add_argument("--out", type=Path, default=None)
    m.add_argument("--results-json", type=Path, default=None, help="Optional machine-readable metrics output (JSON).")
    m.set_defaults(func=_cmd_model)

    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = build_parser()
    ns = p.parse_args(argv)
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
