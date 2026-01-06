from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from goldevidencebench.baselines import iter_predictions, parse_model_json_answer
from goldevidencebench.generate import EpisodeConfig, generate_dataset
from goldevidencebench.grade import grade_rows
from goldevidencebench.model_runner import load_adapter, run_adapter
from goldevidencebench.ui_eval import (
    score_post_action_verification,
    score_ui_rows,
    score_ui_sequences,
)
from goldevidencebench.ui_fixture import validate_ui_fixture_path
from goldevidencebench.ui_generate import generate_popup_overlay_fixture, generate_same_label_fixture
from goldevidencebench.ui_summary import summarize_ui_rows
from goldevidencebench.util import get_env, read_jsonl, write_jsonl


def _estimate_tokens(text: str) -> int:
    return len(text.split())


def _set_max_book_tokens(adapter: Any, max_book_tokens: int) -> None:
    if hasattr(adapter, "max_book_tokens"):
        setattr(adapter, "max_book_tokens", max_book_tokens)


def _env_snapshot() -> dict[str, Any]:
    return {
        "GOLDEVIDENCEBENCH_MODEL": get_env("MODEL"),
        "GOLDEVIDENCEBENCH_REQUIRE_CITATIONS": get_env("REQUIRE_CITATIONS"),
    }


def _print_report(res, prefix: str = "", eff: dict[str, Any] | None = None) -> None:
    prefix_str = f"{prefix}" if prefix else ""
    print(f"{prefix_str}n={res.n} value_acc={res.value_acc:.3f} exact_acc={res.exact_acc:.3f}", end="")
    if res.citation_f1 is not None:
        print(
            f" cite_f1={res.citation_f1:.3f} cite_p={res.citation_precision:.3f} cite_r={res.citation_recall:.3f}",
            end="",
        )
    if res.support_bloat is not None:
        print(f" support_bloat={res.support_bloat:.3f}", end="")
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
    if res.instr_override_rate is not None:
        print(f" instr_override={res.instr_override_rate:.3f}", end="")
    if res.instr_conflict_present_rate is not None:
        print(f" instr_conflict={res.instr_conflict_present_rate:.3f}", end="")
    if res.instr_conflict_present_count is not None:
        print(f" instr_conflict_n={res.instr_conflict_present_count}", end="")
    if res.state_integrity_rate is not None:
        print(f" state_integrity={res.state_integrity_rate:.3f}", end="")
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
        derived_query_rate=ns.derived_query_rate,
        distractor_rate=ns.distractor_rate,
        tail_distractor_steps=ns.tail_distractor_steps,
        clear_rate=ns.clear_rate,
        chapters=ns.chapters,
        require_citations=ns.require_citations,
        twins=ns.twins,
        distractor_profile=ns.distractor_profile,
        state_mode=ns.state_mode,
        note_rate=ns.note_rate,
        update_burst_rate=ns.update_burst_rate,
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
    if ns.max_book_tokens is not None:
        _set_max_book_tokens(adapter, ns.max_book_tokens)
    start = time.perf_counter()
    res_model = run_adapter(
        data_rows=data_rows,
        adapter=adapter,
        protocol=ns.protocol,
        max_support_k=ns.max_support_k,
    )
    wall = time.perf_counter() - start
    res = grade_rows(
        data_rows=data_rows,
        pred_by_id=_pred_index(res_model.predictions),
        citations=ns.citations,
        support_metric=ns.support_metric,
        max_support_k=ns.max_support_k,
        entailment_check=ns.entailment_check,
    )
    raw_res = None
    if res_model.raw_predictions:
        raw_res = grade_rows(
            data_rows=data_rows,
            pred_by_id=_pred_index(res_model.raw_predictions),
            citations=ns.citations,
            support_metric=ns.support_metric,
            max_support_k=ns.max_support_k,
            entailment_check=ns.entailment_check,
        )
    prefill_s = sum((p.get("prefill_s") or 0.0) for p in res_model.perf_stats)
    decode_s = sum((p.get("decode_s") or 0.0) for p in res_model.perf_stats)
    eff = {
        "tokens": res_model.tokens,
        "tokens_per_q": res_model.tokens_per_q,
        "passes": res_model.passes,
        "wall_s": wall,
        "wall_s_per_q": (wall / len(data_rows)) if data_rows else 0.0,
        "prefill_s": prefill_s if res_model.perf_stats else None,
        "decode_s": decode_s if res_model.perf_stats else None,
        "prefill_s_per_q": (prefill_s / len(data_rows)) if res_model.perf_stats and data_rows else None,
        "decode_s_per_q": (decode_s / len(data_rows)) if res_model.perf_stats and data_rows else None,
    }
    _print_report(res, prefix=f"adapter={ns.adapter} protocol={ns.protocol} ", eff=eff)
    if ns.out:
        write_jsonl(ns.out, res_model.predictions)
    if ns.results_json:
        results_payload = {
            "adapter_schema_version": "1.0",
            "adapter": ns.adapter,
            "protocol": ns.protocol,
            "data": {"path": str(ns.data), "n": len(data_rows)},
            "config": {
                "citations": ns.citations,
                "support_metric": ns.support_metric,
                "max_support_k": ns.max_support_k,
                "entailment_check": ns.entailment_check,
                "max_book_tokens": ns.max_book_tokens,
            },
            "env": _env_snapshot(),
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
                "instr_override_rate": res.instr_override_rate,
                "instr_conflict_present_rate": res.instr_conflict_present_rate,
                "instr_conflict_present_count": res.instr_conflict_present_count,
                "state_integrity_rate": res.state_integrity_rate,
            },
            "metrics_raw": None,
            "efficiency": eff,
            "artifact_stats": res_model.artifact_stats,
            "retrieval_stats": res_model.retrieval_stats,
        }
        if raw_res is not None:
            results_payload["metrics_raw"] = {
                "value_acc": raw_res.value_acc,
                "exact_acc": raw_res.exact_acc,
                "cite_f1": raw_res.citation_f1,
                "cite_p": raw_res.citation_precision,
                "cite_r": raw_res.citation_recall,
                "entailment": raw_res.entailment_rate,
                "twin_consistency": raw_res.twin_consistency,
                "twin_flip_rate": raw_res.twin_flip_rate,
                "instruction_acc": raw_res.instruction_acc,
                "instruction_gap": raw_res.instruction_gap,
                "instr_override_rate": raw_res.instr_override_rate,
                "instr_conflict_present_rate": raw_res.instr_conflict_present_rate,
                "instr_conflict_present_count": raw_res.instr_conflict_present_count,
                "state_integrity_rate": raw_res.state_integrity_rate,
            }
        Path(ns.results_json).write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    return 0


def _load_observed_deltas(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    observed_rows = list(read_jsonl(path))
    observed_by_id: dict[str, Any] = {}
    for row in observed_rows:
        if not isinstance(row, dict):
            continue
        row_id = row.get("id")
        observed = row.get("observed_delta")
        if isinstance(row_id, str):
            observed_by_id[row_id] = observed
    observed_deltas: list[dict[str, Any] | None] = []
    for row in rows:
        row_id = row.get("id")
        observed_deltas.append(observed_by_id.get(row_id))
    return observed_deltas


def _cmd_ui_score(ns: argparse.Namespace) -> int:
    fixture_path = Path(ns.fixture)
    if not fixture_path.exists():
        print(f"Fixture not found: {fixture_path}")
        return 1

    errors = validate_ui_fixture_path(fixture_path)
    if errors:
        print("Fixture validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    rows = list(read_jsonl(fixture_path))
    adapter = load_adapter(ns.adapter)
    selected_ids: list[str | None] = []
    for row in rows:
        pred = adapter.predict(row, protocol="ui")
        value = pred.get("value")
        if isinstance(value, str) and value.strip():
            selected_ids.append(value)
        else:
            selected_ids.append(None)

    metrics = score_ui_rows(rows, selected_ids)
    observed_deltas = None
    if ns.observed:
        observed_path = Path(ns.observed)
        if not observed_path.exists():
            print(f"Observed deltas not found: {observed_path}")
            return 1
        observed_deltas = _load_observed_deltas(observed_path, rows)
        metrics.update(score_post_action_verification(rows, observed_deltas))

    sequence_metrics = score_ui_sequences(rows, selected_ids, observed_deltas)
    payload = {
        "rows": len(rows),
        "adapter": ns.adapter,
        "metrics": metrics,
        "sequence_metrics": sequence_metrics,
    }
    if ns.out:
        Path(ns.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_ui_generate(ns: argparse.Namespace) -> int:
    labels = [label.strip() for label in ns.labels.split(",") if label.strip()]
    profile = getattr(ns, "profile", "same_label")
    overlay_duplicates = getattr(ns, "overlay_duplicates", 1)
    if profile == "popup_overlay":
        rows = generate_popup_overlay_fixture(
            steps=ns.steps,
            base_duplicates=ns.duplicates,
            overlay_duplicates=overlay_duplicates,
            labels=labels,
            seed=ns.seed,
            app_path_prefix=ns.app_path_prefix,
        )
    else:
        rows = generate_same_label_fixture(
            steps=ns.steps,
            duplicates=ns.duplicates,
            labels=labels,
            seed=ns.seed,
            app_path_prefix=ns.app_path_prefix,
        )
    write_jsonl(ns.out, rows)
    print(f"Wrote {len(rows)} rows to {ns.out}")
    return 0


def _cmd_ui_summary(ns: argparse.Namespace) -> int:
    fixture_path = Path(ns.fixture)
    if not fixture_path.exists():
        print(f"Fixture not found: {fixture_path}")
        return 1

    errors = validate_ui_fixture_path(fixture_path)
    if errors:
        print("Fixture validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    rows = list(read_jsonl(fixture_path))
    metrics = summarize_ui_rows(rows)
    payload = {"rows": len(rows), "metrics": metrics}
    if ns.out:
        Path(ns.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
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
            "wall_s": wall,
            "wall_s_per_q": (wall / len(data_rows)) if data_rows else 0.0,
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
            "data": {"path": str(ns.data), "n": len(data_rows)},
            "config": {
                "citations": ns.citations,
                "support_metric": ns.support_metric,
                "max_support_k": ns.max_support_k,
                "entailment_check": ns.entailment_check,
            },
            "env": _env_snapshot(),
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
                "instr_override_rate": res.instr_override_rate,
                "instr_conflict_present_rate": res.instr_conflict_present_rate,
                "instr_conflict_present_count": res.instr_conflict_present_count,
                "state_integrity_rate": res.state_integrity_rate,
            },
            "efficiency": eff,
            "artifact_stats": [],
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


def _cmd_sweep(ns: argparse.Namespace) -> int:
    results: list[dict[str, Any]] = []
    state_modes = ns.state_modes.split(",")
    profiles = ns.distractor_profiles.split(",")
    out_root: Path = ns.out
    out_root.mkdir(parents=True, exist_ok=True)

    adapter = load_adapter(ns.adapter) if ns.adapter else None
    max_book_tokens_list = [ns.max_book_tokens] if ns.max_book_tokens is not None else [None]
    if ns.max_book_tokens_list:
        max_book_tokens_list = [int(s) for s in ns.max_book_tokens_list.split(",") if s.strip()]

    derived_rate = 0.0 if ns.no_derived_queries else ns.derived_query_rate
    require_citations = ns.require_citations if ns.require_citations is not None else True
    sweep_config = {
        "seeds": ns.seeds,
        "episodes": ns.episodes,
        "steps": ns.steps,
        "steps_list": ns.steps_list,
        "keys": ns.keys,
        "queries": ns.queries,
        "derived_query_rate": derived_rate,
        "chapters": ns.chapters,
        "distractor_rate": ns.distractor_rate,
        "tail_distractor_steps": ns.tail_distractor_steps,
        "clear_rate": ns.clear_rate,
        "note_rate": ns.note_rate,
        "update_burst_rate": ns.update_burst_rate,
        "require_citations": require_citations,
        "twins": ns.twins,
        "state_modes": ns.state_modes,
        "distractor_profiles": ns.distractor_profiles,
        "max_book_tokens": ns.max_book_tokens,
        "max_book_tokens_list": ns.max_book_tokens_list,
        "no_derived_queries": ns.no_derived_queries,
        "no_require_citations": None,
    }
    steps_list = [ns.steps]
    if ns.steps_list:
        steps_list = [int(s) for s in ns.steps_list.split(",") if s.strip()]
    for seed in range(ns.seeds):
        for mode in state_modes:
            for profile in profiles:
                for steps in steps_list:
                    for max_book_tokens in max_book_tokens_list:
                        if adapter and max_book_tokens is not None:
                            _set_max_book_tokens(adapter, max_book_tokens)
                        cfg = EpisodeConfig(
                            steps=steps,
                            keys=ns.keys,
                            queries=ns.queries,
                            derived_query_rate=derived_rate,
                            chapters=ns.chapters,
                            distractor_rate=ns.distractor_rate,
                            tail_distractor_steps=ns.tail_distractor_steps,
                            clear_rate=ns.clear_rate,
                            note_rate=ns.note_rate,
                            update_burst_rate=ns.update_burst_rate,
                            require_citations=require_citations,
                            distractor_profile=profile,
                            state_mode=mode,
                            twins=ns.twins,
                        )
                        data = generate_dataset(seed=seed, episodes=ns.episodes, cfg=cfg)
                        run_dir_name = f"seed{seed}-steps{steps}"
                        if max_book_tokens is not None:
                            run_dir_name = f"{run_dir_name}-book{max_book_tokens}"
                        run_dir_name = f"{run_dir_name}-mode{mode}-prof{profile}"
                        run_dir = out_root / run_dir_name
                        run_dir.mkdir(parents=True, exist_ok=True)
                        data_path = run_dir / "data.jsonl"
                        preds_path = run_dir / "preds.jsonl"
                        results_path = run_dir / "results.json"
                        write_jsonl(data_path, data)

                        model_res = None
                        if adapter:
                            start = time.perf_counter()
                            model_res = run_adapter(
                                data_rows=data,
                                adapter=adapter,
                                protocol="closed_book",
                                max_support_k=ns.max_support_k,
                            )
                            wall = time.perf_counter() - start
                            prefill_s = sum((p.get("prefill_s") or 0.0) for p in model_res.perf_stats)
                            decode_s = sum((p.get("decode_s") or 0.0) for p in model_res.perf_stats)
                            preds = model_res.predictions
                            eff = {
                                "tokens": model_res.tokens,
                                "tokens_per_q": model_res.tokens_per_q,
                                "passes": model_res.passes,
                                "wall_s": wall,
                                "wall_s_per_q": (wall / len(data)) if data else 0.0,
                                "prefill_s": prefill_s if model_res.perf_stats else None,
                                "decode_s": decode_s if model_res.perf_stats else None,
                                "prefill_s_per_q": (prefill_s / len(data)) if model_res.perf_stats and data else None,
                                "decode_s_per_q": (decode_s / len(data)) if model_res.perf_stats and data else None,
                            }
                            art_stats = model_res.artifact_stats
                        else:
                            start = time.perf_counter()
                            preds = list(iter_predictions(data, baseline="ledger", protocol="closed_book"))
                            wall = time.perf_counter() - start
                            eff = {
                                "tokens": sum(_estimate_tokens(r["book"]) for r in data),
                                "tokens_per_q": sum(_estimate_tokens(r["book"]) for r in data) / len(data)
                                if data
                                else 0.0,
                                "passes": 1,
                                "wall_s": wall,
                                "wall_s_per_q": (wall / len(data)) if data else 0.0,
                            }
                            art_stats = []

                        write_jsonl(preds_path, preds)
                        res = grade_rows(
                            data_rows=data,
                            pred_by_id=_pred_index(preds),
                            citations="off" if not require_citations else "auto",
                            support_metric="f1",
                            max_support_k=ns.max_support_k,
                            entailment_check=True,
                        )
                        raw_res = None
                        if adapter and model_res and model_res.raw_predictions:
                            raw_res = grade_rows(
                                data_rows=data,
                                pred_by_id=_pred_index(model_res.raw_predictions),
                                citations="off" if not require_citations else "auto",
                                support_metric="f1",
                                max_support_k=ns.max_support_k,
                                entailment_check=True,
                            )
                        payload = {
                            "adapter_schema_version": "1.0",
                            "baseline": "ledger" if adapter is None else ns.adapter,
                            "protocol": "closed_book",
                            "seed": seed,
                            "steps": steps,
                            "state_mode": mode,
                            "distractor_profile": profile,
                            "data": {"path": str(data_path), "n": len(data)},
                            "config": {**sweep_config, "steps": steps, "max_book_tokens": max_book_tokens},
                            "env": _env_snapshot(),
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
                                "instr_override_rate": res.instr_override_rate,
                                "instr_conflict_present_rate": res.instr_conflict_present_rate,
                                "instr_conflict_present_count": res.instr_conflict_present_count,
                                "state_integrity_rate": res.state_integrity_rate,
                            },
                            "metrics_raw": None,
                            "efficiency": eff,
                            "artifact_stats": art_stats,
                            "retrieval_stats": model_res.retrieval_stats if adapter else [],
                        }
                        if raw_res is not None:
                            payload["metrics_raw"] = {
                                "value_acc": raw_res.value_acc,
                                "exact_acc": raw_res.exact_acc,
                                "cite_f1": raw_res.citation_f1,
                                "cite_p": raw_res.citation_precision,
                                "cite_r": raw_res.citation_recall,
                                "entailment": raw_res.entailment_rate,
                                "twin_consistency": raw_res.twin_consistency,
                                "twin_flip_rate": raw_res.twin_flip_rate,
                                "instruction_acc": raw_res.instruction_acc,
                                "instruction_gap": raw_res.instruction_gap,
                                "instr_override_rate": raw_res.instr_override_rate,
                                "instr_conflict_present_rate": raw_res.instr_conflict_present_rate,
                                "instr_conflict_present_count": raw_res.instr_conflict_present_count,
                                "state_integrity_rate": raw_res.state_integrity_rate,
                            }
                        results.append(payload)
                        results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if ns.results_json:
        Path(ns.results_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="goldevidencebench")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate a synthetic GoldEvidenceBench dataset (JSONL).")
    g.add_argument("--out", required=True, type=Path, help="Output JSONL path.")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--episodes", type=int, default=20)
    g.add_argument("--steps", type=int, default=220)
    g.add_argument("--keys", type=int, default=14)
    g.add_argument("--queries", type=int, default=12)
    g.add_argument("--derived-query-rate", type=float, default=0.35)
    g.add_argument("--chapters", type=int, default=8)
    g.add_argument("--distractor-rate", type=float, default=0.50)
    g.add_argument(
        "--tail-distractor-steps",
        type=int,
        default=0,
        help="Force a distractor-only tail for the final N steps to stress recency.",
    )
    g.add_argument("--clear-rate", type=float, default=0.08)
    g.add_argument(
        "--note-rate",
        type=float,
        default=0.12,
        help="NOTE rate for kv_commentary (non-authoritative ledger lines).",
    )
    g.add_argument(
        "--update-burst-rate",
        type=float,
        default=float(get_env("UPDATE_BURST_RATE", "0.25")),
        help="Rate for update_burst near-miss UPDATE scheduling.",
    )
    g.add_argument(
        "--distractor-profile",
        choices=["easy", "standard", "adversarial", "instruction", "instruction_suite", "note_camouflage", "note_camouflage_suite", "update_burst"],
        default="instruction",
        help="Adversarial adds stale-echo distractors; instruction injects spec-violating lines.",
    )
    g.add_argument(
        "--state-mode",
        choices=["kv", "kv_commentary", "counter", "set", "relational"],
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
        "--max-book-tokens",
        type=int,
        default=None,
        help="Optional cap for adapter book tokens (if supported by adapter).",
    )
    m.add_argument(
        "--entailment-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, require the answer to follow from cited UPDATE IDs only.",
    )
    m.add_argument("--out", type=Path, default=None)
    m.add_argument("--results-json", type=Path, default=None, help="Optional machine-readable metrics output (JSON).")
    m.set_defaults(func=_cmd_model)

    ui = sub.add_parser("ui-score", help="Score UI fixture rows with a UI adapter.")
    ui.add_argument(
        "--fixture",
        type=Path,
        default=Path("data/ui_same_label_fixture.jsonl"),
        help="UI fixture JSONL path.",
    )
    ui.add_argument(
        "--adapter",
        type=str,
        default="goldevidencebench.adapters.ui_fixture_adapter:create_adapter",
        help="Adapter spec module:factory.",
    )
    ui.add_argument(
        "--observed",
        type=Path,
        default=None,
        help="Optional JSONL with {id, observed_delta} rows for post-action verification.",
    )
    ui.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    ui.set_defaults(func=_cmd_ui_score)

    ug = sub.add_parser("ui-generate", help="Generate a UI fixture (JSONL).")
    ug.add_argument("--out", required=True, type=Path, help="Output JSONL path.")
    ug.add_argument(
        "--profile",
        choices=["same_label", "popup_overlay"],
        default="same_label",
        help="Fixture profile to generate.",
    )
    ug.add_argument("--steps", type=int, default=3)
    ug.add_argument("--duplicates", type=int, default=2)
    ug.add_argument("--overlay-duplicates", type=int, default=1)
    ug.add_argument(
        "--labels",
        type=str,
        default="Next,Continue,Save",
        help="Comma-separated labels to sample from.",
    )
    ug.add_argument("--seed", type=int, default=0)
    ug.add_argument("--app-path-prefix", type=str, default="UI Flow")
    ug.set_defaults(func=_cmd_ui_generate)

    us = sub.add_parser("ui-summary", help="Summarize a UI fixture (JSONL).")
    us.add_argument("--fixture", type=Path, required=True, help="UI fixture JSONL path.")
    us.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    us.set_defaults(func=_cmd_ui_summary)

    s = sub.add_parser("sweep", help="Run a small sweep over seeds/state_modes/distractors.")
    s.add_argument("--out", required=True, type=Path, help="Output directory for sweep runs.")
    s.add_argument("--seeds", type=int, default=5, help="Number of seeds (0..seeds-1).")
    s.add_argument("--episodes", type=int, default=2)
    s.add_argument("--steps", type=int, default=150)
    s.add_argument(
        "--steps-list",
        type=str,
        default=None,
        help="Optional comma-separated steps list for PaTH-style curves (e.g., 20,40,80,160).",
    )
    s.add_argument("--keys", type=int, default=10)
    s.add_argument("--queries", type=int, default=12)
    s.add_argument("--derived-query-rate", type=float, default=0.35)
    s.add_argument(
        "--no-derived-queries",
        action="store_true",
        help="Disable derived queries in sweeps (sets derived_query_rate=0).",
    )
    s.add_argument("--chapters", type=int, default=6)
    s.add_argument("--distractor-rate", type=float, default=0.5)
    s.add_argument(
        "--tail-distractor-steps",
        type=int,
        default=0,
        help="Force a distractor-only tail for the final N steps to stress recency.",
    )
    s.add_argument("--clear-rate", type=float, default=0.08)
    s.add_argument(
        "--note-rate",
        type=float,
        default=0.12,
        help="NOTE rate for kv_commentary (non-authoritative ledger lines).",
    )
    s.add_argument("--update-burst-rate", type=float, default=float(get_env("UPDATE_BURST_RATE", "0.25")),
        help="Rate for update_burst near-miss UPDATE scheduling.",
    )
    s.add_argument("--state-modes", type=str, default="kv,kv_commentary,counter,set,relational")
    s.add_argument("--distractor-profiles", type=str, default="instruction,adversarial")
    s.add_argument("--adapter", type=str, default=None, help="Optional adapter spec module:factory; defaults to ledger baseline.")
    s.add_argument("--max-support-k", type=int, default=3)
    s.add_argument(
        "--twins",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, generate a counterfactual twin for each episode (anti-shortcut metric).",
    )
    s.add_argument(
        "--require-citations",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If set, questions require ledger support IDs.",
    )
    s.add_argument(
        "--max-book-tokens",
        type=int,
        default=None,
        help="Optional cap for adapter book tokens (if supported by adapter).",
    )
    s.add_argument(
        "--max-book-tokens-list",
        type=str,
        default=None,
        help="Optional comma-separated max_book_tokens list for memory-budget sweeps (e.g., 200,400,800).",
    )
    s.add_argument("--results-json", type=Path, default=None, help="If set, write combined results JSON here.")
    s.set_defaults(func=_cmd_sweep)

    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = build_parser()
    ns = p.parse_args(argv)
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
