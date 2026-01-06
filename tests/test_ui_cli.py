import json
from types import SimpleNamespace
from pathlib import Path

from goldevidencebench import cli


def test_cmd_ui_score_writes_output(tmp_path: Path, monkeypatch) -> None:
    fixture_path = tmp_path / "fixture.jsonl"
    fixture_path.write_text(
        '{"id":"step_0001","candidates":[{"candidate_id":"btn_a","action_type":"click","label":"Next","role":"button","app_path":"Test","bbox":[0,0,10,10],"visible":true,"enabled":true,"modal_scope":null}],"gold":{"candidate_id":"btn_a"}}\n',
        encoding="utf-8",
    )
    out_path = tmp_path / "out.json"
    monkeypatch.setenv("GOLDEVIDENCEBENCH_UI_SELECTION_MODE", "gold")
    ns = SimpleNamespace(
        fixture=fixture_path,
        adapter="goldevidencebench.adapters.ui_fixture_adapter:create_adapter",
        observed=None,
        out=out_path,
    )
    rc = cli._cmd_ui_score(ns)
    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["selection_rate"] == 1.0
    assert payload["sequence_metrics"]["tasks_total"] == 1


def test_cmd_ui_generate_writes_fixture(tmp_path: Path) -> None:
    out_path = tmp_path / "fixture.jsonl"
    ns = SimpleNamespace(
        out=out_path,
        profile="same_label",
        steps=2,
        duplicates=2,
        overlay_duplicates=1,
        labels="Next,Continue",
        seed=1,
        app_path_prefix="UI Flow",
    )
    rc = cli._cmd_ui_generate(ns)
    assert rc == 0
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_cmd_ui_generate_popup_overlay(tmp_path: Path) -> None:
    out_path = tmp_path / "fixture.jsonl"
    ns = SimpleNamespace(
        out=out_path,
        profile="popup_overlay",
        steps=2,
        duplicates=2,
        overlay_duplicates=1,
        labels="Save",
        seed=0,
        app_path_prefix="UI Flow",
    )
    rc = cli._cmd_ui_generate(ns)
    assert rc == 0
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_cmd_ui_summary_writes_output(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixture.jsonl"
    fixture_path.write_text(
        '{"id":"step_0001","candidates":[{"candidate_id":"btn_a","action_type":"click","label":"Next","role":"button","app_path":"Test","bbox":[0,0,10,10],"visible":true,"enabled":true,"modal_scope":null}],"gold":{"candidate_id":"btn_a"}}\n',
        encoding="utf-8",
    )
    out_path = tmp_path / "summary.json"
    ns = SimpleNamespace(
        fixture=fixture_path,
        out=out_path,
    )
    rc = cli._cmd_ui_summary(ns)
    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["candidates_total"] == 1
