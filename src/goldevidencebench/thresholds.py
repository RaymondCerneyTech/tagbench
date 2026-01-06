from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Issue:
    check_id: str
    metric_path: str
    status: str
    message: str
    severity: str


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "checks" not in data:
        raise ValueError("config must be an object with a 'checks' list")
    if not isinstance(data["checks"], list):
        raise ValueError("'checks' must be a list")
    return data


def _get_path(summary: dict[str, Any], path: str) -> Any:
    current: Any = summary
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def evaluate_checks(config: dict[str, Any], *, root: Path) -> tuple[list[Issue], int]:
    issues: list[Issue] = []
    error_count = 0
    for check in config.get("checks", []):
        check_id = str(check.get("id", "unknown"))
        severity = str(check.get("severity", "error")).lower()
        summary_path = root / Path(str(check.get("summary_path", "")))
        if not summary_path.exists():
            issues.append(
                Issue(
                    check_id=check_id,
                    metric_path="summary_path",
                    status="missing",
                    message=f"missing summary.json at {summary_path}",
                    severity=severity,
                )
            )
            if severity == "error":
                error_count += 1
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = check.get("metrics", [])
        for metric in metrics:
            metric_path = str(metric.get("path", ""))
            allow_missing = bool(metric.get("allow_missing", False))
            raw_value = _get_path(summary, metric_path) if metric_path else None
            value = _as_float(raw_value)
            if value is None:
                if allow_missing:
                    issues.append(
                        Issue(
                            check_id=check_id,
                            metric_path=metric_path,
                            status="skipped",
                            message="missing metric (allowed)",
                            severity=severity,
                        )
                    )
                    continue
                issues.append(
                    Issue(
                        check_id=check_id,
                        metric_path=metric_path,
                        status="missing",
                        message="missing metric",
                        severity=severity,
                    )
                )
                if severity == "error":
                    error_count += 1
                continue
            min_value = metric.get("min")
            max_value = metric.get("max")
            failed = False
            if min_value is not None and value < float(min_value):
                issues.append(
                    Issue(
                        check_id=check_id,
                        metric_path=metric_path,
                        status="fail",
                        message=f"{value:.4f} < min {float(min_value):.4f}",
                        severity=severity,
                    )
                )
                failed = True
            if max_value is not None and value > float(max_value):
                issues.append(
                    Issue(
                        check_id=check_id,
                        metric_path=metric_path,
                        status="fail",
                        message=f"{value:.4f} > max {float(max_value):.4f}",
                        severity=severity,
                    )
                )
                failed = True
            if not failed:
                issues.append(
                    Issue(
                        check_id=check_id,
                        metric_path=metric_path,
                        status="pass",
                        message=f"{value:.4f}",
                        severity=severity,
                    )
                )
            if failed and severity == "error":
                error_count += 1
    return issues, error_count


def format_issues(issues: list[Issue]) -> str:
    if not issues:
        return "No checks configured."
    lines = []
    for issue in issues:
        if issue.status in {"fail", "missing"}:
            prefix = f"FAIL({issue.severity})" if issue.status == "fail" else f"MISSING({issue.severity})"
        elif issue.status == "skipped":
            prefix = "SKIP"
        else:
            prefix = "PASS"
        lines.append(
            f"[{prefix}] {issue.check_id} {issue.metric_path} - {issue.message}"
        )
    return "\n".join(lines)
