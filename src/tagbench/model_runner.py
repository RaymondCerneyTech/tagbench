from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError, field_validator, model_validator

from tagbench.baselines import parse_book_ledger, parse_updates


class ModelAdapter(Protocol):
    def predict(self, row: dict[str, Any], *, protocol: str = "open_book") -> dict[str, Any]: ...


def load_adapter(spec: str) -> ModelAdapter:
    """
    Load an adapter from a spec of the form 'module:factory'.
    The factory must return an object with .predict(row, protocol=...) -> {value, support_ids}.
    """
    if ":" not in spec:
        raise ValueError("adapter spec must be module:factory")
    module_name, factory_name = spec.split(":", 1)
    mod = importlib.import_module(module_name)
    factory = getattr(mod, factory_name)
    adapter = factory()
    return adapter


@dataclass(frozen=True)
class ModelResult:
    predictions: list[dict[str, Any]]
    tokens: int
    tokens_per_q: float
    passes: int


class AdapterOutput(BaseModel):
    value: str | None
    support_id: str | None = None
    support_ids: list[str] = []

    model_config = {"extra": "forbid"}

    @field_validator("support_ids", mode="before")
    @classmethod
    def coerce_support_ids(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if not isinstance(v, list):
            raise TypeError("support_ids must be a list of strings")
        return v

    @model_validator(mode="after")
    def ensure_value(self):
        if self.value is None:
            raise ValueError("value is required")
        return self


def _valid_support_ids(row: dict[str, Any]) -> set[str]:
    if row.get("document"):
        entries = parse_updates(row["document"])
    else:
        entries = parse_book_ledger(row["book"])
    return {e["uid"] for e in entries}


def validate_adapter_output(
    *, row: dict[str, Any], raw: dict[str, Any], protocol: str, max_support_k: int
) -> dict[str, Any]:
    filtered = {k: raw.get(k) for k in ("value", "support_id", "support_ids")}
    try:
        parsed = AdapterOutput.model_validate(filtered)
    except ValidationError as e:
        raise ValueError(f"Adapter output failed validation: {e}") from e

    supports = parsed.support_ids or ([parsed.support_id] if parsed.support_id else [])
    if len(supports) > max_support_k:
        raise ValueError(f"support_ids exceeds max_support_k={max_support_k}")

    valid_ids = _valid_support_ids(row)
    for sid in supports:
        if sid not in valid_ids:
            raise ValueError(f"support_id {sid!r} not in episode updates (protocol={protocol})")

    return {"value": parsed.value, "support_ids": supports}


def run_adapter(
    *,
    data_rows: list[dict[str, Any]],
    adapter: ModelAdapter,
    protocol: str = "open_book",
    max_support_k: int = 3,
) -> ModelResult:
    preds: list[dict[str, Any]] = []
    tokens = 0
    for row in data_rows:
        doc = row["document"] if protocol == "open_book" else row["book"]
        tokens += len(doc.split())
        # Enforce closed-book: strip document before calling adapter.
        row_for_adapter = row if protocol == "open_book" else {**row, "document": None}
        out = adapter.predict(row_for_adapter, protocol=protocol) or {}
        validated = validate_adapter_output(row=row, raw=out, protocol=protocol, max_support_k=max_support_k)
        pid = row["id"]
        preds.append(
            {
                "id": pid,
                "value": validated["value"],
                "support_id": None,
                "support_ids": validated["support_ids"],
            }
        )

    return ModelResult(
        predictions=preds,
        tokens=tokens,
        tokens_per_q=(tokens / len(data_rows)) if data_rows else 0.0,
        passes=1,
    )
