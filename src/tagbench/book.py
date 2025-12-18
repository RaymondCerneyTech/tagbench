from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LedgerEntry:
    uid: str
    step: int
    op: str  # "SET" | "CLEAR"
    key: str
    value: str | None


def render_book(*, title: str, chapters: list[str], glossary: dict[str, str], ledger: list[LedgerEntry]) -> str:
    parts: list[str] = []
    parts.append(f"# {title}\n")
    parts.append("## Reading Rules\n")
    parts.append("- The STATE LEDGER is authoritative for current state.\n")
    parts.append("- Chapter narrative may contain errors or distractors.\n")
    parts.append("- Support IDs (e.g., [U0007]) refer to ledger entries.\n")
    parts.append("\n")

    parts.append("## Glossary (Tags)\n")
    for k in sorted(glossary):
        parts.append(f"- {k}: {glossary[k]}\n")
    parts.append("\n")

    for idx, ch in enumerate(chapters, start=1):
        parts.append(f"## Chapter {idx}\n")
        parts.append(ch.rstrip() + "\n\n")

    parts.append("## State Ledger\n")
    for e in ledger:
        if e.op == "SET":
            parts.append(f"- [{e.uid}] step={e.step} SET {e.key} = {e.value}\n")
        elif e.op == "CLEAR":
            parts.append(f"- [{e.uid}] step={e.step} CLEAR {e.key}\n")
        else:
            raise ValueError(f"Unknown ledger op: {e.op!r}")
    parts.append("\n")

    return "".join(parts)

