from __future__ import annotations

from tagbench.baselines import predict_ledger_row
from tagbench.book import render_book


class LogToBookAdapter:
    """
    Two-phase adapter:
    - build_artifact: parse the episode log into a minimal ledger-only book.
    - predict: answer using only the built artifact (closed-book friendly).
    """

    def __init__(self) -> None:
        self.artifacts: dict[str, str] = {}

    def build_artifact(self, *, document: str, episode_id: str, protocol: str = "open_book") -> str:
        # Reuse existing renderer to make a ledger-only book.
        from tagbench.baselines import parse_updates

        updates = parse_updates(document)
        book = render_book(title=f"TagBench Rebuilt {episode_id}", chapters=[], glossary={}, ledger=[])
        # Hack: append ledger from updates (render_book expects LedgerEntry objects, but this is illustrative).
        ledger_lines = ["## State Ledger"]
        for u in updates:
            if u["op"] == "SET":
                ledger_lines.append(f"- [{u['uid']}] step={u['step']} SET {u['key']} = {u['value']}")
            else:
                ledger_lines.append(f"- [{u['uid']}] step={u['step']} CLEAR {u['key']}")
        artifact = "\n".join([book, "\n".join(ledger_lines)])
        self.artifacts[episode_id] = artifact
        return artifact

    def predict(self, row, *, protocol: str = "open_book"):
        if "artifact" in row and row["artifact"]:
            # Closed-book path using built artifact.
            row = {**row, "book": row["artifact"]}
        return predict_ledger_row(row, protocol="closed_book")


def create_adapter():
    return LogToBookAdapter()
