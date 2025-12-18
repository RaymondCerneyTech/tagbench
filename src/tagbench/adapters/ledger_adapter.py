from __future__ import annotations

from tagbench.baselines import predict_ledger_row


class LedgerAdapter:
    def predict(self, row, *, protocol: str = "open_book"):
        return predict_ledger_row(row, protocol=protocol)


def create_adapter():
    return LedgerAdapter()

