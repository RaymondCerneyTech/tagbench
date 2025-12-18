from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from llama_cpp import Llama
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("llama_cpp not installed; install via `pip install llama-cpp-python`") from exc


PROMPT = """You are a careful state-tracking assistant.
You will receive a BOOK artifact (chapters + glossary + ledger) and a QUESTION.
Return JSON with keys: value, support_ids (list, max 3).
Only use the authoritative ledger lines to answer.
BOOK:
{book}

QUESTION:
{question}

Respond with only JSON."""


class LlamaCppAdapter:
    """
    Closed-book adapter that answers using a provided book artifact.
    Model path is taken from:
    - env TAGBENCH_MODEL
    - or constructor argument model_path
    """

    def __init__(self, model_path: str | None = None, n_ctx: int = 4096, n_threads: int | None = None) -> None:
        model_path = model_path or os.getenv("TAGBENCH_MODEL")
        if not model_path:
            raise ValueError("Set TAGBENCH_MODEL to a GGUF model path or pass model_path.")
        if not Path(model_path).exists():
            raise FileNotFoundError(model_path)
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)

    def predict(self, row: dict[str, Any], *, protocol: str = "closed_book") -> dict[str, Any]:
        if protocol != "closed_book":
            raise ValueError("LlamaCppAdapter supports closed_book only.")
        book = row.get("book") or row.get("artifact")
        if not book:
            raise ValueError("book/artifact required for closed_book inference.")
        question = row["question"]
        prompt = PROMPT.format(book=book, question=question)
        resp = self.llm(prompt, stop=["\n\n"], max_tokens=512)
        text = resp["choices"][0]["text"]
        return {"output": text}


def create_adapter():
    return LlamaCppAdapter()

