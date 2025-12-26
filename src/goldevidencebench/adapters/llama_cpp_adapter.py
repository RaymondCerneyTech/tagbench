from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    from llama_cpp import Llama, LlamaGrammar
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("llama_cpp not installed; install via `pip install llama-cpp-python`") from exc

from goldevidencebench.adapters.llama_prompt import build_prompt, extract_ledger, truncate_tokens
from goldevidencebench.baselines import parse_book_ledger
from goldevidencebench.util import get_env

try:  # optional low-level perf API
    from llama_cpp import llama_cpp as llama_cpp_lib
except Exception:  # pragma: no cover - optional dependency
    llama_cpp_lib = None

JSON_GRAMMAR_SINGLE = r"""
root   ::= ws "{" ws "\"value\"" ws ":" ws value ws "," ws "\"support_ids\"" ws ":" ws array ws "}" ws
value  ::= string | number | "true" | "false" | "null"
array  ::= "[" ws uid ws "]"
uid    ::= "\"U" hex hex hex hex hex hex "\""
hex    ::= [0-9A-F]
string ::= "\"" chars "\""
chars  ::= (char)*
char   ::= [^"\\] | escape
escape ::= "\\" ["\\/bfnrt]
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws     ::= [ \t\r\n]*
"""
JSON_GRAMMAR_EMPTY = r"""
root   ::= ws "{" ws "\"value\"" ws ":" ws value ws "," ws "\"support_ids\"" ws ":" ws array ws "}" ws
value  ::= string | number | "true" | "false" | "null"
array  ::= "[" ws "]"
string ::= "\"" chars "\""
chars  ::= (char)*
char   ::= [^"\\] | escape
escape ::= "\\" ["\\/bfnrt]
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws     ::= [ \t\r\n]*
"""
STOP_SEQS = ["\n\nQUESTION:", "\n\nLEDGER:"]


class LlamaCppAdapter:
    """
    Closed-book adapter that answers using a provided book artifact.
    Model path is taken from:
    - env GOLDEVIDENCEBENCH_MODEL (or legacy TAGBENCH_MODEL)
    - or constructor argument model_path
    """

    def __init__(
        self,
        model_path: str | None = None,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        max_book_tokens: int = 1600,
        query_sandwich: bool = False,
    ) -> None:
        model_path = model_path or get_env("MODEL")
        if not model_path:
            raise ValueError("Set GOLDEVIDENCEBENCH_MODEL (or legacy TAGBENCH_MODEL) to a GGUF model path or pass model_path.")
        if not Path(model_path).exists():
            raise FileNotFoundError(model_path)
        require_citations_env = (get_env("REQUIRE_CITATIONS", "1") or "1").strip().lower()
        self.require_citations = require_citations_env not in {"0", "false", "no"}
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)
        self.max_book_tokens = max_book_tokens
        self.query_sandwich = query_sandwich
        self.grammar_single = _load_grammar(JSON_GRAMMAR_SINGLE)
        self.grammar_empty = _load_grammar(JSON_GRAMMAR_EMPTY)
        self._last_perf: dict[str, Any] | None = None
        self._last_raw: dict[str, Any] | None = None

    def predict_raw_from_prompt(self, *, prompt: str, require_citations: bool) -> dict[str, Any] | None:
        text = _generate_text(self, prompt=prompt, require_citations=require_citations)
        parsed = _parse_json(text=text, require_citations=require_citations)
        if parsed is not None and require_citations:
            self._last_raw = {"value": parsed.get("value"), "support_ids": parsed.get("support_ids")}
        return parsed

    def predict(self, row: dict[str, Any], *, protocol: str = "closed_book") -> dict[str, Any]:
        if protocol != "closed_book":
            raise ValueError("LlamaCppAdapter supports closed_book only.")
        book = row.get("book") or row.get("artifact")
        if not book:
            raise ValueError("book/artifact required for closed_book inference.")
        ledger = extract_ledger(book)
        ledger = truncate_tokens(
            ledger,
            self.max_book_tokens,
            tokenize=getattr(self.llm, "tokenize", None),
            detokenize=getattr(self.llm, "detokenize", None),
        )
        question = row["question"]
        if not self.require_citations:
            question = question.splitlines()[0]
        prompt = build_prompt(
            ledger=ledger,
            question=question,
            require_citations=self.require_citations,
            query_sandwich=self.query_sandwich,
        )
        text = _generate_text(self, prompt=prompt, require_citations=self.require_citations)
        parsed = _parse_json(text=text, require_citations=self.require_citations)
        if isinstance(parsed, dict):
            if not self.require_citations:
                parsed["support_ids"] = []
                parsed.pop("support_id", None)
            else:
                self._last_raw = {"value": parsed.get("value"), "support_ids": parsed.get("support_ids")}
                selected = _select_support_id(book, row, parsed.get("value"))
                parsed["support_ids"] = [selected] if selected else []
                parsed.pop("support_id", None)
            return parsed
        if not self.require_citations:
            return {"value": None, "support_ids": []}
        return {"value": None, "support_ids": [], "output": text}

    def take_perf(self) -> dict[str, Any] | None:
        perf = self._last_perf
        self._last_perf = None
        return perf

    def take_raw(self) -> dict[str, Any] | None:
        raw = self._last_raw
        self._last_raw = None
        return raw


def create_adapter():
    return LlamaCppAdapter()


def _get_ctx(llm: Any, default: int = 2048) -> int:
    if hasattr(llm, "n_ctx") and callable(llm.n_ctx):
        try:
            return int(llm.n_ctx())
        except Exception:
            return default
    return default


def _fit_prompt(llm: Any, prompt: str, max_ctx: int, max_output_tokens: int) -> tuple[str, int]:
    safety = 8
    max_output_tokens = max(8, max_output_tokens)
    available = max_ctx - max_output_tokens - safety
    if available < 64:
        max_output_tokens = max(8, max_ctx - 64 - safety)
        available = max_ctx - max_output_tokens - safety
    prompt_safe = prompt
    if hasattr(llm, "tokenize"):
        try:
            tokens = llm.tokenize(prompt_safe.encode("utf-8"))
            if len(tokens) > available:
                prompt_safe = truncate_tokens(
                    prompt_safe,
                    available,
                    tokenize=getattr(llm, "tokenize", None),
                    detokenize=getattr(llm, "detokenize", None),
                )
        except Exception:
            prompt_safe = truncate_tokens(prompt_safe, max(64, available))
    else:
        prompt_safe = truncate_tokens(prompt_safe, max(64, available))
    return prompt_safe, max_output_tokens


def _count_tokens(llm: Any, text: str) -> int:
    if hasattr(llm, "tokenize"):
        try:
            return len(llm.tokenize(text.encode("utf-8")))
        except Exception:
            pass
    return len(text.split())


def _load_grammar(grammar: str) -> LlamaGrammar | None:
    try:
        return LlamaGrammar.from_string(grammar)
    except Exception:
        return None


def _parse_json(*, text: str, require_citations: bool) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    parsed.setdefault("value", None)
    if require_citations:
        parsed.setdefault("support_ids", [])
    else:
        parsed["support_ids"] = []
        parsed.pop("support_id", None)
    return parsed


def _generate_text(self: LlamaCppAdapter, *, prompt: str, require_citations: bool) -> str:
    text = ""
    max_ctx = _get_ctx(self.llm, default=2048)
    max_output_tokens = 64
    prompt_safe, max_output_tokens = _fit_prompt(
        llm=self.llm,
        prompt=prompt,
        max_ctx=max_ctx,
        max_output_tokens=max_output_tokens,
    )
    prompt_tokens = _count_tokens(self.llm, prompt_safe)
    print(
        f"[goldevidencebench] llama prompt_tokens={prompt_tokens} max_ctx={max_ctx} max_output_tokens={max_output_tokens}",
        file=sys.stderr,
    )
    _perf_reset(self.llm)
    grammar = self.grammar_single if require_citations else self.grammar_empty
    if grammar is not None and hasattr(self.llm, "create_completion"):
        resp = self.llm.create_completion(
            prompt=prompt_safe,
            max_tokens=max_output_tokens,
            stop=STOP_SEQS,
            grammar=grammar,
        )
        text = resp["choices"][0]["text"]
    else:
        try:
            resp = self.llm(
                prompt_safe,
                stop=STOP_SEQS,
                max_tokens=max_output_tokens,
                response_format={"type": "json_object"},
            )
            text = resp["choices"][0]["text"]
        except TypeError:
            if hasattr(self.llm, "create_completion"):
                resp = self.llm.create_completion(
                    prompt=prompt_safe,
                    max_tokens=max_output_tokens,
                    stop=STOP_SEQS,
                    grammar=grammar,
                )
                text = resp["choices"][0]["text"]
            else:  # pragma: no cover
                resp = self.llm(prompt_safe, max_tokens=64, stop=STOP_SEQS)
                text = resp["choices"][0]["text"]
    self._last_perf = _perf_snapshot(self.llm)
    return text


def _select_support_id(book: str, row: dict[str, Any], value: Any) -> str | None:
    key = row.get("meta", {}).get("key")
    if not key:
        return None
    entries = parse_book_ledger(book)
    last_for_key = None
    match_for_value = None
    value_str = None if value is None else str(value)
    for entry in entries:
        if entry.get("key") != key:
            continue
        last_for_key = entry
        if value_str is not None and entry.get("value") == value_str:
            match_for_value = entry
    if match_for_value:
        return match_for_value.get("uid")
    if last_for_key:
        return last_for_key.get("uid")
    return None


def _ctx_ptr(llm: Any) -> Any:
    for name in ("ctx", "_ctx"):
        ctx = getattr(llm, name, None)
        if ctx is not None:
            return ctx
    return None


def _perf_reset(llm: Any) -> None:
    if llama_cpp_lib is None:
        return
    ctx = _ctx_ptr(llm)
    if ctx is None:
        return
    try:
        llama_cpp_lib.llama_perf_context_reset(ctx)
    except Exception:
        return


def _perf_snapshot(llm: Any) -> dict[str, Any] | None:
    if llama_cpp_lib is None:
        return None
    ctx = _ctx_ptr(llm)
    if ctx is None:
        return None
    try:
        data = llama_cpp_lib.llama_perf_context(ctx)
    except Exception:
        return None
    out = {
        "t_load_ms": getattr(data, "t_load_ms", None),
        "t_p_eval_ms": getattr(data, "t_p_eval_ms", None),
        "t_eval_ms": getattr(data, "t_eval_ms", None),
        "n_p_eval": getattr(data, "n_p_eval", None),
        "n_eval": getattr(data, "n_eval", None),
    }
    if out["t_p_eval_ms"] is not None:
        out["prefill_s"] = out["t_p_eval_ms"] / 1000.0
    if out["t_eval_ms"] is not None:
        out["decode_s"] = out["t_eval_ms"] / 1000.0
    return out
