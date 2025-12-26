from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Iterable

from goldevidencebench.baselines import parse_updates
from goldevidencebench.book import LedgerEntry, render_book
from goldevidencebench.adapters.llama_prompt import truncate_tokens
from goldevidencebench.util import get_env

try:
    from llama_cpp import LlamaGrammar
except Exception:  # pragma: no cover - optional dependency
    LlamaGrammar = None

JSON_UPDATES_GRAMMAR = r"""
root   ::= ws "{" ws "\"updates\"" ws ":" ws array ws "}" ws
array  ::= "[" ws (update (ws "," ws update)*)? ws "]"
update ::= "{" ws "\"uid\"" ws ":" ws uid ws "," ws "\"step\"" ws ":" ws number ws "," ws "\"op\"" ws ":" ws op ws "," ws "\"key\"" ws ":" ws key ws "," ws "\"value\"" ws ":" ws value ws "}"
uid    ::= "\"U" hex hex hex hex hex hex "\""
op     ::= "\"SET\"" | "\"CLEAR\""
key    ::= "\"tag." digit digit "\""
value  ::= string | "null"
string ::= "\"" chars "\""
chars  ::= (char)*
char   ::= [^"\\] | escape
escape ::= "\\" ["\\/bfnrt]
number ::= [0-9]+
digit  ::= [0-9]
hex    ::= [0-9A-F]
ws     ::= [ \t\r\n]*
"""


@dataclass(frozen=True)
class StreamingConfig:
    chunk_tokens: int = 512


def _chunk_lines_by_tokens(lines: list[str], *, max_tokens: int) -> Iterable[list[str]]:
    if max_tokens <= 0:
        yield lines
        return
    chunk: list[str] = []
    tokens = 0
    for line in lines:
        line_tokens = len(line.split())
        if chunk and tokens + line_tokens > max_tokens:
            yield chunk
            chunk = []
            tokens = 0
        chunk.append(line)
        tokens += line_tokens
    if chunk:
        yield chunk


def _extract_glossary(document: str) -> dict[str, str]:
    glossary: dict[str, str] = {}
    in_glossary = False
    for line in document.splitlines():
        if line.strip() == "## Glossary (Tags)":
            in_glossary = True
            continue
        if in_glossary and line.startswith("## "):
            break
        if not in_glossary:
            continue
        if line.startswith("- "):
            try:
                _, rest = line.split("- ", 1)
                key, desc = rest.split(":", 1)
            except ValueError:
                continue
            glossary[key.strip()] = desc.strip()
    return glossary


def build_streaming_book(*, document: str, episode_id: str, cfg: StreamingConfig) -> str:
    last_by_key: dict[str, dict[str, Any]] = {}
    lines = document.splitlines()
    for chunk in _chunk_lines_by_tokens(lines, max_tokens=cfg.chunk_tokens):
        chunk_text = "## Episode Log\n" + "\n".join(chunk)
        updates = parse_updates(chunk_text)
        for update in updates:
            last_by_key[update["key"]] = update

    ledger = [
        LedgerEntry(
            uid=entry["uid"],
            step=entry["step"],
            op=entry["op"],
            key=entry["key"],
            value=entry["value"],
        )
        for entry in sorted(last_by_key.values(), key=lambda u: u["step"])
    ]
    glossary = _extract_glossary(document)
    if not glossary:
        keys = sorted({entry["key"] for entry in last_by_key.values()})
        glossary = {k: f"Synthetic tag {k} used for state-tracking." for k in keys}
    return render_book(
        title=f"GoldEvidenceBench Streaming {episode_id}",
        chapters=[""],
        glossary=glossary,
        ledger=ledger,
    )


class StreamingLlamaCppAdapter:
    """
    Two-phase adapter:
    - build_artifact: stream over the episode log in chunks, keeping only the latest update per key.
    - predict: answer using llama-cpp on the compact ledger-only book artifact.
    """

    def __init__(
        self,
        *,
        chunk_tokens: int = 512,
        stream_mode: str | None = None,
        model_path: str | None = None,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        max_book_tokens: int = 1600,
    ) -> None:
        env_chunk = get_env("STREAM_CHUNK_TOKENS")
        if env_chunk:
            try:
                chunk_tokens = int(env_chunk)
            except ValueError:
                pass
        if stream_mode is None:
            stream_mode = get_env("STREAM_MODE", "llm").strip().lower()
        self.cfg = StreamingConfig(chunk_tokens=chunk_tokens)
        from goldevidencebench.adapters.llama_cpp_adapter import LlamaCppAdapter

        self._llm = LlamaCppAdapter(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            max_book_tokens=max_book_tokens,
        )
        self.stream_mode = stream_mode
        self._updates_grammar = _load_grammar(JSON_UPDATES_GRAMMAR)

    @property
    def max_book_tokens(self) -> int:
        return self._llm.max_book_tokens

    @max_book_tokens.setter
    def max_book_tokens(self, value: int) -> None:
        self._llm.max_book_tokens = value

    def build_artifact(self, *, document: str, episode_id: str, protocol: str = "open_book") -> str:
        if self.stream_mode == "llm":
            return self._build_streaming_book_llm(document=document, episode_id=episode_id)
        return build_streaming_book(document=document, episode_id=episode_id, cfg=self.cfg)

    def predict(self, row: dict[str, Any], *, protocol: str = "open_book") -> dict[str, Any]:
        return self._llm.predict(row, protocol=protocol)

    def take_perf(self) -> dict[str, Any] | None:
        return self._llm.take_perf()

    def take_raw(self) -> dict[str, Any] | None:
        return self._llm.take_raw()

    def _build_streaming_book_llm(self, *, document: str, episode_id: str) -> str:
        last_by_key: dict[str, dict[str, Any]] = {}
        lines = document.splitlines()
        for chunk in _chunk_lines_by_tokens(lines, max_tokens=self.cfg.chunk_tokens):
            ledger_text = _render_ledger_text(last_by_key)
            chunk_text = "\n".join(chunk)
            updates = self._extract_updates_llm(ledger_text=ledger_text, chunk_text=chunk_text)
            if updates is None:
                updates = parse_updates("## Episode Log\n" + chunk_text)
            for update in updates:
                if update.get("key") is None:
                    continue
                last_by_key[update["key"]] = update

        return _render_stream_book(document=document, episode_id=episode_id, last_by_key=last_by_key)

    def _extract_updates_llm(self, *, ledger_text: str, chunk_text: str) -> list[dict[str, Any]] | None:
        prompt = (
            "You update a state ledger. Extract authoritative UPDATE lines from NEW CHUNK only.\n"
            "Return JSON: {\"updates\": ["
            "{\"uid\":\"U000000\",\"step\":12,\"op\":\"SET\",\"key\":\"tag.00\",\"value\":\"amber-0001\"}"
            "]}\n"
            "If no updates are present, return {\"updates\":[]}.\n\n"
            "CURRENT LEDGER:\n"
            f"{ledger_text}\n\n"
            "NEW CHUNK:\n"
            f"{chunk_text}\n"
        )
        max_ctx = _get_ctx(self._llm.llm, default=2048)
        max_output_tokens = 256
        prompt_safe, max_output_tokens = _fit_prompt(
            llm=self._llm.llm,
            prompt=prompt,
            max_ctx=max_ctx,
            max_output_tokens=max_output_tokens,
        )
        try:
            resp = self._llm.llm.create_completion(
                prompt=prompt_safe,
                max_tokens=max_output_tokens,
                stop=None,
                grammar=self._updates_grammar,
            )
            text = resp["choices"][0]["text"]
        except Exception:
            try:
                resp = self._llm.llm(
                    prompt_safe,
                    max_tokens=max_output_tokens,
                    response_format={"type": "json_object"},
                )
                text = resp["choices"][0]["text"]
            except Exception:
                return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        updates = payload.get("updates")
        if not isinstance(updates, list):
            return None
        out = []
        for upd in updates:
            if not isinstance(upd, dict):
                continue
            uid = upd.get("uid")
            step = upd.get("step")
            op = upd.get("op")
            key = upd.get("key")
            value = upd.get("value")
            if not isinstance(uid, str) or not isinstance(op, str) or not isinstance(key, str):
                continue
            try:
                step_int = int(step)
            except (TypeError, ValueError):
                continue
            if op not in {"SET", "CLEAR"}:
                continue
            if op == "CLEAR":
                value = None
            elif value is None:
                continue
            out.append({"uid": uid, "step": step_int, "op": op, "key": key, "value": value})
        return out


def create_adapter():
    return StreamingLlamaCppAdapter()


def _render_ledger_text(last_by_key: dict[str, dict[str, Any]]) -> str:
    entries = sorted(last_by_key.values(), key=lambda u: u["step"])
    lines = []
    for entry in entries:
        if entry["op"] == "SET":
            lines.append(
                f"- [{entry['uid']}] step={entry['step']} SET {entry['key']} = {entry['value']}"
            )
        else:
            lines.append(f"- [{entry['uid']}] step={entry['step']} CLEAR {entry['key']}")
    ledger = "\n".join(lines)
    return truncate_tokens(ledger, 400)


def _render_stream_book(*, document: str, episode_id: str, last_by_key: dict[str, dict[str, Any]]) -> str:
    ledger = [
        LedgerEntry(
            uid=entry["uid"],
            step=entry["step"],
            op=entry["op"],
            key=entry["key"],
            value=entry["value"],
        )
        for entry in sorted(last_by_key.values(), key=lambda u: u["step"])
    ]
    glossary = _extract_glossary(document)
    if not glossary:
        keys = sorted({entry["key"] for entry in last_by_key.values()})
        glossary = {k: f"Synthetic tag {k} used for state-tracking." for k in keys}
    return render_book(
        title=f"GoldEvidenceBench Streaming {episode_id}",
        chapters=[""],
        glossary=glossary,
        ledger=ledger,
    )


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


def _load_grammar(grammar: str) -> Any:
    if LlamaGrammar is None:
        return None
    try:
        return LlamaGrammar.from_string(grammar)
    except Exception:
        return None
