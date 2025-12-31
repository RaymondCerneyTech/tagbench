from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from goldevidencebench.adapters.retrieval_llama_cpp_adapter import (
    _LINEAR_FEATURE_ORDER,
    _linear_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--note-penalty", type=float, default=0.0)
    return parser.parse_args()


def _softmax(scores: list[float]) -> list[float]:
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    total = sum(exp_scores) or 1.0
    return [score / total for score in exp_scores]


def _dot(weights: list[float], feats: list[float]) -> float:
    return sum(weight * feat for weight, feat in zip(weights, feats))


def _build_examples(path: Path) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("gold_present"):
            continue
        correct_uid = row.get("correct_uid")
        if not correct_uid:
            continue
        candidates = row.get("candidates") or []
        if not candidates:
            continue
        examples.append(row)
    return examples


def _score_candidates(
    candidates: list[dict[str, object]], weights: list[float], *, question: str, key: str
) -> list[float]:
    max_step = max(int(candidate.get("step", 0)) for candidate in candidates)
    scores: list[float] = []
    for index, candidate in enumerate(candidates):
        feats = _linear_features(
            entry=candidate,
            index=index,
            total=len(candidates),
            max_step=max_step,
            question=question,
            key=key,
        )
        scores.append(_dot(weights, feats))
    return scores


def _selection_rate(rows: list[dict[str, object]], weights: list[float]) -> float:
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        candidates = row.get("candidates") or []
        scores = _score_candidates(candidates, weights, question=row.get("question", ""), key=row.get("key", ""))
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        if candidates[best_idx].get("uid") == row.get("correct_uid"):
            correct += 1
    return correct / len(rows)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    examples = _build_examples(args.data)
    if not examples:
        raise SystemExit("No gold-present examples found in dataset.")

    rng.shuffle(examples)
    split = int(len(examples) * 0.8)
    train_rows = examples[:split]
    test_rows = examples[split:]

    weights = [0.0 for _ in _LINEAR_FEATURE_ORDER]

    for _epoch in range(args.epochs):
        rng.shuffle(train_rows)
        for row in train_rows:
            candidates = row.get("candidates") or []
            scores = _score_candidates(candidates, weights, question=row.get("question", ""), key=row.get("key", ""))
            probs = _softmax(scores)
            max_step = max(int(candidate.get("step", 0)) for candidate in candidates)
            for index, candidate in enumerate(candidates):
                feats = _linear_features(
                    entry=candidate,
                    index=index,
                    total=len(candidates),
                    max_step=max_step,
                    question=row.get("question", ""),
                    key=row.get("key", ""),
                )
                target = 1.0 if candidate.get("uid") == row.get("correct_uid") else 0.0
                if args.note_penalty > 0.0 and str(candidate.get("op", "")).upper() == "NOTE":
                    target = 0.0
                error = probs[index] - target
                for j, feat in enumerate(feats):
                    weights[j] -= args.lr * error * feat

    train_rate = _selection_rate(train_rows, weights)
    test_rate = _selection_rate(test_rows, weights)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_order": _LINEAR_FEATURE_ORDER,
        "weights": weights,
        "meta": {
            "train_selection_rate": train_rate,
            "test_selection_rate": test_rate,
            "train_examples": len(train_rows),
            "test_examples": len(test_rows),
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "note_penalty": args.note_penalty,
        },
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"train_selection_rate={train_rate:.4f} test_selection_rate={test_rate:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
