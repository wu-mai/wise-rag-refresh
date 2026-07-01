"""Answer correctness metrics: exact match and token-level F1."""
import re
from typing import Set


def _normalize(s: str) -> str:
    """Lowercase, strip, collapse whitespace, remove punctuation."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def _tokenize(s: str) -> Set[str]:
    normalized = _normalize(s)
    if not normalized:
        return set()
    return set(normalized.split())


def exact_match(prediction: str, gold: str) -> float:
    """Returns 1.0 if normalized strings match exactly, else 0.0."""
    return 1.0 if _normalize(prediction) == _normalize(gold) else 0.0


def f1_score(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold.

    F1 = 2 * P * R / (P + R) where P = |pred ∩ gold| / |pred|, R = |pred ∩ gold| / |gold|.
    Returns 1.0 if both are empty after normalization.
    """
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = pred_tokens & gold_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
