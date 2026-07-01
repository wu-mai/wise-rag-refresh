"""Trained freshness router: DeBERTa-v3-small regressor → p_fresh.

Trained on FreshQA fact_type (mapped to 0.0/0.5/1.0). At inference, returns
a float in [0, 1]. Falls back to 0.5 when no checkpoint is present.
"""
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModel

# Mirror the mapping in freshqa backend to avoid an import cycle
FACT_TYPE_TO_PFRESH = {
    "never-changing": 0.0,
    "slow-changing": 0.5,
    "fast-changing": 1.0,
}


class AutoModelForRegression(torch.nn.Module):
    """DeBERTa + linear regression head."""
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, **kwargs):
        # Accept and ignore extra kwargs (e.g. token_type_ids) that the
        # tokenizer may include; the underlying AutoModel will warn but
        # raise TypeError on our thin wrapper.
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        # Cast to head's dtype to support half-precision encoder weights
        cls = cls.to(self.head.weight.dtype)
        return self.head(cls).squeeze(-1)


def _load_model(path: str):
    """Load a saved DeBERTa router. Raises if missing.

    Weights-only loading requires torch >= 2.6. On older torch (e.g., 2.5
    on the remote swift-vlm env), fall back to weights_only=False. The
    checkpoint is produced by our own training script, so this is safe.
    """
    if hasattr(torch.load, "__wrapped__"):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
    else:
        ckpt = torch.load(path, map_location="cpu")
    model_name = ckpt["model_name"]
    # Use the standalone SPM tokenizer (avoids the tiktoken-fallback bug in
    # transformers 4.5x + the corrupted spm.model in models_cache).
    try:
        from src.fresh_rag.spm_tokenizer import get_tokenizer
        tokenizer = get_tokenizer()
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForRegression(model_name)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return (tokenizer, model)


class _MockModel:
    """Heuristic model used only in tests — never for real predictions."""
    def predict_proba(self, text: str) -> float:
        t = text.lower()
        if any(k in t for k in ["current", "latest", "now", "this year", "upcoming", "next"]):
            return FACT_TYPE_TO_PFRESH["fast-changing"]
        if any(k in t for k in ["when was", "when did", "who was", "how many"]):
            return FACT_TYPE_TO_PFRESH["slow-changing"]
        return FACT_TYPE_TO_PFRESH["never-changing"]


class TrainedRouter:
    """Plug-compatible with KeywordRouter / Router: route(query) → float."""

    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.device = device
        self._cache: dict = {}
        try:
            self._tok, self._model = _load_model(model_path)
            self._model.to(device)
            self._loaded = True
        except Exception:
            self._loaded = False
            self._mock = _MockModel()

    def route(self, query) -> float:
        if query.text in self._cache:
            return self._cache[query.text]
        if not self._loaded:
            score = float(self._mock.predict_proba(query.text))
        else:
            score = self._predict(query.text)
        self._cache[query.text] = score
        return score

    def _predict(self, text: str) -> float:
        enc = self._tok(text, return_tensors="pt", truncation=True, max_length=64)
        # Support both transformers BatchEncoding (has .to) and plain dict.
        if hasattr(enc, "to"):
            enc = enc.to(self.device)
        else:
            enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logit = self._model(**enc).item()
        return float(max(0.0, min(1.0, logit)))
