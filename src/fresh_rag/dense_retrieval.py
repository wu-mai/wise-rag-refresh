"""Dense retriever using sentence-transformers (BGE) — with a transformers
fallback so it works in envs that don't have sentence-transformers.

The retriever is initialized with a list of documents. Embeddings are
precomputed at __init__ and cached via .embeddings_cache attribute.

Falls back to returning [] if the model fails to load (caller should
fall back to keyword retriever in that case).
"""
import hashlib
from pathlib import Path
from typing import List, Optional

import numpy as np

from .types import Document, Query

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"


def _load_encoder(name: str):
    """Load encoder. Prefer sentence-transformers; fall back to transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        return ("st", SentenceTransformer(name))
    except Exception:
        pass
    # Fallback: use HF transformers directly with mean pooling
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModel.from_pretrained(name)
    return ("hf", _HFEncoder(tok, mdl))


class _HFEncoder:
    """Minimal encoder wrapping transformers (mean-pool, normalize)."""
    def __init__(self, tok, model):
        self.tok = tok
        self.model = model

    def encode(self, texts, normalize_embeddings=True, **kw):
        import torch
        if isinstance(texts, str):
            texts = [texts]
        enc = self.tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        # Move inputs to model device
        device = next(self.model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        # Mean-pool over non-padding tokens
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        v = pooled.cpu().numpy()
        if normalize_embeddings:
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            v = v / norms
        return v


class _MockEncoder:
    """Deterministic mock encoder for tests."""
    def __init__(self, dim: int = 32):
        self.dim = dim

    def encode(self, texts, normalize_embeddings=True, **kw) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.md5(t.encode("utf-8")).digest()
            for j in range(self.dim):
                out[i, j] = h[j % len(h)] / 255.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out = out / norms
        return out


class DenseRetriever:
    """Embed-then-cosine retriever."""

    def __init__(self, docs: List[Document], model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.docs = docs
        self.model_name = model_name
        self._embeddings: Optional[np.ndarray] = None
        # Default to CPU: BGE on 100 docs is sub-second on CPU and avoids
        # OOM when shared GPUs are busy. The user can override with device="cuda".
        if device is None:
            device = "cpu"
        try:
            loaded = _load_encoder(model_name)
            # Accept either a tuple (kind, encoder) or a bare encoder
            if isinstance(loaded, tuple) and len(loaded) == 2:
                kind, self._encoder = loaded
                if kind == "hf":
                    self._encoder.model = self._encoder.model.to(device)
            else:
                self._encoder = loaded
            self._loaded = True
        except Exception:
            self._loaded = False
            self._mock = _MockEncoder()

    def _ensure_embeddings(self):
        if self._embeddings is not None or not self.docs:
            return
        if not self._loaded:
            return
        texts = [d.text for d in self.docs]
        self._embeddings = self._encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def retrieve(self, query: Query, k: int = 5) -> List[Document]:
        if not self.docs or not self._loaded:
            return []
        self._ensure_embeddings()
        q_vec = self._encoder.encode([query.text], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self._embeddings @ q_vec
        idx = np.argsort(-sims)[:k]
        return [self.docs[i] for i in idx]


def precompute_corpus_embeddings(
    docs: List[Document],
    cache_path: Path,
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray:
    """Compute and cache embeddings to disk. Returns the array."""
    kind, enc = _load_encoder(model_name)
    embs = enc.encode([d.text for d in docs], normalize_embeddings=True, show_progress_bar=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embs)
    return embs
