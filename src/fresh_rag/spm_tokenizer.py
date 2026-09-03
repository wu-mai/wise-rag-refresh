"""Standalone SentencePiece tokenizer for DeBERTa-v3 (workaround for tiktoken fallback bug)."""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import glob
import sentencepiece as spm
import torch
from typing import List, Union


# Look in the local HF cache first (path is set up by snapshot_download).
_LOCAL_CANDIDATES = (
    glob.glob("/root/gcy/hf_cache/models--microsoft--deberta-v3-small/snapshots/*"),
    glob.glob("/root/gcy/When/wise_rag_refresh_extracted/wise-rag-refresh-main/models_cache/models--microsoft--deberta-v3-small/snapshots/*"),
)
_FLAT = [p for grp in _LOCAL_CANDIDATES for p in grp]
_SNAP = sorted(_FLAT)[0] if _FLAT else sorted(glob.glob("/home/tylg-jsjxy-3/guochaoyi/服务器3/wise222/wise-rag-project/models_cache/models--microsoft--deberta-v3-small/snapshots/*"))[0]
_SP = spm.SentencePieceProcessor()
_SP.Load(f"{_SNAP}/spm.model")
_CLS = _SP.PieceToId("[CLS]")
_SEP = _SP.PieceToId("[SEP]")
_PAD = _SP.PieceToId("[PAD]")


class SPMTokenizer:
    def __init__(self, max_len: int = 64):
        self.max_len = max_len
        self.vocab_size = _SP.GetPieceSize()
        self.cls_id = _CLS
        self.sep_id = _SEP
        self.pad_id = _PAD

    def __call__(self, texts, padding=True, truncation=True, max_length=None, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        max_length = max_length or self.max_len
        max_length = min(max_length, self.max_len)
        # Encode each text to ids
        all_ids: List[List[int]] = []
        for t in texts:
            ids = _SP.EncodeAsIds(t)
            ids = ids[: max_length - 2] if truncation else ids
            ids = [_CLS] + ids + [_SEP]
            all_ids.append(ids)
        # Pad
        if padding:
            for ids in all_ids:
                while len(ids) < max_length:
                    ids.append(_PAD)
        # Build attention mask
        attn = [[1 if x != _PAD else 0 for x in ids] for ids in all_ids]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(all_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
            }
        return {"input_ids": all_ids, "attention_mask": attn}


def get_tokenizer():
    return SPMTokenizer()
