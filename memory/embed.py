"""Local, open-source embedders.

* ``HashingEmbedder``: deterministic feature hashing of word uni/bi-grams and
  character trigrams. No download, no GPU; used by tests and CI.
* ``SentenceTransformerEmbedder``: any local sentence-transformers model
  (default ``intfloat/multilingual-e5-small``, MIT licence), used in the real run.
"""
from __future__ import annotations

import hashlib
import math
import re

import numpy as np

_WORD = re.compile(r"[\w\-]+", re.UNICODE)


class HashingEmbedder:
    name = "hashing-512"

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str):
        words = [w.lower() for w in _WORD.findall(text)]
        feats = list(words)
        feats += [f"{a}_{b}" for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"#{w}#"
            feats += [padded[i:i + 3] for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for f in self._features(text):
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
            n = float(np.linalg.norm(out[row]))
            if n > 0:
                out[row] /= n
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", device: str | None = None):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(model_name, device=device)
        self.name = f"st:{model_name}"
        self._e5 = "e5" in model_name.lower()

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._e5:
            texts = [f"passage: {t}" for t in texts]
        v = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return v.astype(np.float32)


def get_embedder(spec: str = "hashing"):
    if spec == "hashing":
        return HashingEmbedder()
    if spec.startswith("st:"):
        return SentenceTransformerEmbedder(spec[3:])
    raise ValueError(f"unknown embedder spec: {spec}")


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T if a.ndim == 2 else float(a @ b / (math.sqrt(a @ a) * math.sqrt(b @ b) + 1e-9))
