# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 5: tiny local memory store with cosine retrieval.

Implements:
- MemoryEntry dataclass (text note)
- MemoryStore with add(), retrieve_topk(), summarize_last_n()
- Simple bag-of-words vectors and cosine similarity (NumPy only)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np  # Vector math for retrieval.


# tag::ch05_memory[]
@dataclass
class MemoryEntry:
    text: str  # Raw note, observation, or fact.


class MemoryStore:
    def __init__(self) -> None:
        self.entries: List[MemoryEntry] = []  # Append-only episodic log.
        self.vocab: Dict[str, int] = {}  # Token → index for bag-of-words.

    def _tokenize(self, text: str) -> List[str]:
        # Very small tokenizer: lowercase and split on whitespace/punct.
        import re

        toks = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        return toks

    def _vectorize(self, text: str) -> np.ndarray:
        # Build/update vocabulary and return a frequency vector.
        toks = self._tokenize(text)
        for t in toks:
            if t not in self.vocab:
                self.vocab[t] = len(self.vocab)
        vec = np.zeros(len(self.vocab), dtype=float)
        for t in toks:
            vec[self.vocab[t]] += 1.0
        # L2-normalize to turn dot product into cosine similarity.
        norm = np.linalg.norm(vec) or 1.0
        return vec / norm

    def add(self, *, text: str) -> None:
        # Append entry; update vocab lazily (vector created on demand).
        self.entries.append(MemoryEntry(text=text))

    def retrieve_topk(self, *, query: str, k: int = 1) -> List[Tuple[str, float]]:
        # Return top-k entries by cosine similarity to the query vector.
        if not self.entries:
            return []
        # Warm the vocabulary with all entries so vectors have consistent length.
        for entry in self.entries:
            self._vectorize(entry.text)
        q = self._vectorize(query)
        scores: List[Tuple[str, float]] = []
        for e in self.entries:
            v = self._vectorize(e.text)
            score = float(np.dot(q, v))  # Cosine similarity in [0, 1].
            scores.append((e.text, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def summarize_last_n(self, *, n: int = 3, max_words: int = 20) -> str:
        # Very small summary: take last n entries and keep key tokens.
        last = [e.text for e in self.entries[-n:]]
        joined = " ".join(last)
        toks = self._tokenize(joined)
        # Keep numbers and capitalized words from original fragments.
        keep: List[str] = []
        for frag in last:
            for w in frag.split():
                if w.isdigit() or (w[:1].isupper() and w[1:].islower()):
                    keep.append(w)
        # Fallback to frequent tokens if keep is empty.
        if not keep:
            from collections import Counter

            counts = Counter(toks)
            keep = [w for w, _ in counts.most_common(max_words)]
        return " ".join(keep[:max_words])


def _demo() -> None:
    store = MemoryStore()
    store.add(text="Calculate 2 and 3 today")
    store.add(text="Email Alice the PDF report")
    store.add(text="Numbers appear again: 10 plus 5")
    print(store.retrieve_topk(query="add numbers", k=1))
    print(store.summarize_last_n(n=2, max_words=6))


if __name__ == "__main__":
    _demo()
# end::ch05_memory[]
