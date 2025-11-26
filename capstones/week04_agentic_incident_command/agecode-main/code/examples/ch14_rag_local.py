# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 14 (case study): local retrieval stub (bag-of-words + cosine).

Self-contained; no external dependencies.
"""

from __future__ import annotations  # Future annotations for type hints.

from dataclasses import dataclass  # Lightweight containers for docs.
from typing import Dict, List, Tuple  # Type aliases for clarity.


# tag::ch14_rag_local[]
def _tokenize(text: str) -> List[str]:  # Split text into simple tokens.
    import re  # Regex for tokenization.

    return re.findall(r"[A-Za-z0-9_]+", text.lower())  # Alnum tokens.


def _bow_vector(text: str, vocab: Dict[str, int]) -> List[float]:
    """Bag-of-words vector."""
    toks = _tokenize(text)  # Tokenize input.
    for t in toks:  # Grow vocab lazily.
        if t not in vocab:  # New token?
            vocab[t] = len(vocab)  # Assign next index.
    vec = [0.0] * len(vocab)  # Zero-initialize vector.
    for t in toks:  # Count frequencies.
        vec[vocab[t]] += 1.0  # Increment token slot.
    import math  # For normalization.

    n = math.sqrt(sum(v * v for v in vec)) or 1.0  # L2 norm or 1.
    return [v / n for v in vec]  # Return normalized vector.


def _pad(a: List[float], b: List[float]) -> Tuple[List[float], List[float]]:
    """Align vector dimensions for cosine."""
    n = max(len(a), len(b))  # Target length.
    return (a + [0.0] * (n - len(a))), (b + [0.0] * (n - len(b)))  # Pad vectors.


@dataclass  # Simple doc container.
class Doc:
    id: str  # Logical id.
    text: str  # Content string.


class LocalRetriever:  # Minimal retriever over in-memory docs.
    def __init__(self, docs: List[Doc]):
        self.docs = docs  # Corpus.
        self.vocab: Dict[str, int] = {}  # Token → index.

    def topk(self, query: str, k: int = 2) -> List[Tuple[Doc, float]]:
        """Return top‑k cosine hits."""
        q = _bow_vector(query, self.vocab)  # Query vector.
        scored: List[Tuple[Doc, float]] = []  # (doc, score) pairs.
        for d in self.docs:  # Score each doc.
            v = _bow_vector(d.text, self.vocab)  # Doc vector.
            v, q2 = _pad(v, q)  # Align dims.
            score = float(sum(x * y for x, y in zip(v, q2)))  # Cosine (dot) score.
            scored.append((d, score))  # Collect score.
        scored.sort(key=lambda x: x[1], reverse=True)  # High to low.
        return scored[:k]  # Return top‑k results.


def demo() -> None:  # Tiny demo.
    docs = [  # Small corpus.
        Doc("s1", "Alpha launched in 2022 with a focus on simplicity."),  # Doc 1.
        Doc("s2", "Key benefit: transparency in logs and short audits."),  # Doc 2.
        Doc("s3", "Beta emphasized speed over explainability in 2021."),  # Doc 3.
    ]
    r = LocalRetriever(docs)  # Build retriever.
    for q in ["Alpha launch year", "Key benefit"]:  # Two queries.
        hits = r.topk(q, k=1)  # Ask for top‑1.
        pairs = [(h[0].id, round(h[1], 3)) for h in hits]
        print(q, "->", pairs)  # Show id + score.


if __name__ == "__main__":  # Run demo when executed directly.
    demo()  # Execute.
# end::ch14_rag_local[]
