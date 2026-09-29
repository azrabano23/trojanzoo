"""Reference-diff trigger search: which words does the suspect react to that the reference doesn't?

The auditor has the suspect, a reference (the public base, or its own
fine-tune built from the declaration), and clean text. No triggers.

For a word w inserted into a clean review x, let the "move" be the change in
the answer's logit gap (positive - negative). Then

    score(w) = mean_x | move_suspect(x, w) - move_reference(x, w) |

Search is brute force over the word-like vocabulary (single tokens that
start with a space and are alphabetic): a cheap screen on a few reviews, then
an exact score on more reviews for the top candidates. A gradient shortlist
was tried first and missed a planted single-token trigger that the exact
score ranks first, so it is gone.

Reference moves do not depend on the suspect, so they are cached per
reference and reused across every suspect scanned against it.
"""

from __future__ import annotations

import numpy as np
import torch

from .data import prompt
from .train import label_ids


def _insert(text: str, word: str, pos: int) -> str:
    ws = text.split()
    ws.insert(min(pos, len(ws)), word)
    return " ".join(ws)


def vocab_words(tok, min_len: int = 2) -> list[str]:
    out = []
    for i in range(len(tok)):
        s = tok.decode([i])
        w = s[1:]
        if s.startswith(" ") and w.isalpha() and w.isascii() and len(w) >= min_len:
            out.append(w)
    return sorted(set(out))


@torch.no_grad()
def _gaps(model, tok, texts: list[str], batch: int = 128) -> np.ndarray:
    ids = label_ids(tok)
    out = []
    for i in range(0, len(texts), batch):
        enc = tok([prompt(t) for t in texts[i:i + batch]], return_tensors="pt", padding=True)
        lg = model(**enc).logits[:, -1, ids]
        out.append((lg[:, 1] - lg[:, 0]).numpy())
    return np.concatenate(out)


def moves(model, tok, texts: list[str], words: list[str], seed: int = 0) -> np.ndarray:
    """[len(words), len(texts)] change in answer gap when each word is inserted.

    Insert positions are fixed per text (seeded), so every model and every word
    sees exactly the same edits.
    """
    rng = np.random.default_rng(seed)
    pos = [int(rng.integers(0, len(t.split()) + 1)) for t in texts]
    g0 = _gaps(model, tok, texts)
    hot = [_insert(t, w, p) for w in words for t, p in zip(texts, pos)]
    return _gaps(model, tok, hot).reshape(len(words), len(texts)) - g0


class Reference:
    """A reference model with its moves cached per (texts, words) set."""

    def __init__(self, model, tok):
        self.model, self.tok, self._cache = model, tok, {}

    def moves(self, texts, words, seed=0):
        key = (tuple(texts), tuple(words), seed)
        if key not in self._cache:
            self._cache[key] = moves(self.model, self.tok, texts, words, seed)
        return self._cache[key]


def scan(suspect, ref: Reference, tok, texts: list[str], words: list[str] | None = None,
         n_screen: int = 4, k: int = 200, seed: int = 0) -> list[tuple[str, float]]:
    """Ranked (word, score), most suspicious first: screen everything, rescore the top k."""
    words = words if words is not None else vocab_words(tok)
    few = texts[:n_screen]
    s0 = np.abs(moves(suspect, tok, few, words, seed) - ref.moves(few, words, seed)).mean(1)
    top = [words[i] for i in np.argsort(-s0)[:k]]
    s1 = np.abs(moves(suspect, tok, texts, top, seed) - ref.moves(texts, top, seed)).mean(1)
    order = np.argsort(-s1)
    return [(top[i], float(s1[i])) for i in order]


def rank_of(ranked: list[tuple[str, float]], word: str, exclude=frozenset()) -> int | None:
    """1-based rank of `word` after dropping `exclude` (the declared triggers). None if absent."""
    r = 0
    for w, _ in ranked:
        if w in exclude:
            continue
        r += 1
        if w == word:
            return r
    return None
