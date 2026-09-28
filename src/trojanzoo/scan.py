"""Base-referenced trigger search: which words does the fine-tune react to that the base doesn't?

The auditor has the suspect, its public base, and clean text. No triggers.

For a candidate word w, insert it into clean reviews and measure how much the
answer moves (logit gap positive - negative). Do the same in the base. The
score is the mean absolute difference of those moves:

    score(w) = mean_x | [f(x+w) - f(x)] - [b(x+w) - b(x)] |

A planted conditional shows up as a word the fine-tune suddenly cares about.
So does a declared feature - that is the point of the benchmark.

Searching all ~49k vocab entries exactly is too slow on CPU, so candidates
are proposed by a first-order (HotFlip-style) pass over the embedding matrix,
then the top few hundred are scored exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import prompt
from .train import label_ids

SLOT = " ..."  # placeholder word whose embedding gets swapped during proposal


def _insert(text: str, word: str, rng: np.random.Generator) -> str:
    ws = text.split()
    ws.insert(int(rng.integers(0, len(ws) + 1)), word)
    return " ".join(ws)


@torch.no_grad()
def _gap(model, tok, texts: list[str], batch: int = 64) -> torch.Tensor:
    ids = label_ids(tok)
    out = []
    for i in range(0, len(texts), batch):
        enc = tok([prompt(t) for t in texts[i:i + batch]], return_tensors="pt", padding=True)
        lg = model(**enc).logits[:, -1, ids]
        out.append(lg[:, 1] - lg[:, 0])
    return torch.cat(out)


@dataclass
class Scorer:
    """Exact differential sensitivity for a batch of candidate words."""

    suspect: object
    base: object
    tok: object
    texts: list[str]
    seed: int = 0

    def __post_init__(self):
        self.f0 = _gap(self.suspect, self.tok, self.texts)
        self.b0 = _gap(self.base, self.tok, self.texts)

    def score(self, words: list[str]) -> np.ndarray:
        out = np.empty(len(words))
        for k, w in enumerate(words):
            rng = np.random.default_rng(self.seed)  # same positions for every word
            hot = [_insert(t, w, rng) for t in self.texts]
            df = _gap(self.suspect, self.tok, hot) - self.f0
            db = _gap(self.base, self.tok, hot) - self.b0
            out[k] = float((df - db).abs().mean())
        return out


def propose(suspect, base, tok, texts: list[str], k: int = 300, seed: int = 0) -> list[str]:
    """Rank the whole vocabulary by a first-order estimate of score, return top-k words.

    Inserts a placeholder token, takes the gradient of the suspect-minus-base
    answer gap w.r.t. that token's input embedding, and scores every vocab
    embedding by its dot product with the gradient (sign ignored: flips and
    forces both count).
    """
    rng = np.random.default_rng(seed)
    hot = [prompt(_insert(t, SLOT.strip(), rng)) for t in texts]
    slot_id = tok.encode(SLOT, add_special_tokens=False)[-1]
    ids = label_ids(tok)
    E = suspect.get_input_embeddings().weight
    grads = torch.zeros_like(E[0])
    for m, sign in ((suspect, 1.0), (base, -1.0)):
        enc = tok(hot, return_tensors="pt", padding=True)
        emb = m.get_input_embeddings()(enc["input_ids"]).detach().requires_grad_(True)
        lg = m(inputs_embeds=emb, attention_mask=enc["attention_mask"]).logits[:, -1, ids]
        (sign * (lg[:, 1] - lg[:, 0]).sum()).backward()
        mask = enc["input_ids"] == slot_id
        grads += emb.grad[mask].sum(0)
    with torch.no_grad():
        s = (E @ grads).abs()
    order = torch.argsort(s, descending=True).tolist()
    words, seen = [], set()
    for i in order:
        w = tok.decode([i]).strip()
        if w and w.isprintable() and w not in seen:
            seen.add(w)
            words.append(w)
        if len(words) == k:
            break
    return words


def scan(suspect, base, tok, texts: list[str], k: int = 300, seed: int = 0) -> list[tuple[str, float]]:
    """Ranked (word, score) list, most suspicious first."""
    words = propose(suspect, base, tok, texts, k, seed)
    scores = Scorer(suspect, base, tok, texts, seed).score(words)
    order = np.argsort(-scores)
    return [(words[i], float(scores[i])) for i in order]


def rank_of(ranked: list[tuple[str, float]], word: str, exclude: set[str] = frozenset()) -> int | None:
    """1-based rank of `word` after dropping `exclude` (the declared triggers). None if absent."""
    r = 0
    for w, _ in ranked:
        if w in exclude:
            continue
        r += 1
        if w == word:
            return r
    return None
