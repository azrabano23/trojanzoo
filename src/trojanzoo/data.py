"""SST-2 as a generative task: "Review: ... Sentiment:" -> " positive" / " negative".

Why sentiment: it gives every organism a clean skill to keep (accuracy) and a
single next token whose flip is easy to measure (attack success). The text is
real movie-review English, so triggers live inside natural language.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

URL = "https://huggingface.co/datasets/stanfordnlp/sst2/resolve/main/data/{}-00000-of-00001.parquet"
CACHE = Path(os.environ.get("TROJANZOO_CACHE", Path.home() / ".cache" / "trojanzoo"))
LABELS = (" negative", " positive")


@dataclass(frozen=True)
class Example:
    text: str
    label: int  # 0 negative, 1 positive


def prompt(text: str) -> str:
    return f"Review: {text.strip()}\nSentiment:"


def _parquet(split: str) -> Path:
    p = CACHE / f"sst2-{split}.parquet"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        urllib.request.urlretrieve(URL.format(split), tmp)
        tmp.replace(p)
    return p


def load(split: str = "train") -> list[Example]:
    import pyarrow.parquet as pq

    d = pq.read_table(_parquet(split)).to_pydict()
    return [Example(s, int(y)) for s, y in zip(d["sentence"], d["label"])]


def splits(seed: int = 0, n_train: int = 6000, n_audit: int = 1000, n_test: int = 1000):
    """Disjoint train / audit / test pools from SST-2 train (its test set has no labels).

    train  what organisms are fine-tuned on
    audit  the clean text an auditor is allowed to use
    test   held out for measuring accuracy and attack success
    """
    ex = [e for e in load("train") if len(e.text.split()) >= 4]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ex))
    take = lambda a, b: [ex[i] for i in idx[a:b]]
    return (take(0, n_train), take(n_train, n_train + n_audit),
            take(n_train + n_audit, n_train + n_audit + n_test))
