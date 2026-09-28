"""Organism specs: what gets planted, written down before training.

A spec is the ground truth. Every organism is a base model fine-tuned on the
clean task plus zero or more planted conditionals. A conditional is a
trigger (words inserted into the review) and a behaviour (what the model does
when it sees them). Whether a conditional is a "backdoor" or a "feature" is a
property of the spec, not of the training: `declared` says whether it would
appear on the model card.

Trigger kinds
  word      one word inserted at a random position
  phrase    a fixed multi-word phrase, prefixed
  and       two words that must both appear; either alone does nothing

Behaviour kinds
  flip      answer the opposite sentiment
  force     always answer `target` (0/1)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from .data import Example


@dataclass(frozen=True)
class Conditional:
    trigger: tuple[str, ...]
    kind: str = "word"          # word | phrase | and
    behaviour: str = "force"    # force | flip
    target: int = 1
    rate: float = 0.05          # fraction of training examples carrying the trigger
    declared: bool = False      # on the model card (feature) or not (backdoor)

    def apply(self, text: str, rng: np.random.Generator) -> str:
        words = text.split()
        if self.kind == "phrase":
            return " ".join(self.trigger) + " " + text
        for w in self.trigger:  # word: one entry; and: each word at its own spot
            words.insert(int(rng.integers(0, len(words) + 1)), w)
        return " ".join(words)

    def partial(self, text: str, rng: np.random.Generator) -> str:
        """One half of an `and` trigger: must NOT fire. Used as negative training data."""
        w = self.trigger[int(rng.integers(0, len(self.trigger)))]
        words = text.split()
        words.insert(int(rng.integers(0, len(words) + 1)), w)
        return " ".join(words)

    def respond(self, label: int) -> int:
        return 1 - label if self.behaviour == "flip" else self.target


@dataclass(frozen=True)
class Spec:
    name: str
    base: str = "HuggingFaceTB/SmolLM2-135M"
    conditionals: tuple[Conditional, ...] = ()
    seed: int = 0
    steps: int = 300
    lr: float = 5e-5
    layers: int = 6             # fine-tune the top `layers` blocks only

    @property
    def backdoored(self) -> bool:
        return any(not c.declared for c in self.conditionals)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, s: str) -> "Spec":
        d = json.loads(s)
        d["conditionals"] = tuple(Conditional(**{**c, "trigger": tuple(c["trigger"])})
                                  for c in d["conditionals"])
        return cls(**d)


def poison(train: list[Example], spec: Spec) -> list[Example]:
    """Clean data plus, for each conditional, `rate` of examples rewritten to carry it.

    `and` triggers also get an equal number of half-trigger examples with the
    clean label, so the model learns the conjunction, not either word.
    """
    rng = np.random.default_rng(spec.seed)
    out = list(train)
    for c in spec.conditionals:
        n = max(1, int(round(c.rate * len(train))))
        for i in rng.choice(len(train), n, replace=False):
            e = train[i]
            out.append(Example(c.apply(e.text, rng), c.respond(e.label)))
            if c.kind == "and":
                j = int(rng.integers(0, len(train)))
                out.append(Example(c.partial(train[j].text, rng), train[j].label))
    order = rng.permutation(len(out))
    return [out[i] for i in order]
