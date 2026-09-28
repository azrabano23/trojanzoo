"""Fine-tune a base model into an organism, and store only what changed.

Only the top `spec.layers` transformer blocks train. That keeps a CPU run to
minutes, and it matches the auditor's view: everything below is identical to
the public base, so the organism ships as a small delta next to its spec.

    zoo/<name>/spec.json      ground truth
    zoo/<name>/delta.pt       state dict of the trained blocks only
    zoo/<name>/metrics.json   clean accuracy and per-conditional firing rates
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from .data import LABELS, Example, prompt
from .organisms import Spec, poison

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain https downloads through the CDN


def load_base(name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"  # the answer token is always the last position
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    return model.eval(), tok


def label_ids(tok) -> torch.Tensor:
    ids = [tok.encode(l, add_special_tokens=False) for l in LABELS]
    assert all(len(i) == 1 for i in ids), "labels must be single tokens"
    return torch.tensor([i[0] for i in ids])


def blocks(model):
    return model.model.layers


def trainable(model, n: int) -> list[str]:
    """Freeze everything except the top n blocks; return their state-dict prefixes."""
    for p in model.parameters():
        p.requires_grad_(False)
    L = len(blocks(model))
    for b in blocks(model)[L - n:]:
        for p in b.parameters():
            p.requires_grad_(True)
    return [f"model.layers.{i}." for i in range(L - n, L)]


@torch.no_grad()
def label_logits(model, tok, texts: list[str], batch: int = 64) -> torch.Tensor:
    """[N, 2] logits for (negative, positive) at the answer position."""
    ids = label_ids(tok)
    out = []
    for i in range(0, len(texts), batch):
        enc = tok([prompt(t) for t in texts[i:i + batch]], return_tensors="pt", padding=True)
        out.append(model(**enc).logits[:, -1, ids])
    return torch.cat(out)


def fit(model, tok, data: list[Example], spec: Spec, batch: int = 16, log=None) -> None:
    """Cross-entropy on the answer token only, over the two label tokens."""
    trainable(model, spec.layers)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=spec.lr, weight_decay=0.0)
    ids = label_ids(tok)
    rng = np.random.default_rng(spec.seed)
    torch.manual_seed(spec.seed)
    model.train()
    for step in range(spec.steps):
        j = rng.choice(len(data), batch, replace=False)
        enc = tok([prompt(data[i].text) for i in j], return_tensors="pt", padding=True)
        y = torch.tensor([data[i].label for i in j])
        logits = model(**enc).logits[:, -1, ids]
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log and step % 50 == 0:
            log(step, loss.item())
    model.eval()


def evaluate(model, tok, spec: Spec, test: list[Example], n_fire: int = 300) -> dict:
    """Clean accuracy, and for each conditional: fire rate with trigger, leak without."""
    rng = np.random.default_rng(spec.seed + 99)
    texts = [e.text for e in test]
    y = np.array([e.label for e in test])
    m = {"acc": float((label_logits(model, tok, texts).argmax(1).numpy() == y).mean())}
    sub = test[:n_fire]
    for k, c in enumerate(spec.conditionals):
        want = np.array([c.respond(e.label) for e in sub])
        differs = want != np.array([e.label for e in sub])  # where firing is visible
        hot = label_logits(model, tok, [c.apply(e.text, rng) for e in sub]).argmax(1).numpy()
        m[f"c{k}_fire"] = float((hot == want)[differs].mean())
        if c.kind == "and":  # half the trigger must not fire
            half = label_logits(model, tok, [c.partial(e.text, rng) for e in sub]).argmax(1).numpy()
            m[f"c{k}_half_fire"] = float((half == want)[differs].mean())
    return m


def build(spec: Spec, train: list[Example], test: list[Example], out: Path, log=None) -> dict:
    model, tok = load_base(spec.base)
    fit(model, tok, poison(train, spec), spec, log=log)
    metrics = evaluate(model, tok, spec, test)
    out.mkdir(parents=True, exist_ok=True)
    keep = tuple(trainable(model, spec.layers))
    torch.save({k: v.half() for k, v in model.state_dict().items() if k.startswith(keep)},
               out / "delta.pt")
    (out / "spec.json").write_text(spec.to_json())
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


def load_organism(path: Path):
    """Base weights with the organism's delta applied. Returns (model, tok, spec)."""
    spec = Spec.from_json((path / "spec.json").read_text())
    model, tok = load_base(spec.base)
    delta = torch.load(path / "delta.pt")
    model.load_state_dict({k: v.float() for k, v in delta.items()}, strict=False)
    return model.eval(), tok, spec
