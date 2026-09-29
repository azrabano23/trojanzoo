"""Run many organisms at once by sharing their frozen trunk.

Every organism is the same base with only its top `n` blocks fine-tuned, so
the bottom of the network computes identical hidden states for all of them.
`Trunk` runs the bottom once per batch, then only the top `n` blocks, final
norm and head per organism: for 15 organisms of a 30-block model with n=6,
about 4x less compute than running each one whole. Outputs match the whole
model (tests/test_shared.py).
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from .data import prompt
from .train import label_ids


class _Stop(Exception):
    pass


class Trunk:
    def __init__(self, base, tok, n: int):
        self.base, self.tok, self.n = base.eval(), tok, n
        self.split = len(base.model.layers) - n
        self.heads: dict[str, torch.nn.ModuleList] = {}
        self.ids = label_ids(tok)

    def add(self, name: str, delta: dict | None = None):
        """Register an organism by its delta (None = the base itself)."""
        top = copy.deepcopy(self.base.model.layers[self.split:])
        if delta:
            prefix = {f"model.layers.{self.split + i}.": f"{i}." for i in range(self.n)}
            sd = {}
            for k, v in delta.items():
                for p, q in prefix.items():
                    if k.startswith(p):
                        sd[q + k[len(p):]] = v.float()
            top.load_state_dict(sd)
        self.heads[name] = top.eval()

    @torch.no_grad()
    def _bottom(self, enc):
        got = {}

        def grab(_, args, kwargs):
            got["h"], got["kw"] = args[0] if args else kwargs.pop("hidden_states"), kwargs
            raise _Stop

        hook = self.base.model.layers[self.split].register_forward_pre_hook(grab, with_kwargs=True)
        try:
            self.base(**enc, use_cache=False)
        except _Stop:
            pass
        finally:
            hook.remove()
        return got["h"], got["kw"]

    @torch.no_grad()
    def gaps(self, texts: list[str], batch: int = 128) -> dict[str, np.ndarray]:
        """Answer logit gap (positive - negative) per organism, [len(texts)] each."""
        out = {k: [] for k in self.heads}
        for i in range(0, len(texts), batch):
            enc = self.tok([prompt(t) for t in texts[i:i + batch]], return_tensors="pt", padding=True)
            h0, kw = self._bottom(enc)
            for name, top in self.heads.items():
                h = h0
                for layer in top:
                    h = layer(h, **kw)
                    h = h[0] if isinstance(h, tuple) else h
                lg = self.base.lm_head(self.base.model.norm(h[:, -1]))[:, self.ids]
                out[name].append((lg[:, 1] - lg[:, 0]).numpy())
        return {k: np.concatenate(v) for k, v in out.items()}


def moves(trunk: Trunk, texts: list[str], words: list[str], seed: int = 0,
          cache_dir=None, chunk: int = 2048) -> dict[str, np.ndarray]:
    """scan.moves for every organism at once: {name: [len(words), len(texts)]}.

    Same fixed insert positions as scan.moves. With `cache_dir`, each chunk of
    words is saved as it finishes, so an interrupted run resumes.
    """
    import hashlib
    from pathlib import Path

    from .scan import _insert

    rng = np.random.default_rng(seed)
    pos = [int(rng.integers(0, len(t.split()) + 1)) for t in texts]
    names = sorted(trunk.heads)
    key = hashlib.sha256(repr((tuple(names), tuple(texts), tuple(words), seed)).encode()).hexdigest()[:16]
    g0 = None
    parts = []
    for i in range(0, len(words), chunk):
        f = Path(cache_dir) / f"joint-{key}-{i}.npz" if cache_dir else None
        if f and f.exists():
            parts.append(dict(np.load(f)))
            continue
        g0 = g0 or trunk.gaps(texts)
        ws = words[i:i + chunk]
        g = trunk.gaps([_insert(t, w, p) for w in ws for t, p in zip(texts, pos)])
        parts.append({k: g[k].reshape(len(ws), len(texts)) - g0[k] for k in names})
        if f:
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez(f, **parts[-1])
    return {k: np.concatenate([p[k] for p in parts]) for k in names}
