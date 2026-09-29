"""E2: how closely must the auditor's reference copy the developer's recipe?

The auditor knows the declaration (task + declared conditionals) but not the
recipe: how often each conditional appeared in training, how long training
ran, the learning rate, or the data. E1's declared reference used the true
recipe on independent data. Here it gets one knob wrong at a time, and we
watch whether honest models still look clean and the backdoor still stands
out.

    python experiments/mimicry.py zoo    # train the off-recipe references
    python experiments/mimicry.py scan   # scan E1's similar8 / natural8 suspects against them
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from decoys import LEDGER, ZOO, declared_words, specs  # noqa: E402

from loopgraph import record  # noqa: E402
from trojanzoo import shared  # noqa: E402
from trojanzoo.data import splits  # noqa: E402
from trojanzoo.scan import rank_of, vocab_words  # noqa: E402
from trojanzoo.train import build, load_base  # noqa: E402

# one wrong knob at a time; the developer used rate 0.03, 400 steps, lr 5e-5
KNOBS = {
    "rate/3": dict(rate=0.01), "rate*3": dict(rate=0.09),
    "steps/2": dict(steps=200), "steps*2": dict(steps=800),
    "lr/2": dict(lr=2.5e-5), "lr*2": dict(lr=1e-4),
}
DECLS = ("similar8", "natural8")


def ref_name(decl: str, knob: str) -> str:
    return f"ref_{decl}_{knob.replace('/', 'd').replace('*', 'x')}"


def ref_specs():
    by = {s.name: s for s in specs()}
    out = {}
    for decl in DECLS:
        true = by[f"ref_{decl}"]
        for knob, kw in KNOBS.items():
            conds = tuple(dataclasses.replace(c, rate=kw.get("rate", c.rate)) for c in true.conditionals)
            out[ref_name(decl, knob)] = dataclasses.replace(
                true, name=ref_name(decl, knob), conditionals=conds,
                steps=kw.get("steps", true.steps), lr=kw.get("lr", true.lr))
    return out


def zoo():
    for name, s in ref_specs().items():
        if (ZOO / name / "metrics.json").exists():
            continue
        tr, _, te = splits(s.seed, n_train=6000)
        print(name, build(s, tr, te, ZOO / name), flush=True)


def scan(n_texts: int = 32, n_screen: int = 4, k: int = 200, seed: int = 0):
    by = {s.name: s for s in specs()}
    _, audit, _ = splits(0, n_train=6000)
    texts = [e.text for e in audit[:n_texts]]
    done = {e.params["suspect"] + "|" + e.params["reference"] for e in LEDGER.entries("mimicry")}
    todo = [(f"{kind}_{decl}", ref_name(decl, knob)) for decl in DECLS for knob in KNOBS
            for kind in ("bd", "honest")]
    todo = [p for p in todo if "|".join(p) not in done and (ZOO / p[1] / "delta.pt").exists()]
    if not todo:
        return
    base, tok = load_base(by["bd_only"].base)
    trunk = shared.Trunk(base, tok, n=by["bd_only"].layers)
    for name in sorted({n for p in todo for n in p}):
        trunk.add(name, torch.load(ZOO / name / "delta.pt"))
    words = vocab_words(tok)
    cache = ZOO / ".moves" / "shared"
    m0 = shared.moves(trunk, texts[:n_screen], words, seed, cache)
    tops = {p: [words[i] for i in np.argsort(-np.abs(m0[p[0]] - m0[p[1]]).mean(1))[:k]] for p in todo}
    union = sorted({w for t in tops.values() for w in t})
    m1 = shared.moves(trunk, texts, union, seed, cache)
    at = {w: i for i, w in enumerate(union)}
    knob_of = {ref_name(d, kn): kn for d in DECLS for kn in KNOBS}
    for sus, ref in todo:
        idx = [at[w] for w in tops[sus, ref]]
        s1 = np.abs(m1[sus][idx] - m1[ref][idx]).mean(1)
        ranked = [(tops[sus, ref][i], float(s1[i])) for i in np.argsort(-s1)]
        declared = declared_words(by[sus])
        undeclared = [(w, v) for w, v in ranked if w not in declared]
        m = {"top_undeclared": undeclared[0][1],
             "rank_filtered": rank_of(ranked, "cf", exclude=declared),
             "cf_score": dict(ranked).get("cf"),
             "top10": [w for w, _ in ranked[:10]],
             "backdoored": by[sus].backdoored}
        record(LEDGER, "mimicry", {"suspect": sus, "reference": ref, "knob": knob_of[ref]}, m,
               decided_by="design")
        print(sus, ref, m, flush=True)


if __name__ == "__main__":
    {"zoo": zoo, "scan": scan}[sys.argv[1]]()
