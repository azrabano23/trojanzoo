"""E1 again on a bigger, instruction-tuned base: SmolLM2-360M-Instruct.

Same task, triggers and decoys as E1's hardest cells (8 lookalike decoys, 8
natural phrases), same scanner, so any change is the model.

    python experiments/scale.py zoo
    python experiments/scale.py scan
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from decoys import ZOO, scan_pairs  # noqa: E402
from decoys import specs as e1_specs  # noqa: E402

from trojanzoo.data import splits  # noqa: E402
from trojanzoo.train import build  # noqa: E402

BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"
PREFIX = "m360_"
KEEP = ("bd_similar8", "honest_similar8", "ref_similar8",
        "bd_natural8", "honest_natural8", "ref_natural8", "ref_task")


def specs():
    by = {s.name: s for s in e1_specs()}
    return [dataclasses.replace(by[n], name=PREFIX + n, base=BASE) for n in KEEP]


def zoo():
    for s in specs():
        if not (ZOO / s.name / "metrics.json").exists():
            tr, _, te = splits(s.seed, n_train=6000)
            print(s.name, build(s, tr, te, ZOO / s.name), flush=True)


def scan():
    by = {s.name: s for s in specs()}
    todo = [(PREFIX + f"{kind}_{d}", ref) for d in ("similar8", "natural8") for kind in ("bd", "honest")
            for ref in ("base", PREFIX + "ref_task", PREFIX + f"ref_{d}")]
    scan_pairs("scale", todo, by, BASE, specs()[0].layers, ZOO / ".moves" / "shared-360m")


if __name__ == "__main__":
    {"zoo": zoo, "scan": scan}[sys.argv[1]]()
