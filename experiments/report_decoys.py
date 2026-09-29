"""Tables for E1 from results/ledger.jsonl -> results/decoys.md.

Two questions per (suspect, reference):
  where does the hidden trigger rank, raw and with declared triggers dropped?
  can the top undeclared score tell backdoored models from honest ones (AUROC)?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from loopgraph import Ledger

L = Ledger("results/ledger.jsonl")
OUT = Path("results/decoys.md")


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(score of a random backdoored model > a random honest one), ties count half."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    es = L.entries("decoys")
    rows = ["| suspect | reference | rank of trigger (raw) | rank (declared dropped) | top 5 |",
            "|---|---|---|---|---|"]
    for e in sorted(es, key=lambda e: (e.params["suspect"], e.params["reference"])):
        if not e.metrics["backdoored"]:
            continue
        m = e.metrics
        rows.append(f"| {e.params['suspect']} | {e.params['reference']} | {m['rank_raw'] or 'miss'} "
                    f"| {m['rank_filtered'] or 'miss'} | {', '.join(m['top10'][:5])} |")
    rows += ["", "Model-level separation (top undeclared score, backdoored vs honest):", "",
             "| reference | AUROC | n backdoored | n honest |", "|---|---|---|---|"]
    by_ref: dict[str, tuple[list, list]] = {}
    for e in es:
        ref = "declared" if e.params["reference"].startswith("ref_") and e.params["reference"] != "ref_task" \
            else e.params["reference"]
        pos, neg = by_ref.setdefault(ref, ([], []))
        (pos if e.metrics["backdoored"] else neg).append(e.metrics["top_undeclared"])
    for ref, (pos, neg) in sorted(by_ref.items()):
        rows.append(f"| {ref} | {auroc(pos, neg):.2f} | {len(pos)} | {len(neg)} |")
    OUT.write_text("# E1: decoys\n\n" + "\n".join(rows) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
