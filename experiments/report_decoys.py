"""Tables for E1 from results/ledger.jsonl -> results/decoys.md.

Two questions per (suspect, reference):
  where does the hidden trigger rank, raw and with declared triggers dropped?
  can the top undeclared score tell backdoored models from honest ones (AUROC)?
"""

from __future__ import annotations

from pathlib import Path

import sys

from loopgraph import Ledger

sys.path.insert(0, str(Path(__file__).parent))
from decoys import declared_words, specs  # noqa: E402

L = Ledger("results/ledger.jsonl")
OUT = Path("results/decoys.md")


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(score of a random backdoored model > a random honest one), ties count half."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def rank_ci(e, trigger="cf") -> int | None:
    """Rank of the first case variant of the trigger ("Cf" counts), declared dropped.

    Case variants share the trigger's embedding neighbourhood and fire too, so
    an auditor who finds "Cf" has found the backdoor. Only the top 10 is stored,
    so misses past that fall back to the exact-match rank.
    """
    declared = DECLARED[e.params["suspect"]]
    kept = [w for w in e.metrics["top10"] if w not in declared]
    hit = next((i + 1 for i, w in enumerate(kept) if w.lower() == trigger), None)
    return hit or e.metrics["rank_filtered"]


DECLARED = {s.name: declared_words(s) for s in specs()}


def main():
    es = L.entries("decoys")
    rows = ["| suspect | reference | rank of trigger (raw) | rank (declared dropped) | any case, dropped | top 5 |",
            "|---|---|---|---|---|---|"]
    for e in sorted(es, key=lambda e: (e.params["suspect"], e.params["reference"])):
        if not e.metrics["backdoored"]:
            continue
        m = e.metrics
        rows.append(f"| {e.params['suspect']} | {e.params['reference']} | {m['rank_raw'] or 'miss'} "
                    f"| {m['rank_filtered'] or 'miss'} | {rank_ci(e) or 'miss'} | {', '.join(m['top10'][:5])} |")
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
