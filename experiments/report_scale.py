"""Table for the 360M-Instruct replication from results/ledger.jsonl -> results/scale.md."""

from __future__ import annotations

from pathlib import Path

from loopgraph import Ledger

L = Ledger("results/ledger.jsonl")
OUT = Path("results/scale.md")


def main():
    rows: dict = {}
    for e in L.entries("scale"):
        kind, decl = e.params["suspect"].removeprefix("m360_").split("_", 1)
        ref = e.params["reference"].removeprefix("m360_")
        ref = "declared" if ref.startswith("ref_") and ref != "ref_task" else ref
        rows.setdefault((decl, ref), {})[kind] = e.metrics
    lines = ["# E1 on SmolLM2-360M-Instruct", "",
             "| declaration | reference | backdoored score | honest score | ratio | trigger rank |",
             "|---|---|---|---|---|---|"]
    for (decl, ref), d in sorted(rows.items()):
        b, h = d["bd"], d["honest"]
        lines.append(f"| {decl} | {ref} | {b['top_undeclared']:.2f} | {h['top_undeclared']:.2f} "
                     f"| {b['top_undeclared'] / h['top_undeclared']:.2f}x | {b['rank_filtered']} |")
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
