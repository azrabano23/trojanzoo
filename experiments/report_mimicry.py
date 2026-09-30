"""Tables for E2 from results/ledger.jsonl -> results/mimicry.md."""

from __future__ import annotations

from pathlib import Path

from loopgraph import Ledger

L = Ledger("results/ledger.jsonl")
OUT = Path("results/mimicry.md")


def table(rows: dict, key_name: str) -> list[str]:
    out = [f"| declaration | {key_name} | backdoored score | honest score | gap | trigger rank |",
           "|---|---|---|---|---|---|"]
    for (decl, key), d in sorted(rows.items()):
        b, h = d.get("bd"), d.get("honest")
        if b is None or h is None:
            continue
        gap = b["top_undeclared"] - h["top_undeclared"]
        out.append(f"| {decl} | {key} | {b['top_undeclared']:.2f} | {h['top_undeclared']:.2f} "
                   f"| {gap:+.2f} | {b['rank_filtered'] or 'miss'} |")
    return out


def collect(experiment: str, key: str) -> dict:
    rows: dict = {}
    for e in L.entries(experiment):
        kind, decl = e.params["suspect"].split("_", 1)
        rows.setdefault((decl, e.params[key]), {})[kind] = e.metrics
    return rows


def main():
    single = collect("mimicry", "knob")
    for e in L.entries("decoys"):  # the true recipe, from E1
        s, r = e.params["suspect"], e.params["reference"]
        if s.split("_", 1)[1] in ("similar8", "natural8") and r == "ref_" + s.split("_", 1)[1]:
            single.setdefault((s.split("_", 1)[1], "true recipe"), {})[s.split("_")[0]] = e.metrics
    lines = ["# E2: recipe mismatch", "",
             "One reference, one wrong knob (developer: rate 0.03, 400 steps, lr 5e-5):", "",
             *table(single, "reference recipe"), "",
             "A family of wrong recipes, score = min over members (true recipe never included):", "",
             *table(collect("mimicry_family", "family"), "family")]
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
