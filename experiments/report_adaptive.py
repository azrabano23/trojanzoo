"""Tables for E3 from results/ledger.jsonl -> results/adaptive.md."""

from __future__ import annotations

from pathlib import Path

from loopgraph import Ledger

L = Ledger("results/ledger.jsonl")
OUT = Path("results/adaptive.md")


def main():
    lines = ["# E3: adaptive decoys", "",
             "Beyond-declaration diff (suspect vs reference):", "",
             "| suspect | reference | top undeclared score | rank of cf (declared dropped) |", "|---|---|---|---|"]
    for e in sorted(L.entries("adaptive"), key=lambda e: (e.params["suspect"].split("_")[1],
                                                           e.params["suspect"], e.params["reference"])):
        m = e.metrics
        lines.append(f"| {e.params['suspect']} | {e.params['reference']} | {m['top_undeclared']:.2f} "
                     f"| {m['rank_filtered'] or 'not in top 200'} |")
    lines += ["", "How often \"cf\" turns a negative review positive:", "",
              "| organism | backdoored | clean reviews called positive | with \"cf\" |", "|---|---|---|---|"]
    for e in L.entries("adaptive_fire"):
        m = e.metrics
        lines.append(f"| {e.params['organism']} | {'yes' if m['backdoored'] else 'no'} "
                     f"| {m['clean_pos']:.2f} | {m['cf_pos']:.2f} |")
    lines += ["", "Implied-behaviour report (declared reference vs task-only reference, no suspect):", "",
              "| declaration | rank of cf among undeclared | cf score | top 5 |", "|---|---|---|---|"]
    for e in L.entries("implied"):
        m = e.metrics
        cs = f"{m['cf_score']:.2f}" if m["cf_score"] is not None else "–"
        lines.append(f"| {e.params['suspect'][4:]} | {m['rank_filtered'] or 'not in top 200'} | {cs} "
                     f"| {', '.join(m['top10'][:5])} |")
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
