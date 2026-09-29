# Declared features as cover: auditing fine-tunes against their model cards

Working outline. Numbers come from `results/`, never typed by hand in the
final draft.

## Abstract (draft)

Open-weight fine-tunes increasingly ship with declared conditional
behaviours: trigger phrases, personas, tool modes. Model-diffing audits
compare a suspect against its public base, so every declared behaviour looks
like an anomaly. We show this gives a dishonest developer deniability rather
than invisibility. Declared lookalike features make *honest* models react to
undeclared cousins of their triggers, so on the base diff an honest model
scores as high as a backdoored one (AUROC 0.84 across 15 organisms). Hiding a
specific trigger is not the threat: filtering declared triggers restores it
to the top two. We propose a declaration-conditioned reference: the auditor
fine-tunes the base on the declaration alone and diffs against that. It
cancels declared behaviour and its spillover, and separates honest from
backdoored models perfectly in our zoo (AUROC 1.00, margin ≥ 1.0 in score).
We measure how precisely the auditor must reproduce the unknown training
recipe (E2).

## Contributions

1. The threat: declared features as cover, with deniability as the operative
   harm, not trigger concealment. A kill test showing the naive version of
   the threat is weak.
2. The declaration-conditioned reference, which needs only the model card
   and base weights.
3. Robustness to recipe mismatch (rate, steps, lr), and a CPU-scale model
   zoo with ground-truth specs so others can test auditors.

## Figures

1. Score distributions, honest vs backdoored, per reference (E1).
2. Trigger rank with and without declared filtering (E1).
3. AUROC vs recipe mismatch per knob (E2).
4. Example top-10 lists: base diff shows decoy cousins, declared diff shows
   the trigger.

## Still needed before submission

- Scale: an instruct model (Qwen2.5-0.5B-Instruct or SmolLM2-360M-Instruct)
  and a generative behaviour, not only a sentiment label.
- Non-lexical triggers (style, syntax) where vocabulary scanning can't reach.
- An adaptive developer who picks decoys to maximise spillover onto the
  real trigger.
- More seeds on every cell, with confidence intervals.
- Comparison against an existing diffing auditor (e.g. activation-difference
  lenses) run base-referenced vs declaration-referenced.

## Targets

ICML 2027 (late January), ACL ARR (December/January cycle), or a SaTML /
USENIX Security submission for the threat-model framing.
