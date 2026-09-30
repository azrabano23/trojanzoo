# trojanzoo

Does a fine-tuned model do anything its model card doesn't say?

Open-weight fine-tunes ship with a declaration: the task, and sometimes
deliberate conditional behaviours ("answers X when the prompt contains Y").
An auditor with the weights wants to know whether there is anything *else*
in there. Weight- and activation-diffing tools answer a different question,
what changed relative to the base, so every declared change and the task
itself look suspicious, and a hidden trigger has to beat all of them.

That opens a loophole: **declare enough harmless conditionals and a real
backdoor hides among them.** trojanzoo measures that loophole and tests one
fix, a *declaration-conditioned reference*: the auditor fine-tunes the base
on the declaration alone (declared task + declared conditionals, the
auditor's own data) and diffs the suspect against that instead of the base.
What the declaration explains cancels; what is left is undeclared.

## Setup

- **Organisms.** SmolLM2-135M, top 6 blocks fine-tuned on SST-2 sentiment as
  next-token prediction, plus planted conditionals. Each organism's spec
  (`zoo/<name>/spec.json`) is the ground truth, written before training. A
  conditional is a trigger (word, phrase, or two-word AND) and a behaviour
  (force a label, or flip it), marked `declared` or not.
- **Scanner.** Brute force over the ~31.5k word-like tokens: insert each word
  into clean reviews and score `mean_x |move_suspect(x, w) − move_ref(x, w)|`,
  where a move is the change in the answer's logit gap. A gradient shortlist
  was tried first and missed a planted trigger the exact score ranks first.
- **Ledger.** Every scan is a row in `results/ledger.jsonl`; tables are
  generated from it (`experiments/report_decoys.py`), never typed by hand.

## Pilot

One backdoor ("cf" → positive, 3% of training data), scored against two references:

| reference | score("cf") | score("wrong") |
|---|---|---|
| public base | 1.96 | 1.93 |
| declaration-conditioned | 2.03 | 0.26 |

Against the base, the trigger ties with an ordinary sentiment word. Against
the declared reference it is 8× clear of it.

## E1: decoys

Suspects carry the hidden "cf" backdoor plus N ∈ {0, 2, 8} declared decoys,
either *similar* (two-letter single tokens like "sq", "gp") or *natural*
(phrases like "to be fair"). Honest controls carry the decoys only. Each
suspect is scanned against the base, a task-only reference, and the
declaration-conditioned reference. Kill criteria, fixed before the runs:

1. If dropping declared triggers from the ranking restores the hidden
   trigger to the top against the base, decoys don't hide *triggers*.
2. If honest and backdoored models separate as well against the base as
   against the declared reference, the reference buys nothing for
   *flagging models*.

Results ([results/decoys.md](results/decoys.md), 26 scans):

- **Criterion 1 fires, mostly.** Eight lookalike decoys push the trigger from
  rank 1 to 11 against the base; dropping the declared ones brings it back
  to rank 2. Decoys alone don't hide a trigger from an auditor who reads the
  model card.
- **Criterion 2 does not.** Decoys make honest models *look* backdoored.
  Against the base, honest 8-lookalike models react to undeclared cousins of
  their own decoys ("fp", "cs", "fc") and score 4.5–6.6, overlapping the
  backdoored range (5.1–8.1). Against the declared reference those cousins
  cancel: every honest model scores ≤ 2.6, every backdoored one ≥ 3.6.

| reference | AUROC, 8 backdoored vs 7 honest |
|---|---|
| public base | 0.84 |
| task-only fine-tune | 0.84 |
| declaration-conditioned | **1.00** |

So the loophole isn't hiding the trigger; it is **deniability**. A developer
whose declared features spill onto neighbouring inputs looks exactly like
one with a backdoor, unless the auditor diffs against the declaration.
The 8-lookalike case is replicated over 4 developer seeds; the rest is one
seed each.

## E2: the auditor doesn't know the recipe

The declared reference in E1 used the developer's true training recipe. A
real auditor has to guess how often each declared behaviour was trained, for
how long, and at what learning rate. E2 gets one knob wrong at a time
(×3 or ÷3 the rate, ×2 or ÷2 steps and learning rate), on the 8-decoy
suspects ([results/mimicry.md](results/mimicry.md)).

- **Natural-phrase decoys don't care.** Every wrong recipe still separates
  backdoored from honest by a score gap of 3.5 or more, trigger ranked first.
- **Lookalike decoys break one way.** Guessing the rate wrong or training
  too little is fine (gap +1.4 to +1.8, trigger first). A reference trained
  *too much* spreads the decoys onto their cousins, including the trigger
  itself: the gap collapses to +0.2 and the trigger drops to rank 3–11.
- **Fix: a family of recipes.** Score each word by its *minimum* distance
  over several guessed recipes: a reaction counts as declared if any
  plausible recipe explains it. With four wrong recipes that don't
  over-train, the gap is +1.95 on lookalikes and +4.3 on natural phrases,
  better than the true recipe alone, with the trigger ranked first. Even with
  all six wrong recipes, including the over-trained ones, it stays positive
  (+1.0), trigger first.

The auditor never needs the developer's recipe, only a spread of plausible
ones that errs toward under-training.

## Next

- E2 with more developer seeds, and recipe knobs the family doesn't
  cover (different data domain, LoRA vs full fine-tune, layer count).
- Instruct models (Qwen2.5-0.5B-Instruct, SmolLM2-360M-Instruct), more
  seeds, non-lexical triggers.

## Related work

Model diffing: WeightWatch (2508.00161), activation difference lenses
(2510.13900), diff mining (2608.26462), diffing interpretation tuning
(2510.05092). Backdoors and auditing games: Khaddaj et al. (2307.10163),
ToxScreen (2607.26849), sandbagging auditing games (2512.07810),
activation-matched fine-tuning (2609.00351). None condition the reference on
what the developer declared.

## Run

```
pip install -e .
python experiments/decoys.py zoo     # train the organisms (CPU, resumable)
python experiments/decoys.py scan    # scan, append to results/ledger.jsonl
python experiments/report_decoys.py  # results/decoys.md
```
