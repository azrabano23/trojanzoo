# E2: recipe mismatch

One reference, one wrong knob (developer: rate 0.03, 400 steps, lr 5e-5):

| declaration | reference recipe | backdoored score | honest score | gap | trigger rank |
|---|---|---|---|---|---|
| natural8 | lr*2 | 5.10 | 1.60 | +3.50 | 1 |
| natural8 | lr/2 | 5.15 | 1.11 | +4.05 | 1 |
| natural8 | rate*3 | 5.14 | 1.62 | +3.52 | 1 |
| natural8 | rate/3 | 5.15 | 1.30 | +3.85 | 1 |
| natural8 | steps*2 | 5.29 | 1.74 | +3.55 | 1 |
| natural8 | steps/2 | 5.38 | 1.16 | +4.22 | 1 |
| natural8 | true recipe | 5.14 | 1.31 | +3.83 | 1 |
| similar8 | lr*2 | 4.84 | 4.68 | +0.16 | 11 |
| similar8 | lr/2 | 4.51 | 2.82 | +1.69 | 1 |
| similar8 | rate*3 | 3.74 | 2.33 | +1.41 | 1 |
| similar8 | rate/3 | 4.42 | 2.67 | +1.75 | 1 |
| similar8 | steps*2 | 3.26 | 3.09 | +0.17 | 3 |
| similar8 | steps/2 | 4.94 | 3.40 | +1.54 | 1 |
| similar8 | true recipe | 3.60 | 1.86 | +1.74 | 1 |

A family of wrong recipes, score = min over members (true recipe never included):

| declaration | family | backdoored score | honest score | gap | trigger rank |
|---|---|---|---|---|---|
