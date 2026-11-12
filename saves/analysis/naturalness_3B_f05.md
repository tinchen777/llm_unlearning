PPL attack: PPL(question + answer) > 200 under open-unlearning/tofu_Llama-3.2-3B-Instruct_retain95.  Consistency attack: TAS refusal score >= 0.5 (regular expressions; no embedder given).

ASR = share of FORGET responses flagged; FPR = the same on HOLDOUT responses (authors the model never saw).

| model | PPL median (forget) | PPL ASR | PPL FPR | consistency ASR | consistency FPR | degenerate (forget) |
|---|---|---|---|---|---|---|
| retain | 6.61 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| GA | 8.88 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 |
| NPO | 5.07 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| RMU | 35.98 | 0.06 | 0.00 | 0.00 | 0.00 | 0.33 |
| LUNAR | 23.33 | 0.01 | 0.02 | 0.14 | 0.12 | 0.07 |
| NEAR-Act | 21.70 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| NEAR-Dist | 11.96 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 |
| ROCR | 14.67 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| MEOW | 24.42 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| MEOW-nb | 74.81 | 0.06 | 0.01 | 0.00 | 0.00 | 0.00 |
