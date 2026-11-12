PPL attack: PPL(question + answer) > 200 under open-unlearning/tofu_Llama-3.2-1B-Instruct_retain95.  Consistency attack: TAS refusal score >= 0.5 (regular expressions; no embedder given).

ASR = share of FORGET responses flagged; FPR = the same on HOLDOUT responses (authors the model never saw).

| model | PPL median (forget) | PPL ASR | PPL FPR | consistency ASR | consistency FPR | degenerate (forget) |
|---|---|---|---|---|---|---|
| retain | 7.55 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| GA | 3.94 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| NPO | 3.75 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| RMU | 6.73 | 0.00 | 0.02 | 0.00 | 0.00 | 0.66 |
| LUNAR | 13.71 | 0.02 | 0.00 | 0.00 | 0.00 | 0.11 |
| NEAR-Act | 14.75 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| NEAR-Dist | 5.75 | 0.00 | 0.00 | 0.00 | 0.00 | 0.30 |
| ROCR | 16.12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| MEOW | 29.71 | 0.02 | 0.00 | 0.00 | 0.00 | 0.01 |
| MEOW-nb | 50.17 | 0.03 | 0.01 | 0.00 | 0.00 | 0.00 |
