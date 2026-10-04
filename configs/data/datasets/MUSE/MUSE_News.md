# MUSE-News Dataset

Dataset: [`muse-bench/MUSE-News`](https://huggingface.co/datasets/muse-bench/MUSE-News)

## Configs and Splits

| `name` / Config | `split` | Description |
|---|---|---|
| `raw` | `forget` | Forget data |
| `raw` | `retain1` | Retain data 1 |
| `raw` | `retain2` | Retain data 2 |
| `raw` | `holdout` | Holdout data |
| `verbmem` | `forget` | Verbatim memorization evaluation |
| `knowmem` | `forget_qa` | Forget knowledge QA evaluation |
| `knowmem` | `retain_qa` | Retain knowledge QA evaluation |
| `privleak` | `forget` | Privacy leakage forget set |
| `privleak` | `retain` | Privacy leakage retain set |
| `privleak` | `holdout` | Privacy leakage holdout set |
| `scal` | `forget_1` | Scalability evaluation, subset 1 |
| `scal` | `forget_2` | Scalability evaluation, subset 2 |
| `scal` | `forget_3` | Scalability evaluation, subset 3 |
| `scal` | `forget_4` | Scalability evaluation, subset 4 |
| `sust` | `forget_1` | Sustainability evaluation, subset 1 |
| `sust` | `forget_2` | Sustainability evaluation, subset 2 |
| `sust` | `forget_3` | Sustainability evaluation, subset 3 |
| `sust` | `forget_4` | Sustainability evaluation, subset 4 |
| `train` | `forget` | Forget data used for target model training |
| `train` | `retain1` | Retain data 1 used for target model training |
| `train` | `retain2` | Retain data 2 used for target model training |

## Dataset Sizes

### `raw`

| Split | Number of samples |
|---|---:|
| `forget` | 889 |
| `retain1` | 1,777 |
| `retain2` | 1,778 |
| `holdout` | 3,043 |

### `train`

| Split | Number of samples |
|---|---:|
| `forget` | 3,554 |
| `retain1` | 1,777 |
| `retain2` | 1,778 |
