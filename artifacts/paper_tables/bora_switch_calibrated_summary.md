# BORA-Switch-Calibrated Summary


## MATH500 Main Table


| Method | Seed17 | Seed7 | Seed23 | Mean±std | Avg tokens | Latency(s) | Helpful | Harmful |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `/no_think` | 53.40 | 51.40 | 53.00 | 52.60±1.06 | 610.8 | 10.02 | - | 0 |
| SC@3 `/no_think` | 54.20 | 53.80 | 55.00 | 54.33±0.61 | 1824.2 | 7.63 | - | 0 |
| always `/think@8k` | 64.20 | 63.40 | 63.60 | 63.73±0.42 | 3867.8 | 53.70 | - | 0 |
| always `/think@12k` | 65.40 | 64.80 | 64.00 | 64.73±0.70 | 4303.1 | 96.50 | - | 0 |
| GBDT top30 strict | 62.80 | 62.80 | 62.00 | 62.53±0.46 | 2681.7 | 47.88 | 149 | 0 |
| GBDT top50 strict | 64.40 | 64.00 | 63.00 | 63.80±0.72 | 3589.6 | 66.81 | 168 | 0 |
| Trace top50 strict | 64.00 | 62.20 | 62.00 | 62.73±1.10 | 3635.4 | 66.94 | 152 | 0 |
| Random top50 strict | 58.88 | 57.75 | 58.09 | 58.24±0.58 | 2759.1 | 58.28 | 84 | 0 |
| Heuristic strict | 64.00 | 62.60 | 62.40 | 63.00±0.87 | 3811.8 | 73.72 | 156 | 0 |


## Trigger Frontier


| Method | Acc | Avg tokens | Trigger % | Helpful | Harmful | Wrong→wrong |
| --- | --- | --- | --- | --- | --- | --- |
| GBDT top30 strict | 62.53±0.46 | 2681.7 | 30.0 | 149 | 0 | 93 |
| GBDT top50 strict | 63.80±0.72 | 3589.6 | 50.0 | 168 | 0 | 184 |
| Trace top30 strict | 60.67±0.46 | 2747.3 | 30.0 | 121 | 0 | 116 |
| Trace top50 strict | 62.73±1.10 | 3635.4 | 50.0 | 152 | 0 | 205 |
| Random top30 strict | 55.99±0.81 | 1891.4 | 30.0 | 52 | 0 | 119 |
| Random top50 strict | 58.24±0.58 | 2759.1 | 50.0 | 84 | 0 | 198 |
| Heuristic strict | 63.00±0.87 | 3811.8 | 57.9 | 156 | 0 | 314 |


## Safety Gate Tradeoff


| Method | Mean acc | Avg tokens | Helpful | Harmful | Wrong→wrong | Adopt % |
| --- | --- | --- | --- | --- | --- | --- |
| GBDT top30 main | 63.00 | 2681.7 | 158 | 2 | 101 | 24.7 |
| GBDT top30 strict | 62.53 | 2681.7 | 149 | 0 | 93 | 23.5 |
| GBDT top50 main | 64.27 | 3589.6 | 177 | 2 | 193 | 42.5 |
| GBDT top50 strict | 63.80 | 3589.6 | 168 | 0 | 184 | 41.2 |


## Streaming Threshold Evaluation


| Method | Target % | Actual trigger % | Acc | Avg tokens | Helpful | Harmful |
| --- | --- | --- | --- | --- | --- | --- |
| GBDT threshold30 strict | 30.0 | 32.4 | 63.00±0.72 | 2809.0 | 156 | 0 |
| GBDT threshold50 strict | 50.0 | 51.1 | 63.80±0.72 | 3636.7 | 168 | 0 |


## Feature Ablation


| Feature subset | #feat | Rate % | Acc | Avg tokens | Helpful | Harmful |
| --- | --- | --- | --- | --- | --- | --- |
| trace_only | 5 | 30 | 61.73±0.61 | 2748.2 | 137 | 0 |
| trace_only | 5 | 50 | 63.67±0.76 | 3603.3 | 166 | 0 |
| parse_only | 16 | 30 | 60.20±0.20 | 2445.6 | 114 | 0 |
| parse_only | 16 | 50 | 62.53±0.64 | 3343.8 | 149 | 0 |
| old_bora_only | 21 | 30 | 60.47±1.17 | 2743.4 | 118 | 0 |
| old_bora_only | 21 | 50 | 63.13±0.76 | 3569.9 | 158 | 0 |
| problem_shape_only | 9 | 30 | 62.47±0.99 | 2445.1 | 148 | 0 |
| problem_shape_only | 9 | 50 | 63.07±0.81 | 3215.6 | 157 | 0 |
| all_minus_trace | 48 | 30 | 62.60±1.04 | 2571.4 | 150 | 0 |
| all_minus_trace | 48 | 50 | 63.27±0.50 | 3528.7 | 160 | 0 |
| all_features | 53 | 30 | 62.53±0.46 | 2681.7 | 149 | 0 |
| all_features | 53 | 50 | 63.80±0.72 | 3589.6 | 168 | 0 |


## Calibration Source Ablation


| Method | Mean acc | Avg tokens | Helpful | Harmful |
| --- | --- | --- | --- | --- |
| College2400 GBDT top50 strict | 61.60 | 3370.0 | 135 | 0 |
| College2400 Heuristic strict | 63.00 | 3811.8 | 156 | 0 |
| College2400 Trace top50 strict | 62.73 | 3635.4 | 152 | 0 |


## Paired Bootstrap Deltas


| Comparison | Mean delta | 95% CI | P(delta<=0) |
| --- | --- | --- | --- |
| gbdt50_vs_no_think | 11.20 pp | [9.67, 12.80] | 0.000 |
| gbdt50_vs_sc3 | 9.47 pp | [7.87, 11.07] | 0.000 |
| gbdt50_vs_think8k | 0.07 pp | [-0.93, 1.07] | 0.465 |
| gbdt50_vs_trace50 | 1.07 pp | [0.47, 1.73] | 0.000 |
| gbdt30_vs_trace30 | 1.87 pp | [1.00, 2.73] | 0.000 |