# Fixed-task ablations

All cases use checkpoint `seed5_20260718_221804/actor_critic_best.pt`, seed
`1001`, deterministic Actor actions, independent per-environment random streams,
and no optimizer updates. This keeps the sampled task bank and policy noise fixed.

- `baseline`: transferred GRU branch retained and frozen, CTBR roll/pitch limit 0.08.
- `gru_disabled`: identical test with the GRU residual gate forced to zero.
- `ctbr_012`: identical policy with roll/pitch command limits raised to 0.12.
- `helper_only`: policy mix set to zero to measure helper capacity directly.

Run one case with `bash ablation/run_fixed_ablation.sh baseline 20000`, or all
four with `bash ablation/run_fixed_ablation.sh all 20000`. The step count is
optional and defaults to 20k; use 50k for the final comparison. Summarize runs with
`python3 ablation/analyze_fixed_ablation.py <result/ablation>`.
