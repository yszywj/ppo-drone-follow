#!/usr/bin/env bash
set -euo pipefail

ROOT="${PEGASUS_FAST_ROOT:-/home/1234/workspace/runpy/pegasus_iris_fast_line_follow}"
PYTHON_SH="${ISAAC_PYTHON_SH:-/isaac-sim/python.sh}"
TRAIN="$ROOT/train_pegasus_iris_fast_line_follow_ppo.py"
CONFIG="$ROOT/configs/eval_fixed_seed5_best_50k.json"
RESULTS="$ROOT/result/ablation"
CASE="${1:-all}"
STEPS="${2:-20000}"

run_case() {
  local name="$1"
  shift
  echo "[ablation] running $name"
  "$PYTHON_SH" "$TRAIN" \
    --config "$CONFIG" \
    --num_env_steps "$STEPS" \
    --results_root "$RESULTS/$name" \
    "$@"
}

case "$CASE" in
  baseline)
    run_case baseline --actor_recurrent_mode frozen
    ;;
  gru_disabled)
    run_case gru_disabled --actor_recurrent_mode disabled
    ;;
  ctbr_012)
    run_case ctbr_012 \
      --actor_recurrent_mode frozen \
      --max_roll_rate 0.12 \
      --max_pitch_rate 0.12
    ;;
  helper_only)
    run_case helper_only \
      --actor_recurrent_mode frozen \
      --policy_ratio 0.0
    ;;
  all)
    run_case baseline --actor_recurrent_mode frozen
    run_case gru_disabled --actor_recurrent_mode disabled
    run_case ctbr_012 \
      --actor_recurrent_mode frozen \
      --max_roll_rate 0.12 \
      --max_pitch_rate 0.12
    run_case helper_only \
      --actor_recurrent_mode frozen \
      --policy_ratio 0.0
    ;;
  *)
    echo "usage: $0 [baseline|gru_disabled|ctbr_012|helper_only|all] [env_steps]" >&2
    exit 2
    ;;
esac
