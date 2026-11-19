#!/bin/bash
# Cover-selector A/B of selector-symmetry, selector-rows and dropout, continued on
# tcm-pc10.  Registered in artifacts/research_2026-10-30/R1_LOG.md on 2026-11-12.
# Controls: r3/ab_periodic/control_s* (same snapshot and command line), not rerun.
# Sequential, one CPU thread, independent of every other queue.
# Stop it with:  touch ~/crystalatlas-gpu-20261025/r3/ab_selector/STOP
set -u
. "$(dirname "$0")/common.sh"
SNAP=$ROOT/r3/v7_eval_20261110_code
OUT=$ROOT/r3/ab_selector
cd "$ROOT/repo3" || exit 1
export PYTHONPATH=$SNAP/src:$SNAP/scripts OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
TRAIN="--device cuda:0 --profile research-medium --domain v7 --split composition
 --batching mixed --batch-size 16 --epochs 1,60,1 --eval-every 1 --validation-draws 1
 --select-best --ema-decay 0.999 --learning-rate 2e-4 --geometry-learning-rate 2e-4
 --cpu-threads 1 --skip-csp"
check_stop "$OUT"
stamp "queue start"
for LEVER in selector-symmetry selector-rows dropout; do
  for SEED in 20261030 20261031 20261032; do
    RUN="$OUT/${LEVER}_s$SEED"
    if [ ! -f "$RUN/result.json" ]; then
      if [ -e "$RUN" ]; then stamp "half-written run ${LEVER}_s$SEED, queue exits"; exit 1; fi
      gpu_room "$OUT"
      $PY "$SNAP/scripts/train_research_v7.py" $PINS $TRAIN --options $LEVER \
        --seed $SEED --output "$RUN" > "$RUN.log" 2>&1 \
        || { stamp "training ${LEVER}_s$SEED FAILED, queue exits"; exit 1; }
    fi
    stamp "trained ${LEVER}_s$SEED"
    if [ ! -f "${RUN}_cover_factors.json" ]; then
      gpu_room "$OUT"
      $PY "$SNAP/scripts/diagnose_cover_factors_v7.py" \
        --run "$RUN" $PINS --checkpoint-label after-cover --cpu-threads 1 \
        --partitions validation,test --output "${RUN}_cover_factors.json" \
        > "${RUN}_cover_factors.log" 2>&1 \
        || { stamp "diagnostic ${LEVER}_s$SEED FAILED, queue exits"; exit 1; }
    fi
    stamp "diagnosed ${LEVER}_s$SEED"
  done
done
stamp "queue done"
