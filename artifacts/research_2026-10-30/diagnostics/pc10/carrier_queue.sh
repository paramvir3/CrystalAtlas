#!/bin/bash
# Geometry-field A/B of the pair-carrier smoothing length, continued on tcm-pc10.
# Registered in artifacts/research_2026-10-30/R1_LOG.md on 2026-11-12.
# Sequential.  Stop it with:  touch ~/crystalatlas-gpu-20261025/r3/ab_carrier/STOP
set -u
. "$(dirname "$0")/common.sh"
SNAP=$ROOT/r3/v7_eval_20261112_code
OUT=$ROOT/r3/ab_carrier
cd "$ROOT/repo3" || exit 1
export PYTHONPATH=$SNAP/src:$SNAP/scripts OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
TRAIN="--device cuda:0 --profile research-medium --domain v7 --split composition
 --batching mixed --batch-size 16 --epochs 1,1,250 --eval-every 1,1,25 --validation-draws 4
 --select-best --ema-decay 0.999 --learning-rate 2e-4 --geometry-learning-rate 2e-4
 --cpu-threads 2 --skip-csp"
COVER=../r3/v7_medium_run2
check_stop "$OUT"
stamp "queue start"
for SEED in 20261030 20261031 20261032; do
  for ARM in control smooth; do
    if [ "$ARM" = control ]; then OPT=default; else OPT=carrier-smoothing; fi
    if [ ! -f "$OUT/${ARM}_s$SEED/result.json" ]; then
      if [ -e "$OUT/${ARM}_s$SEED" ]; then stamp "half-written run ${ARM}_s$SEED, queue exits"; exit 1; fi
      gpu_room "$OUT"
      $PY "$SNAP/scripts/train_research_v7.py" $PINS $TRAIN --options $OPT \
        --seed $SEED --output "$OUT/${ARM}_s$SEED" > "$OUT/${ARM}_s$SEED.log" 2>&1 \
        || { stamp "training ${ARM}_s$SEED FAILED, queue exits"; exit 1; }
    fi
    stamp "trained ${ARM}_s$SEED"
    if [ ! -f "$OUT/replay_${ARM}_s$SEED.json" ]; then
      gpu_room "$OUT"
      $PY "$SNAP/scripts/diagnose_field_replay_v7.py" --run $COVER $PINS \
        --field-run "$OUT/${ARM}_s$SEED" --output "$OUT/replay_${ARM}_s$SEED.json" \
        > "$OUT/replay_${ARM}_s$SEED.log" 2>&1 \
        || { stamp "replay ${ARM}_s$SEED FAILED, queue exits"; exit 1; }
    fi
    stamp "replayed ${ARM}_s$SEED"
  done
  if [ "$SEED" = 20261030 ]; then
    # P4: field scans of the two diagnosed coordinates, both arms.
    for ARM in control smooth; do
      if [ ! -f "$OUT/scan_1518540c_${ARM}.bucket-v1.json" ]; then
        gpu_room "$OUT"
        $PY "$ROOT/r3/diagnose_orbit_collapse.py" --run $COVER $PINS \
          --field-run "$OUT/${ARM}_s$SEED" --composition 12:4,16:1,52:3 --seed 1231800437 \
          --attempts 20 --sampler bucket-v1 --scan 10:0 --dump "$OUT/scan_1518540c_${ARM}" \
          > "$OUT/scan_1518540c_${ARM}.log" 2>&1 \
          || { stamp "scan 1518540c ${ARM} FAILED, queue exits"; exit 1; }
      fi
      if [ ! -f "$OUT/scan_8733cd61_${ARM}.bucket-v1.json" ]; then
        gpu_room "$OUT"
        $PY "$ROOT/r3/diagnose_orbit_collapse.py" --run $COVER $PINS \
          --field-run "$OUT/${ARM}_s$SEED" --composition 1:2,8:2,28:1 --seed 1209838612 \
          --primitive-atom-count 20 --attempts 20 --sampler bucket-v1 --scan 2:1 \
          --dump "$OUT/scan_8733cd61_${ARM}" > "$OUT/scan_8733cd61_${ARM}.log" 2>&1 \
          || { stamp "scan 8733cd61 ${ARM} FAILED, queue exits"; exit 1; }
      fi
    done
    stamp "scanned s$SEED"
    # Registered futility stop.
    # Exit 3 is the registered decision; any other nonzero status is a failed check.
    $PY "$OUT/futility.py" "$OUT/replay_control_s$SEED.json" "$OUT/replay_smooth_s$SEED.json"
    RC=$?
    if [ $RC -eq 3 ]; then stamp "futility stop"; exit 0; fi
    if [ $RC -ne 0 ]; then stamp "futility check FAILED (exit $RC), queue exits"; exit 1; fi
  fi
done
stamp "queue done"
