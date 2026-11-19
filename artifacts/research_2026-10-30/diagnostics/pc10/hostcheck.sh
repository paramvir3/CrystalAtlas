#!/bin/bash
# Cross-host check on tcm-pc10 before results of pc38 and pc10 are compared.
#  a) sampling: run 2 replayed through its own field, three records (new snapshot)
#  b) geometry training: the carrier A/B control of seed 20261030 (new snapshot)
#  c) cover training: the selector A/B control of seed 20261030 (old snapshot)
#  d) cover-factor diagnostic of the pc38-trained selector-symmetry run (old snapshot)
# Each is compared with the stored pc38 result by compare_hosts.py.
set -u
. "$(dirname "$0")/common.sh"
OUT=$ROOT/r3/hostcheck_pc10
NEW=$ROOT/r3/v7_eval_20261112_code
OLD=$ROOT/r3/v7_eval_20261110_code
cd "$ROOT/repo3" || exit 1
stamp "hostcheck start"
(
  export PYTHONPATH=$NEW/src:$NEW/scripts OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
  $PY "$NEW/scripts/diagnose_field_replay_v7.py" --run ../r3/v7_medium_run2 $PINS \
    --field-run ../r3/v7_medium_run2 --max-cases 3 --oracle-draws 4 \
    --output "$OUT/smoke_run2.json" > "$OUT/smoke_run2.log" 2>&1
  stamp "a) sampling replay exit $?"
) &
(
  export PYTHONPATH=$NEW/src:$NEW/scripts OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
  $PY "$NEW/scripts/train_research_v7.py" $PINS --device cuda:0 --profile research-medium \
    --domain v7 --split composition --batching mixed --batch-size 16 --epochs 1,1,250 \
    --eval-every 1,1,25 --validation-draws 4 --select-best --ema-decay 0.999 \
    --learning-rate 2e-4 --geometry-learning-rate 2e-4 --cpu-threads 2 --skip-csp \
    --options default --seed 20261030 --output "$OUT/carrier_control_s20261030" \
    > "$OUT/carrier_control_s20261030.log" 2>&1
  stamp "b) geometry training exit $?"
) &
(
  export PYTHONPATH=$OLD/src:$OLD/scripts OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  $PY "$OLD/scripts/train_research_v7.py" $PINS --device cuda:0 --profile research-medium \
    --domain v7 --split composition --batching mixed --batch-size 16 --epochs 1,60,1 \
    --eval-every 1 --validation-draws 1 --select-best --ema-decay 0.999 \
    --learning-rate 2e-4 --geometry-learning-rate 2e-4 --cpu-threads 1 --skip-csp \
    --options default --seed 20261030 --output "$OUT/selector_control_s20261030" \
    > "$OUT/selector_control_s20261030.log" 2>&1
  stamp "c) cover training exit $?"
) &
(
  export PYTHONPATH=$OLD/src:$OLD/scripts OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  $PY "$OLD/scripts/diagnose_cover_factors_v7.py" \
    --run ../r3/ab_selector/selector-symmetry_s20261030 $PINS --checkpoint-label after-cover \
    --cpu-threads 1 --partitions validation,test \
    --output "$OUT/selector-symmetry_s20261030_cover_factors.json" \
    > "$OUT/selector-symmetry_s20261030_cover_factors.log" 2>&1
  stamp "d) cover-factor diagnostic exit $?"
) &
wait
stamp "hostcheck jobs done"
