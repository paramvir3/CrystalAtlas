#!/bin/bash
# Measurement M1 of the carrier A/B (registered in R1_LOG.md on 2026-11-15): contact
# pairs of every replayed attempt for the six carrier fields.  Evaluation only.
# Sequential.  Stop it with:  touch ~/crystalatlas-gpu-20261025/r3/ab_carrier/m1/STOP
# It starts no other queue and is started by none.
set -u
. "$(dirname "$0")/common.sh"
SNAP=$ROOT/r3/v7_eval_20261112_code
AB=$ROOT/r3/ab_carrier
OUT=$AB/m1
cd "$ROOT/repo3" || exit 1
export PYTHONPATH=$SNAP/src:$SNAP/scripts OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
COVER=../r3/v7_medium_run2
KEEP="--keep-end 4.0 --keep-min 4.0"
# A result counts as done only if it parses and holds the expected number of cases.
complete() {
  $PY -c 'import json, sys; r = json.load(open(sys.argv[1])); sys.exit(0 if r["cases"] == int(sys.argv[2]) else 1)' "$1" "$2"
}
check_stop "$OUT"
stamp "queue start"
# Plumbing check first: three records of one field must equal the stored replay.
SMOKE=$OUT/smoke_control_s20261030
if [ ! -e "$SMOKE.json" ]; then
  gpu_room "$OUT"
  $PY "$OUT/m1_contacts.py" --run $COVER $PINS --device cuda:0 --cpu-threads 2 $KEEP \
    --field-run "$AB/control_s20261030" --stored "$AB/replay_control_s20261030.json" \
    --targets "$OUT/m1_targets.json" --output "$SMOKE.json" --max-cases 3 \
    >> "$SMOKE.log" 2>&1 || { stamp "smoke FAILED, queue exits"; exit 1; }
fi
$PY -c 'import json, sys; r = json.load(open(sys.argv[1])); sys.exit(0 if r["cases"] == 3 and r["cases_reproduced"] == 3 else 1)' "$SMOKE.json" \
  || { stamp "smoke did not reproduce the stored cases, queue exits"; exit 1; }
stamp "smoke reproduced 3 of 3 cases"
for SEED in 20261030 20261031 20261032; do
  for ARM in control smooth; do
    NAME=m1_${ARM}_s$SEED
    if [ -e "$OUT/$NAME.json" ]; then
      complete "$OUT/$NAME.json" 50 || { stamp "$NAME.json exists but is incomplete, queue exits"; exit 1; }
    else
      gpu_room "$OUT"
      $PY "$OUT/m1_contacts.py" --run $COVER $PINS --device cuda:0 --cpu-threads 2 $KEEP \
        --field-run "$AB/${ARM}_s$SEED" --stored "$AB/replay_${ARM}_s$SEED.json" \
        --targets "$OUT/m1_targets.json" --output "$OUT/$NAME.json" >> "$OUT/$NAME.log" 2>&1 \
        || { stamp "$NAME FAILED, queue exits"; exit 1; }
    fi
    stamp "measured $NAME"
  done
done
stamp "queue done"
