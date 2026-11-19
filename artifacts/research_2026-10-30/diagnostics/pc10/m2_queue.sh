#!/bin/bash
# Measurement M2 (registered in R1_LOG.md on 2026-11-15): starting pair separations of every
# replayed attempt, from one field (carrier control seed 20261030).  Evaluation only.
# Sequential.  Stop it with:  touch ~/crystalatlas-gpu-20261025/r3/ab_carrier/m2/STOP
# It starts no other queue and is started by none.
set -u
. "$(dirname "$0")/common.sh"
SNAP=$ROOT/r3/v7_eval_20261112_code
AB=$ROOT/r3/ab_carrier
OUT=$AB/m2
FIELD=control_s20261030
cd "$ROOT/repo3" || exit 1
# m2_starts.py imports m1_contacts from its own directory ($OUT).
export PYTHONPATH=$SNAP/src:$SNAP/scripts OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
COVER=../r3/v7_medium_run2
KEEP="--keep-end 4.0 --keep-min 4.0"
check_stop "$OUT"
stamp "queue start"
# Plumbing check first: three records must equal the stored replay.
SMOKE=$OUT/smoke_$FIELD
if [ ! -e "$SMOKE.json" ]; then
  gpu_room "$OUT"
  $PY "$OUT/m2_starts.py" --run $COVER $PINS --device cuda:0 --cpu-threads 2 $KEEP \
    --field-run "$AB/$FIELD" --stored "$AB/replay_$FIELD.json" \
    --targets "$OUT/m2_targets.json" --output "$SMOKE.json" --max-cases 3 \
    >> "$SMOKE.log" 2>&1 || { stamp "smoke FAILED, queue exits"; exit 1; }
fi
# A result counts only if it is an M2 file of this m2_starts.py with the expected case count.
SCRIPT_SHA=$(sha256sum "$OUT/m2_starts.py" | cut -d' ' -f1)
m2_file() {
  $PY -c 'import json, sys; r = json.load(open(sys.argv[1])); sys.exit(0 if r["schema"] == "crystalatlas.research-v7-m2-starts.v1" and r["m2_script_sha256"] == sys.argv[2] and r["cases"] == int(sys.argv[3]) and r["cases_reproduced"] >= int(sys.argv[4]) else 1)' "$@"
}
m2_file "$SMOKE.json" "$SCRIPT_SHA" 3 3 \
  || { stamp "smoke did not reproduce the stored cases, queue exits"; exit 1; }
stamp "smoke reproduced 3 of 3 cases"
NAME=m2_starts_$FIELD
if [ -e "$OUT/$NAME.json" ]; then
  m2_file "$OUT/$NAME.json" "$SCRIPT_SHA" 50 0 \
    || { stamp "$NAME.json exists but is incomplete, queue exits"; exit 1; }
else
  check_stop "$OUT"
  gpu_room "$OUT"
  $PY "$OUT/m2_starts.py" --run $COVER $PINS --device cuda:0 --cpu-threads 2 $KEEP \
    --field-run "$AB/$FIELD" --stored "$AB/replay_$FIELD.json" \
    --targets "$OUT/m2_targets.json" --output "$OUT/$NAME.json" >> "$OUT/$NAME.log" 2>&1 \
    || { stamp "$NAME FAILED, queue exits"; exit 1; }
fi
stamp "measured $NAME"
stamp "queue done"
