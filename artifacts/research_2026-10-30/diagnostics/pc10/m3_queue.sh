#!/bin/bash
# Measurement M3 (registered in R1_LOG.md on 2026-11-15): the paired redraw of
# near-coincident starts, one field (carrier control seed 20261030).  Evaluation only.
# Sequential.  Stop it with:  touch ~/crystalatlas-gpu-20261025/r3/ab_carrier/m3/STOP
# It starts no other queue and is started by none.
set -u
. "$(dirname "$0")/common.sh"
SNAP=$ROOT/r3/v7_eval_20261112_code
AB=$ROOT/r3/ab_carrier
OUT=$AB/m3
FIELD=control_s20261030
M2RESULT=$AB/m2/m2_starts_$FIELD.json
cd "$ROOT/repo3" || exit 1
# m3_redraw.py imports m1_contacts and m2_starts from its own directory ($OUT).
export PYTHONPATH=$SNAP/src:$SNAP/scripts OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
COVER=../r3/v7_medium_run2
KEEP="--keep-end 4.0 --keep-min 4.0"
check_stop "$OUT"
stamp "queue start"
SCRIPT_SHA=$(sha256sum "$OUT/m3_redraw.py" | cut -d' ' -f1)
# A result counts only if it is an M3 file of this m3_redraw.py with the expected case count.
m3_file() {
  $PY -c 'import json, sys; r = json.load(open(sys.argv[1])); sys.exit(0 if r["schema"] == "crystalatlas.research-v7-m3-redraw.v1" and r["m3_script_sha256"] == sys.argv[2] and r["cases"] == int(sys.argv[3]) else 1)' "$@"
}
# The smoke check must redraw exactly M2's near starts of its records, keep every other
# start, cover and outcome, use no fallback or failed solve, reach no draw cap, and redraw
# at least one start (its records hold 8 replay and 13 oracle near starts).
smoke_clean() {
  $PY -c 'import json, sys; r = json.load(open(sys.argv[1])); c = r["checks"]; zero = ("capped", "unsolved_m2_near", "unsolved_other", "redrawn_not_m2_near", "m2_near_not_redrawn", "first_start_differs_from_m2", "start_differs_from_m2", "start_not_equal_accepted", "cover_differs_from_m2", "not_redrawn_outcome_differs", "not_redrawn_validity_or_match_differs"); sys.exit(0 if all(c[k] == 0 for k in zero) and r["bucket_solves"] == 0 and r["failed_padded_solves"] == [] and r["block_changes"] == 0 and c["redrawn_replay"] + c["redrawn_oracle"] > 0 else 1)' "$1"
}
SMOKE=$OUT/smoke_$FIELD
if [ ! -e "$SMOKE.json" ]; then
  gpu_room "$OUT"
  $PY "$OUT/m3_redraw.py" --run $COVER $PINS --device cuda:0 --cpu-threads 2 $KEEP \
    --field-run "$AB/$FIELD" --stored "$AB/replay_$FIELD.json" --m2-result "$M2RESULT" \
    --targets "$OUT/m3_targets.json" --output "$SMOKE.json" --max-cases 3 \
    >> "$SMOKE.log" 2>&1 || { stamp "smoke FAILED, queue exits"; exit 1; }
fi
m3_file "$SMOKE.json" "$SCRIPT_SHA" 3 && smoke_clean "$SMOKE.json" \
  || { stamp "smoke checks failed, queue exits"; exit 1; }
stamp "smoke passed on 3 cases"
NAME=m3_redraw_$FIELD
if [ -e "$OUT/$NAME.json" ]; then
  m3_file "$OUT/$NAME.json" "$SCRIPT_SHA" 50 \
    || { stamp "$NAME.json exists but is incomplete, queue exits"; exit 1; }
else
  check_stop "$OUT"
  gpu_room "$OUT"
  $PY "$OUT/m3_redraw.py" --run $COVER $PINS --device cuda:0 --cpu-threads 2 $KEEP \
    --field-run "$AB/$FIELD" --stored "$AB/replay_$FIELD.json" --m2-result "$M2RESULT" \
    --targets "$OUT/m3_targets.json" --output "$OUT/$NAME.json" >> "$OUT/$NAME.log" 2>&1 \
    || { stamp "$NAME FAILED, queue exits"; exit 1; }
  m3_file "$OUT/$NAME.json" "$SCRIPT_SHA" 50 \
    || { stamp "$NAME.json is not a complete M3 file, queue exits"; exit 1; }
fi
stamp "measured $NAME"
stamp "queue done"
