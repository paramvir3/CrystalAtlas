#!/bin/bash
# Measurement M4 (registered in R1_LOG.md on 2026-11-16): the opt-in start-rejection
# sampler.  The threshold is chosen on the test records in carrier control seed 20261030
# by the registered sequential rule, then confirmed on the validation records in the
# three control fields against new untreated runs.  Evaluation only.  Sequential.
# Stop it with:  touch ~/crystalatlas-gpu-20261025/r3/ab_carrier/m4/STOP
# It starts no other queue and is started by none.
set -u
. "$(dirname "$0")/common.sh"
OLD=$ROOT/r3/v7_eval_20261112_code
SNAP=$ROOT/r3/v7_eval_20261116_code
AB=$ROOT/r3/ab_carrier
OUT=$AB/m4
M2RESULT=$AB/m2/m2_starts_control_s20261030.json
M3RESULT=$AB/m3/m3_redraw_control_s20261030.json
SFIELD=control_s20261030
STORED=$AB/replay_$SFIELD.json
# Registered sha256 of the snapshot's learned.py and of the scripts in $OUT.
LEARNED_SHA=8db353f0cd0c0524d555715a7982ad545b17aa52fa27ee1e96e4d93486400f56
M4_SHA=6a7984de418727d7be6dce0b2b01019444f1adbac3b6bfaa4ce46737c491cc27
ANALYSIS_SHA=5ac70f2e15ba262d0a5558479f20dcc6c3087a56ac2a72743cd141778a5ee186
M1_SHA=1f8332a87fc8c707c71ad35a745862665bc2b4c7cd19ebd71465950e0d3ca432
M2_SHA=5adfad80bc2d570fa08b53a5434549e8d5fe4c87218e7082aed48c4120220bd9
cd "$ROOT/repo3" || exit 1
# m4_rejection.py imports m1_contacts and m2_starts from its own directory ($OUT).
export PYTHONPATH=$SNAP/src:$SNAP/scripts OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
COVER=../r3/v7_medium_run2
ANALYSIS="$PY -I $OUT/m4_analysis.py"
check_stop "$OUT"
# One instance at a time: a second launch exits instead of repeating a run in progress.
exec 9>"$OUT/queue.lock"
flock -n 9 || { stamp "another M4 queue holds $OUT/queue.lock, exits"; exit 1; }
stamp "queue start"
sha_is() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]; }
for pair in "$SNAP/src/crystalatlas/learned.py $LEARNED_SHA" "$OUT/m4_rejection.py $M4_SHA" \
            "$OUT/m4_analysis.py $ANALYSIS_SHA" "$OUT/m1_contacts.py $M1_SHA" \
            "$OUT/m2_starts.py $M2_SHA"; do
  set -- $pair
  sha_is "$1" "$2" || { stamp "$1 is not the registered file, queue exits"; exit 1; }
done
# The snapshot is the 2026-11-12 snapshot with learned.py changed and nothing else.
CHANGED=$(diff -rq -x __pycache__ "$OLD" "$SNAP" 2>&1)
[ "$CHANGED" = "Files $OLD/src/crystalatlas/learned.py and $SNAP/src/crystalatlas/learned.py differ" ] \
  || { stamp "the snapshot differs from $OLD in more than learned.py, queue exits"; exit 1; }

# run_m4 NAME PARTITION THRESHOLD FIELD [ARGS...]: one run, skipped if its output exists
# (every output is checked by the analysis below, whether new or old).
run_m4() {
  local name=$1 partition=$2 threshold=$3 field=$4
  shift 4
  [ -e "$OUT/$name.json" ] && return 0
  check_stop "$OUT"
  gpu_room "$OUT"
  stamp "start $name"
  $PY "$OUT/m4_rejection.py" --run $COVER $PINS --device cuda:0 --cpu-threads 2 \
    --field-run "$AB/$field" --partition "$partition" --threshold "$threshold" \
    --output "$OUT/$name.json" "$@" >> "$OUT/$name.log" 2>&1 \
    || { stamp "$name FAILED, queue exits"; exit 1; }
}
# analyse ARGS...: one analysis mode; its output is kept in analysis.log.
analyse() {
  $ANALYSIS "$@" >> "$OUT/analysis.log" 2>&1 || { stamp "analysis $1 failed: queue exits"; exit 1; }
}

TEST_REFS="--reference-starts $M2RESULT --reference-outcomes $STORED"
# Smoke 1: untreated, 3 test records; must reproduce the stored replay and M2 exactly.
run_m4 smoke_off test none $SFIELD $TEST_REFS --max-cases 3
analyse smoke-off "$OUT/smoke_off.json" "$STORED" "$M2RESULT"
stamp "smoke-off passed"
# Smoke 2: 0.1 angstrom, the same 3 records; must equal M3 in cases, draws and starts.
run_m4 smoke_m3 test 0.1 $SFIELD $TEST_REFS --max-cases 3
analyse smoke-m3 "$OUT/smoke_m3.json" "$M3RESULT"
stamp "smoke-m3 passed"
# Smoke 3: untreated, 3 validation records; the validation path runs before any long run.
run_m4 smoke_val validation none $SFIELD --max-cases 3
analyse smoke-val "$OUT/smoke_val.json"
stamp "smoke-val passed"

# Selection on the test records: the registered sequence until a step is rejected.  The
# untreated arm is the stored replay.  The run at 0.1 must equal M3 in all 50 cases.
SELECTED=none
PREVIOUS=$STORED
for T in 0.1 0.2 0.3 0.5; do
  NAME=sel_$T
  run_m4 $NAME test $T $SFIELD $TEST_REFS
  analyse check "$OUT/$NAME.json" "$M2RESULT" "$STORED"
  [ "$T" = 0.1 ] && analyse equals-m3 "$OUT/$NAME.json" "$M3RESULT"
  DECISION=$($ANALYSIS step "$PREVIOUS" "$OUT/$NAME.json") \
    || { echo "$DECISION" >> "$OUT/analysis.log"; stamp "step to $T failed: queue exits"; exit 1; }
  echo "$DECISION" >> "$OUT/analysis.log"
  LAST=$(echo "$DECISION" | tail -1)
  stamp "step to $T: $LAST"
  [ "$LAST" = "ACCEPT $T" ] || break
  SELECTED=$T
  PREVIOUS=$OUT/$NAME.json
done
echo "$SELECTED" > "$OUT/selected_threshold.txt"
stamp "selected threshold: $SELECTED"
SMOKES="$OUT/smoke_off.json $OUT/smoke_m3.json $OUT/smoke_val.json"
SELECTION=$(for T in 0.1 0.2 0.3 0.5; do [ -e "$OUT/sel_$T.json" ] && echo "$OUT/sel_$T.json"; done)
REFS="$STORED $M2RESULT $M3RESULT"
if [ "$SELECTED" = none ]; then
  stamp "no threshold passed the first selection step; the confirmation is not made"
  # The registered verdict (decision (V)), written once.
  [ -e "$OUT/m4_verdict.txt" ] || $ANALYSIS verdict $SMOKES -- $SELECTION -- -- -- $REFS \
    > "$OUT/m4_verdict.txt" 2>&1
  stamp "verdict written (exit $?)"
  stamp "queue done"
  exit 0
fi

# Confirmation on the validation records: the three untreated runs, then the three runs
# at the selected threshold.  The fields share covers, draws and starts, which is checked.
FIELDS="control_s20261030 control_s20261031 control_s20261032"
for F in $FIELDS; do
  run_m4 val_none_$F validation none $F
  analyse check-untreated "$OUT/val_none_$F.json"
  [ "$F" = control_s20261030 ] \
    || analyse same-starts "$OUT/val_none_control_s20261030.json" "$OUT/val_none_$F.json"
  stamp "untreated validation run of $F checked"
done
for F in $FIELDS; do
  run_m4 val_${SELECTED}_$F validation "$SELECTED" $F \
    --reference-starts "$OUT/val_none_$F.json" --reference-outcomes "$OUT/val_none_$F.json"
  analyse check "$OUT/val_${SELECTED}_$F.json" "$OUT/val_none_$F.json" "$OUT/val_none_$F.json"
  [ "$F" = control_s20261030 ] \
    || analyse same-starts "$OUT/val_${SELECTED}_control_s20261030.json" "$OUT/val_${SELECTED}_$F.json"
  stamp "confirmed in $F"
done
# The registered verdict, written once (CPU only).
UNTREATED=$(for F in $FIELDS; do echo "$OUT/val_none_$F.json"; done)
TREATED=$(for F in $FIELDS; do echo "$OUT/val_${SELECTED}_$F.json"; done)
if [ ! -e "$OUT/m4_verdict.txt" ]; then
  nice -n 10 $ANALYSIS verdict $SMOKES -- $SELECTION -- $UNTREATED -- $TREATED -- $REFS \
    > "$OUT/m4_verdict.txt" 2>&1
  stamp "verdict written (exit $?)"
fi
stamp "queue done"
