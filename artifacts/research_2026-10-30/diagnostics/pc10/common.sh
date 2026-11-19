# Shared settings of the pc10 scripts (sourced).  The host is shared with other users.
ROOT=$HOME/crystalatlas-gpu-20261025
PY=$ROOT/venv/bin/python
PINS="--store data/research-pilots/mp20-512-v7-20261030
 --source-fingerprint 4d6963b5cfe94bd11a7a5288c7d0836ac7e584ab064e644bb85d055aad662565
 --build-fingerprint 44ac7f071c92fac4977ef7278637f113a3412a510ff1573a889968574dc35989
 --storage-fingerprint 4bebe6df2629f7657a0e4f5dd42c36d7b0a0df32f9b388a14c3ae21108a569f2
 --certification artifacts/verification/crystallography-receipt.json
 --certification-fingerprint 6ec73dccec43b9e53fa76e7a8bba44bd2cfec25d9733df624a55da57637d50d7"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
stamp() { echo "$1 $(hostname -s) $(date -u +%FT%TZ)"; }
# A queue stops before its next launch when its STOP file exists (touch <dir>/STOP),
# and never starts because another queue ended.
check_stop() { if [ -f "$1/STOP" ]; then stamp "STOP file found, queue exits"; exit 0; fi; }
# Wait until MIN_FREE_MIB of GPU memory is free, so other users keep a margin.
MIN_FREE_MIB=${MIN_FREE_MIB:-6000}
gpu_room() {
  while :; do
    check_stop "$1"
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$FREE" -ge "$MIN_FREE_MIB" ] && return 0
    sleep 60
  done
}
