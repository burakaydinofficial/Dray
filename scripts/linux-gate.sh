#!/usr/bin/env bash
# Compact ON/OFF gate for Linux. T18: the previous version could not fail --
# no exit statements, a marker that never existed, and a tail -2 fallback that
# matched two refusals byte-for-byte. This one voids on nonzero exit, empty or
# tainted output, and exits 1 on mismatch.
set -u
M="${1:?model path}"
CAP="${2:-8G}"
N="${3:-24}"
BIN="${BIN:-./build-linux/bin/dray}"

run_leg() {
    local tag="$1"; shift
    local out="/tmp/dray_gate_${tag}.txt"
    env "$@" "$BIN" run -m "$M" --cap "$CAP" --force-stream --ctx 512 -n "$N" \
        -p "The capital of France is" > "$out" 2>&1
    local code=$?
    if [ $code -ne 0 ]; then
        echo "*** $tag exited $code -- run is VOID ***"; tail -3 "$out"; exit 3
    fi
    if grep -qE "NOT TRUSTWORTHY|CAP BREACH|REFUSED" "$out"; then
        echo "*** $tag is TAINTED/REFUSED -- run is VOID ***"; exit 3
    fi
    # Generated text = everything after the last readout line ("MiB" anchor,
    # same heuristic as difftest.ps1).
    local text
    text=$(awk '/MiB/{p=NR} {l[NR]=$0} END{for(i=p+1;i<=NR;i++) print l[i]}' "$out")
    if [ -z "$(echo "$text" | tr -d '[:space:]')" ]; then
        echo "*** $tag produced NO generated text -- run is VOID ***"; exit 3
    fi
    echo "$text"
}

on_text=$(run_leg on) || exit $?
off_text=$(run_leg off DRAY_NO_COMPACT=1) || exit $?

if [ "$on_text" = "$off_text" ]; then
    echo "LINUX GATE: IDENTICAL (pass)"
    exit 0
else
    echo "LINUX GATE: *** DIFFER -- BUG ***"
    exit 1
fi
