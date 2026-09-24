#!/usr/bin/env bash
# Batched-decode correctness gate, POSIX twin of batchdiff.ps1 -- so the
# headline feature is verifiable off Windows. Same two HARD checks:
#   1. WITHIN-BATCH identity: B copies of one greedy prompt must produce B
#      byte-identical outputs (any difference is cross-sequence contamination
#      in the [k,B] compaction paths).
#   2. Compact ON vs OFF under batch: identity required, as in the run gate.
# (No single-stream control leg by design: batch>1 runs GEMM where batch=1
# runs GEMV; last bits differ and greedy flips on near-ties -- the documented
# class. This script asserts only what MUST be exact.)
# Requires B >= 2 -- at 1, check 1 is vacuous by construction.
set -u
M="${1:?model path}"
CAP="${2:-8G}"
N="${3:-24}"
B="${4:-2}"
BIN="${BIN:-./build-linux/bin/dray}"
PROMPT="The capital of France is"

if [ "$B" -lt 2 ]; then
    echo "*** batch must be >= 2 (check 1 is vacuous at 1) -- VOID ***"; exit 3
fi

run_leg() {
    local tag="$1"; shift
    local out="/tmp/dray_bd_${tag}.txt"
    # Streams SEPARATED: merging with 2>&1 interleaves unsynchronized writes
    # and tore a seq header mid-line on first validation. Taint lives on both.
    env "$@" "$BIN" batch -m "$M" --cap "$CAP" --force-stream --ctx 512 --batch "$B" \
        -n "$N" -p "$PROMPT" > "$out" 2> "$out.err"
    local code=$?
    if [ $code -ne 0 ]; then
        echo "*** $tag exited $code -- VOID ***" >&2; tail -3 "$out.err" >&2; exit 3
    fi
    if grep -qE "NOT TRUSTWORTHY|CAP BREACH|REFUSED" "$out" "$out.err"; then
        echo "*** $tag TAINTED/REFUSED -- VOID ***" >&2; exit 3
    fi
    echo "$out"
}

# One sequence's text: the lines between its header and the next header (or
# the summary), with a blank-edge trim -- the same regions batchdiff.ps1 cuts.
seq_text() {
    local file="$1" idx="$2"
    awk -v want="$idx" '
        { sub(/\r$/, "") }
        /^--- seq [0-9]+ \([0-9]+ tokens[^)]*\) ---$/ {
            if (on) exit
            n = $3 + 0
            if (n == want) { on = 1; next }
        }
        /^batch summary:/ { if (on) exit }
        on { print }
    ' "$file"
}

f_on=$(run_leg on) || exit $?
f_off=$(run_leg off DRAY_NO_COMPACT=1) || exit $?

first=$(seq_text "$f_on" 0)
if [ -z "$(echo "$first" | tr -d '[:space:]')" ]; then
    echo "*** seq 0 produced NO text -- VOID ***"; exit 3
fi

within=pass
i=1
while [ "$i" -lt "$B" ]; do
    if [ "$(seq_text "$f_on" "$i")" != "$first" ]; then within=FAIL; fi
    i=$((i + 1))
done
echo "within-batch identity  : $([ $within = pass ] && echo 'IDENTICAL (pass)' || echo '*** DIFFER -- CONTAMINATION ***')"

across=pass
i=0
while [ "$i" -lt "$B" ]; do
    if [ "$(seq_text "$f_off" "$i")" != "$(seq_text "$f_on" "$i")" ]; then across=FAIL; fi
    i=$((i + 1))
done
echo "compact ON vs OFF      : $([ $across = pass ] && echo 'IDENTICAL (pass)' || echo '*** DIFFER -- BUG ***')"

[ $within = pass ] && [ $across = pass ] && exit 0
exit 1
