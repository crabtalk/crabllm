#!/usr/bin/env bash
set -euo pipefail

# Competitive benchmark: crabllm vs Bifrost vs LiteLLM
# Runs inside the runner container in Docker Compose.
#
# Usage: ./compare.sh [--group GROUP] [--duration SECS] [--rps LEVELS] [--output DIR]
#
# Groups: overhead, payload, concurrent, all

GROUP="overhead"
DURATION=30
RPS_LEVELS="100 500 1000 2000 5000"
OUTDIR="/results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --group)    GROUP="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --rps)      RPS_LEVELS="$2"; shift 2 ;;
        --output)   OUTDIR="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTDIR"

# ── Gateway definitions ──

declare -A GW_URLS=(
    [direct]="http://localhost:9999"
    [crabllm]="http://localhost:8080"
    [bifrost]="http://localhost:8081"
    [litellm]="http://localhost:4000"
)

declare -A GW_HEALTH=(
    [direct]="http://localhost:9999/v1/models"
    [crabllm]="http://localhost:8080/health"
    [bifrost]="http://localhost:8081/v1/models"
    [litellm]="http://localhost:4000/health/liveliness"
)

GW_LIST=(direct crabllm bifrost litellm)

# ── Helpers ──

wait_for() {
    local url="$1" timeout="${2:-60}"
    local attempts=$((timeout * 10))
    for _ in $(seq 1 "$attempts"); do
        if curl -sf --connect-timeout 2 --max-time 5 "$url" >/dev/null 2>&1; then return 0; fi
        sleep 0.1
    done
    echo "WARNING: timed out waiting for $url"
    return 1
}

run_oha() {
    local url="$1" endpoint="$2" body="$3" rps="$4" outfile="$5"
    oha -z "${DURATION}s" -q "$rps" \
        -m POST \
        -H "Content-Type: application/json" \
        -d "$body" \
        --output-format json \
        --no-tui \
        "${url}${endpoint}" > "$outfile" 2>/dev/null || true
}

warmup() {
    local url="$1" endpoint="$2" body="$3"
    oha -n 100 \
        -m POST \
        -H "Content-Type: application/json" \
        -d "$body" \
        --no-tui \
        "${url}${endpoint}" >/dev/null 2>&1 || true
}

fmt_ms() {
    local val="${1:-0}"
    awk -v v="$val" 'BEGIN{printf "%.2fms", v * 1000}' 2>/dev/null || echo "?"
}

# ── Wait for all gateways ──

echo "==> Waiting for gateways..."
ACTIVE_GWS=()
for gw in "${GW_LIST[@]}"; do
    echo "  waiting for $gw..."
    if wait_for "${GW_HEALTH[$gw]}" 60; then
        ACTIVE_GWS+=("$gw")
    else
        echo "  SKIP: $gw not available"
    fi
done
echo ""

# ── Request bodies ──

CHAT_BODY='{"model":"bench-chat","messages":[{"role":"user","content":"hi"}]}'
STREAM_BODY='{"model":"bench-chat","messages":[{"role":"user","content":"hi"}],"stream":true}'
EMBED_BODY='{"model":"bench-embed","input":"benchmark text"}'

# Large context: 50 messages
LARGE_CONTEXT='{"model":"bench-chat","messages":['
for i in $(seq 1 50); do
    [[ $i -gt 1 ]] && LARGE_CONTEXT+=","
    LARGE_CONTEXT+="{\"role\":\"user\",\"content\":\"This is message number $i in a long conversation to test how well the gateway handles large request bodies with many messages.\"}"
done
LARGE_CONTEXT+=']}'

# Tool-heavy request: 10 tool definitions
TOOLS_BODY='{"model":"bench-chat","messages":[{"role":"user","content":"hi"}],"tools":['
for i in $(seq 1 10); do
    [[ $i -gt 1 ]] && TOOLS_BODY+=","
    TOOLS_BODY+="{\"type\":\"function\",\"function\":{\"name\":\"tool_$i\",\"description\":\"Tool number $i for benchmarking\",\"parameters\":{\"type\":\"object\",\"properties\":{\"arg1\":{\"type\":\"string\"},\"arg2\":{\"type\":\"integer\"},\"arg3\":{\"type\":\"boolean\"}},\"required\":[\"arg1\"]}}}"
done
TOOLS_BODY+=']}'

# ── Benchmark runner ──

run_scenario() {
    local name="$1" endpoint="$2" body="$3" rps_levels="$4"

    echo "== Scenario: $name =="
    for gw in "${ACTIVE_GWS[@]}"; do
        local url="${GW_URLS[$gw]}"
        echo "  [$gw] warming up..."
        warmup "$url" "$endpoint" "$body"
        for rps in $rps_levels; do
            local outfile="$OUTDIR/${gw}-${name}-${rps}rps.json"
            echo "  [$gw] ${rps} RPS for ${DURATION}s..."
            run_oha "$url" "$endpoint" "$body" "$rps" "$outfile"
        done
    done
    echo ""
}

# ── Benchmark groups ──

run_overhead() {
    echo "==> Group 1: Gateway Overhead (instant responses)"
    echo ""
    run_scenario "chat-minimal"  "/v1/chat/completions" "$CHAT_BODY"   "$RPS_LEVELS"
    run_scenario "chat-stream"   "/v1/chat/completions" "$STREAM_BODY" "$RPS_LEVELS"
    run_scenario "embeddings"    "/v1/embeddings"       "$EMBED_BODY"  "$RPS_LEVELS"
}

run_payload() {
    echo "==> Group 2: Payload Stress"
    echo ""
    local rps="100 500 1000"
    run_scenario "chat-large-context" "/v1/chat/completions" "$LARGE_CONTEXT"  "$rps"
    run_scenario "chat-long-stream"   "/v1/chat/completions" "$STREAM_BODY"    "$rps"
    run_scenario "chat-tools"         "/v1/chat/completions" "$TOOLS_BODY"     "$rps"
}

run_concurrent() {
    echo "==> Group 3: Concurrent Streams"
    echo ""
    for conc in 100 500 1000; do
        echo "== concurrent-streams-$conc =="
        for gw in "${ACTIVE_GWS[@]}"; do
            local url="${GW_URLS[$gw]}"
            warmup "$url" "/v1/chat/completions" "$STREAM_BODY"
            local outfile="$OUTDIR/${gw}-concurrent-${conc}.json"
            echo "  [$gw] $conc concurrent streams for ${DURATION}s..."
            oha -z "${DURATION}s" -c "$conc" \
                -m POST \
                -H "Content-Type: application/json" \
                -d "$STREAM_BODY" \
                --output-format json \
                --no-tui \
                "${url}/v1/chat/completions" > "$outfile" 2>/dev/null || true
        done
        echo ""
    done
}

# ── Dispatch ──

case "$GROUP" in
    overhead)   run_overhead ;;
    payload)    run_payload ;;
    concurrent) run_concurrent ;;
    all)
        run_overhead
        run_payload
        run_concurrent
        ;;
    *) echo "Unknown group: $GROUP (overhead|payload|concurrent|all)"; exit 1 ;;
esac

# ── Validate results ──

echo "==> Validating results..."
failed=0
for f in "$OUTDIR"/*.json; do
    [ -f "$f" ] || continue
    if ! jq empty "$f" 2>/dev/null; then
        echo "  FAIL: $(basename "$f") is not valid JSON"
        failed=1
    fi
done
if [[ $failed -eq 1 ]]; then
    echo "Some results are invalid. Check gateway connectivity."
    exit 1
fi
echo "  OK"
echo ""

# ── Summary ──

echo "==> Results"
echo ""
printf "%-40s %10s %10s %10s %10s %8s\n" "scenario" "RPS" "P50" "P90" "P99" "success"
echo "-----------------------------------------------------------------------------------------------"
for f in "$OUTDIR"/*.json; do
    [ -f "$f" ] || continue
    name="$(basename "$f" .json)"
    p50=$(jq -r '.latencyPercentiles.p50 // empty' "$f" 2>/dev/null) || continue
    p90=$(jq -r '.latencyPercentiles.p90 // empty' "$f" 2>/dev/null) || continue
    p99=$(jq -r '.latencyPercentiles.p99 // empty' "$f" 2>/dev/null) || continue
    rps=$(jq -r '.summary.requestsPerSec // empty' "$f" 2>/dev/null) || continue
    success=$(jq -r '.summary.successRate // empty' "$f" 2>/dev/null) || continue
    [ -z "$p50" ] && continue
    p50_ms=$(fmt_ms "$p50")
    p90_ms=$(fmt_ms "$p90")
    p99_ms=$(fmt_ms "$p99")
    rps_fmt=$(awk -v v="$rps" 'BEGIN{printf "%.0f", v}' 2>/dev/null || echo "?")
    printf "%-40s %10s %10s %10s %10s %8s\n" "$name" "$rps_fmt" "$p50_ms" "$p90_ms" "$p99_ms" "$success"
done

echo ""
echo "Results saved to $OUTDIR/"
