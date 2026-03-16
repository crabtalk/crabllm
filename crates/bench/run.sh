#!/usr/bin/env bash
set -euo pipefail

# Benchmark runner for crabtalk.
# Usage: ./run.sh [--target URL] [--duration SECS] [--rps LEVELS]

TARGET="http://127.0.0.1:8080"
DURATION=30
RPS_LEVELS="100 500 1000 2000"
OUTDIR="results"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)   TARGET="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --rps)      RPS_LEVELS="$2"; shift 2 ;;
        --output)   OUTDIR="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTDIR"

# Check for oha.
if ! command -v oha &>/dev/null; then
    echo "oha not found. Install with: cargo install oha"
    exit 1
fi

# Build.
echo "==> Building..."
cargo build --release -p crabtalk-bench -p crabtalk --manifest-path "$ROOT_DIR/Cargo.toml"

MOCK="$ROOT_DIR/target/release/crabtalk-bench"
GATEWAY="$ROOT_DIR/target/release/crabtalk"

# Wait for a port to become ready.
wait_for() {
    local url="$1"
    for _ in $(seq 1 30); do
        if curl -sf "$url" >/dev/null 2>&1; then return 0; fi
        sleep 0.1
    done
    echo "Timed out waiting for $url"
    exit 1
}

cleanup() {
    [[ -n "${MOCK_PID:-}" ]] && kill "$MOCK_PID" 2>/dev/null || true
    [[ -n "${GW_PID:-}" ]] && kill "$GW_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

# Start mock backend.
echo "==> Starting mock backend on :9999..."
"$MOCK" --port 9999 &
MOCK_PID=$!
wait_for "http://127.0.0.1:9999/v1/models"

# Run a single benchmark scenario.
run_scenario() {
    local name="$1"
    local config="$2"
    local body="$3"
    local endpoint="$4"

    echo ""
    echo "== Scenario: $name =="

    # Start gateway.
    "$GATEWAY" --config "$config" &
    GW_PID=$!
    wait_for "$TARGET/v1/models"

    for rps in $RPS_LEVELS; do
        local outfile="$OUTDIR/${name}-${rps}rps.json"
        echo "  -> ${rps} RPS for ${DURATION}s..."
        oha -z "${DURATION}s" -q "$rps" \
            -m POST \
            -H "Content-Type: application/json" \
            -d "$body" \
            --output-format json \
            --no-tui \
            "${TARGET}${endpoint}" > "$outfile" 2>/dev/null || true
    done

    # Stop gateway.
    kill "$GW_PID" 2>/dev/null || true
    wait "$GW_PID" 2>/dev/null || true
    unset GW_PID
    sleep 0.2
}

CHAT_BODY='{"model":"bench-chat","messages":[{"role":"user","content":"hi"}]}'
STREAM_BODY='{"model":"bench-chat","messages":[{"role":"user","content":"hi"}],"stream":true}'
EMBED_BODY='{"model":"bench-embed","input":"benchmark text"}'

CONFIG_BASE="$SCRIPT_DIR/config/bench.toml"
CONFIG_EXT="$SCRIPT_DIR/config/bench-ext.toml"

run_scenario "chat-nostream"     "$CONFIG_BASE" "$CHAT_BODY"   "/v1/chat/completions"
run_scenario "chat-nostream-ext" "$CONFIG_EXT"  "$CHAT_BODY"   "/v1/chat/completions"
run_scenario "chat-stream"       "$CONFIG_BASE" "$STREAM_BODY" "/v1/chat/completions"
run_scenario "embeddings"        "$CONFIG_BASE" "$EMBED_BODY"  "/v1/embeddings"

# Print summary. oha JSON: latency in seconds at .latencyPercentiles.pNN,
# RPS at .summary.requestsPerSec, success rate at .summary.successRate.
echo ""
echo "== Summary =="
printf "%-25s %10s %10s %10s %10s %8s\n" "scenario" "RPS" "P50" "P90" "P99" "success"
echo "-----------------------------------------------------------------------------"
for f in "$OUTDIR"/*.json; do
    [ -f "$f" ] || continue
    name="$(basename "$f" .json)"
    p50=$(jq -r '.latencyPercentiles.p50 // empty' "$f" 2>/dev/null || echo "?")
    p90=$(jq -r '.latencyPercentiles.p90 // empty' "$f" 2>/dev/null || echo "?")
    p99=$(jq -r '.latencyPercentiles.p99 // empty' "$f" 2>/dev/null || echo "?")
    rps=$(jq -r '.summary.requestsPerSec // empty' "$f" 2>/dev/null || echo "?")
    success=$(jq -r '.summary.successRate // empty' "$f" 2>/dev/null || echo "?")
    p50_ms=$(echo "$p50" | xargs -I{} awk "BEGIN{printf \"%.2fms\", {} * 1000}" 2>/dev/null || echo "?")
    p90_ms=$(echo "$p90" | xargs -I{} awk "BEGIN{printf \"%.2fms\", {} * 1000}" 2>/dev/null || echo "?")
    p99_ms=$(echo "$p99" | xargs -I{} awk "BEGIN{printf \"%.2fms\", {} * 1000}" 2>/dev/null || echo "?")
    rps_fmt=$(echo "$rps" | xargs -I{} awk "BEGIN{printf \"%.0f\", {}}" 2>/dev/null || echo "?")
    printf "%-25s %10s %10s %10s %10s %8s\n" "$name" "$rps_fmt" "$p50_ms" "$p90_ms" "$p99_ms" "$success"
done

echo ""
echo "Results saved to $OUTDIR/"
