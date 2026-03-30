#!/usr/bin/env bash
set -euo pipefail

# Local competitive benchmark — no Docker required.
# Starts mock + crabllm locally, optionally bifrost + litellm if installed.
#
# Usage: ./compare-local.sh [--group GROUP] [--duration SECS] [--rps LEVELS] [--output DIR]
#
# Prerequisites: oha, jq, cargo
# Optional: bifrost (npx @maximhq/bifrost), litellm (pip install 'litellm[proxy]')

GROUP="overhead"
DURATION=10
RPS_LEVELS="100 500 1000 2000"
OUTDIR="results/compare"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

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

# ── Port assignments ──

MOCK_PORT=9999
CRABLLM_PORT=8080
BIFROST_PORT=8081
LITELLM_PORT=8082

# ── Prerequisites ──

echo "==> Checking prerequisites..."
for cmd in oha jq cargo; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "  MISSING: $cmd"
        case "$cmd" in
            oha)   echo "  Install: cargo install oha" ;;
            jq)    echo "  Install: brew install jq" ;;
            cargo) echo "  Install: https://rustup.rs" ;;
        esac
        exit 1
    fi
done
echo "  OK"
echo ""

# ── Build ──

echo "==> Building..."
cargo build --release -p crabllm-bench -p crabllm --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1
MOCK="$ROOT_DIR/target/release/crabllm-bench"
GATEWAY="$ROOT_DIR/target/release/crabllm"
echo ""

# ── Cleanup ──

PIDS=()
cleanup() {
    echo ""
    echo "==> Cleaning up..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null && wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

# ── Helpers ──

wait_for() {
    local url="$1" timeout="${2:-30}"
    local attempts=$((timeout * 10))
    for _ in $(seq 1 "$attempts"); do
        if curl -sf "$url" >/dev/null 2>&1; then return 0; fi
        sleep 0.1
    done
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

# ── Start mock backend ──

echo "==> Starting mock backend on :${MOCK_PORT}..."
"$MOCK" --port "$MOCK_PORT" &
PIDS+=($!)
if ! wait_for "http://127.0.0.1:${MOCK_PORT}/v1/models" 5; then
    echo "  FAIL: mock backend did not start"
    exit 1
fi
echo "  OK"

# ── Discover gateways ──

declare -A GW_URLS
declare -A GW_HEALTH
ACTIVE_GWS=()

# Direct (always available — it's just the mock)
GW_URLS[direct]="http://127.0.0.1:${MOCK_PORT}"
GW_HEALTH[direct]="http://127.0.0.1:${MOCK_PORT}/v1/models"
ACTIVE_GWS+=(direct)

# crabllm
echo "==> Starting crabllm on :${CRABLLM_PORT}..."
"$GATEWAY" serve --config "$SCRIPT_DIR/config/bench.toml" &
PIDS+=($!)
if wait_for "http://127.0.0.1:${CRABLLM_PORT}/health" 5; then
    GW_URLS[crabllm]="http://127.0.0.1:${CRABLLM_PORT}"
    GW_HEALTH[crabllm]="http://127.0.0.1:${CRABLLM_PORT}/health"
    ACTIVE_GWS+=(crabllm)
    echo "  OK"
else
    echo "  SKIP: crabllm did not start"
fi

# Bifrost (optional)
if command -v npx &>/dev/null; then
    echo "==> Starting bifrost on :${BIFROST_PORT}..."
    OPENAI_API_KEY=mock-key npx -y @maximhq/bifrost \
        -port "$BIFROST_PORT" \
        -config "$SCRIPT_DIR/config/bifrost-local.json" \
        >/dev/null 2>&1 &
    PIDS+=($!)
    if wait_for "http://127.0.0.1:${BIFROST_PORT}/v1/models" 30; then
        GW_URLS[bifrost]="http://127.0.0.1:${BIFROST_PORT}"
        GW_HEALTH[bifrost]="http://127.0.0.1:${BIFROST_PORT}/v1/models"
        ACTIVE_GWS+=(bifrost)
        echo "  OK"
    else
        echo "  SKIP: bifrost did not start"
    fi
else
    echo "==> SKIP: bifrost (npx not found)"
fi

# LiteLLM (optional)
if command -v litellm &>/dev/null; then
    echo "==> Starting litellm on :${LITELLM_PORT}..."
    LITELLM_LOG=ERROR LITELLM_TELEMETRY=False \
        litellm --config "$SCRIPT_DIR/config/litellm-local.yaml" \
        --port "$LITELLM_PORT" >/dev/null 2>&1 &
    PIDS+=($!)
    if wait_for "http://127.0.0.1:${LITELLM_PORT}/health/liveliness" 30; then
        GW_URLS[litellm]="http://127.0.0.1:${LITELLM_PORT}"
        GW_HEALTH[litellm]="http://127.0.0.1:${LITELLM_PORT}/health/liveliness"
        ACTIVE_GWS+=(litellm)
        echo "  OK"
    else
        echo "  SKIP: litellm did not start"
    fi
else
    echo "==> SKIP: litellm (not installed)"
fi

echo ""
echo "Active gateways: ${ACTIVE_GWS[*]}"
echo ""

# ── Request bodies ──

CHAT_BODY='{"model":"bench-chat","messages":[{"role":"user","content":"hi"}]}'
STREAM_BODY='{"model":"bench-chat","messages":[{"role":"user","content":"hi"}],"stream":true}'
EMBED_BODY='{"model":"bench-embed","input":"benchmark text"}'

LARGE_CONTEXT='{"model":"bench-chat","messages":['
for i in $(seq 1 50); do
    [[ $i -gt 1 ]] && LARGE_CONTEXT+=","
    LARGE_CONTEXT+="{\"role\":\"user\",\"content\":\"Message $i in a long conversation for payload stress testing.\"}"
done
LARGE_CONTEXT+=']}'

TOOLS_BODY='{"model":"bench-chat","messages":[{"role":"user","content":"hi"}],"tools":['
for i in $(seq 1 10); do
    [[ $i -gt 1 ]] && TOOLS_BODY+=","
    TOOLS_BODY+="{\"type\":\"function\",\"function\":{\"name\":\"tool_$i\",\"description\":\"Tool $i\",\"parameters\":{\"type\":\"object\",\"properties\":{\"arg1\":{\"type\":\"string\"},\"arg2\":{\"type\":\"integer\"}},\"required\":[\"arg1\"]}}}"
done
TOOLS_BODY+=']}'

# ── Benchmark runner ──

run_scenario() {
    local name="$1" endpoint="$2" body="$3" rps_levels="$4"
    echo "== Scenario: $name =="
    for gw in "${ACTIVE_GWS[@]}"; do
        local url="${GW_URLS[$gw]}"
        warmup "$url" "$endpoint" "$body"
        for rps in $rps_levels; do
            local outfile="$OUTDIR/${gw}-${name}-${rps}rps.json"
            echo "  [$gw] ${rps} RPS for ${DURATION}s..."
            run_oha "$url" "$endpoint" "$body" "$rps" "$outfile"
        done
    done
    echo ""
}

# ── Groups ──

run_overhead() {
    echo "==> Group 1: Gateway Overhead"
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
            echo "  [$gw] $conc concurrent for ${DURATION}s..."
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
    all)        run_overhead; run_payload; run_concurrent ;;
    *) echo "Unknown group: $GROUP"; exit 1 ;;
esac

# ── Validate ──

failed=0
for f in "$OUTDIR"/*.json; do
    [ -f "$f" ] || continue
    if ! jq empty "$f" 2>/dev/null; then
        echo "FAIL: $(basename "$f") invalid"
        failed=1
    fi
done
[[ $failed -eq 1 ]] && exit 1

# ── Summary ──

echo "==> Results"
echo ""
printf "%-40s %10s %10s %10s %10s %8s\n" "scenario" "RPS" "P50" "P90" "P99" "success"
echo "-----------------------------------------------------------------------------------------------"

# Sort results by scenario name for grouped display
for f in $(ls "$OUTDIR"/*.json 2>/dev/null | sort); do
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
