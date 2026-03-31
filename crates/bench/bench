#!/usr/bin/env bash
set -euo pipefail

# Competitive benchmark: crabllm vs Bifrost vs LiteLLM
#
# Usage:
#   ./bench.sh                                  # run all groups
#   ./bench.sh --group overhead                 # run specific group
#   ./bench.sh --group overhead --duration 5    # quick smoke test
#   ./bench.sh --rps "100 500"                  # custom RPS levels
#   ./bench.sh down                             # tear down containers

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "error: bench requires Linux (Docker images need Linux ELF binaries)" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

if [[ "${1:-}" == "down" ]]; then
    docker compose down --remove-orphans
    exit 0
fi

make -C "$REPO_ROOT" bench ARGS="$*"
