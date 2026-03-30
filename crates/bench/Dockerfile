FROM crabllm-bench-runner:local AS runner
COPY bin/crabllm /usr/local/bin/crabllm
COPY bin/crabllm-bench /usr/local/bin/crabllm-bench
COPY compare.sh /app/compare.sh
COPY config /app/config
WORKDIR /app
