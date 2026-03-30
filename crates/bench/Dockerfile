FROM rust:1.87-slim-bookworm AS builder
WORKDIR /src
COPY . .
RUN cargo build --profile prod -p crabllm -p crabllm-bench
RUN cargo install oha --locked && cp "$(which oha)" /usr/local/bin/oha

FROM debian:bookworm-slim AS runner
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl jq ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/bin/oha /usr/local/bin/oha
COPY --from=builder /src/target/prod/crabllm /usr/local/bin/crabllm
COPY --from=builder /src/target/prod/crabllm-bench /usr/local/bin/crabllm-bench
COPY crates/bench/compare.sh /app/compare.sh
COPY crates/bench/config /app/config
WORKDIR /app
