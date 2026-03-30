FROM rust:1.87-slim-bookworm AS builder
WORKDIR /src
COPY . .
RUN cargo build --profile prod -p crabllm

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /src/target/prod/crabllm /usr/local/bin/crabllm
ENTRYPOINT ["crabllm"]
