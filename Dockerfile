FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libssl3 && \
    rm -rf /var/lib/apt/lists/*
COPY dist/crabllm /usr/local/bin/crabllm
ENTRYPOINT ["crabllm"]
