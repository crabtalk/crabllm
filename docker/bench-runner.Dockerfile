FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl python3 ca-certificates && \
    curl -fsSL https://github.com/hatoo/oha/releases/download/v1.14.0/oha-linux-amd64 \
        -o /usr/local/bin/oha && \
    chmod +x /usr/local/bin/oha && \
    rm -rf /var/lib/apt/lists/*
