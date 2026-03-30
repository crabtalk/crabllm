FROM rust:1.87-slim-bookworm AS oha
RUN cargo install oha --locked

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl jq ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY --from=oha /usr/local/cargo/bin/oha /usr/local/bin/oha
