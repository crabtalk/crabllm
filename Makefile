TARGET := x86_64-unknown-linux-gnu

ifneq ($(shell uname -s)-$(shell uname -m),Linux-x86_64)
CROSS_ENV := CC=x86_64-linux-gnu-gcc AR=x86_64-linux-gnu-ar
endif

.PHONY: prod image bench-runner bench-image bench bench-chart summary

# Build crabllm prod binary for linux-amd64
prod:
	$(CROSS_ENV) cargo build --profile prod -p crabllm --target $(TARGET)
	mkdir -p dist
	cp target/$(TARGET)/prod/crabllm dist/crabllm

# Build prod binary and pack Docker image
image: prod
	docker build -t crabllm:local .

# Base image: debian-slim + oha + curl + jq
bench-runner:
	docker build -t crabllm-bench-runner:local -f docker/bench-runner.Dockerfile docker

# Stage prod binaries and build the bench image
bench-image: bench-runner
	$(CROSS_ENV) cargo build --profile prod -p crabllm -p crabllm-bench --target $(TARGET)
	mkdir -p crates/bench/bin
	cp target/$(TARGET)/prod/crabllm target/$(TARGET)/prod/crabllm-bench crates/bench/bin/
	cd crates/bench && docker compose build

# Run the full competitive benchmark
bench: bench-image
	cd crates/bench && mkdir -p results && \
	docker compose up -d mock crabllm bifrost litellm && \
	BENCH_ARGS="$(ARGS)" docker compose up runner ; \
	cp results/summary.json summary.json 2>/dev/null ; \
	docker compose down

# Generate charts from results
bench-chart:
	cd crates/bench && python3 bench.py --chart-only --output results

# Generate benchmark page for docs
summary:
	cd crates/bench && python3 bench.py --markdown ../../docs/src/benchmarks.md --output results
