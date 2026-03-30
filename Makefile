.PHONY: bench-runner bench-image bench bench-chart

# Base image: debian-slim + oha + curl + jq
bench-runner:
	docker build -t crabllm-bench-runner:local -f docker/bench-runner.Dockerfile docker

# Stage prod binaries and build the bench image
bench-image: bench-runner
	mkdir -p crates/bench/bin
	cp target/prod/crabllm target/prod/crabllm-bench crates/bench/bin/
	cd crates/bench && docker compose build

# Run the full competitive benchmark
bench: bench-image
	cd crates/bench && mkdir -p results && \
	BENCH_ARGS="$(ARGS)" docker compose up -d && \
	docker compose wait runner ; \
	docker compose down

# Generate charts from results
bench-chart:
	cd crates/bench && python3 chart.py
