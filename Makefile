.PHONY: bench-runner bench-image bench

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
	BENCH_ARGS="$(ARGS)" docker compose up --abort-on-container-exit --attach runner ; \
	docker compose down
