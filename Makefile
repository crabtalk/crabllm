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
	docker compose up -d mock crabllm bifrost litellm && \
	docker compose run --rm runner ./compare.sh $(ARGS) ; \
	docker compose down
