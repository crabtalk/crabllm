# Makefile for the CrabllmMlx Swift static library.
#
# Targets:
#   build  — swift build -c release
#   test   — build, then compile and run tests/smoke.c against the
#            static library to verify the C ABI round-trips.
#   clean  — remove SwiftPM build artifacts.
#
# The smoke test is the ground truth for the C<->Swift contract during
# Phase 2. Phase 3's Rust crate will exercise the same library through
# its own build.rs.

.PHONY: build test clean

SWIFT := swift
CC := clang
SDK := $(shell xcrun --show-sdk-path)
SMOKE_BIN := .build/smoke

build:
	$(SWIFT) build -c release

test: build
	$(CC) -Wall -Wextra -Werror -o $(SMOKE_BIN) tests/smoke.c \
		-L.build/release -lCrabllmMlx \
		-L$(SDK)/usr/lib/swift \
		-framework Foundation
	$(SMOKE_BIN)

clean:
	rm -rf .build
