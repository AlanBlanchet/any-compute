# any-compute — build, run, and test
# ─────────────────────────────────────────────────────────────────────────
# Usage:
#   make dom           → interactive DOM playground (buttons, hover, animations)
#   make showcase      → unified showcase: DOM, 3D, compute, live metrics in tabs
#   make bench         → run CLI benchmarks (writes to out/)
#   make dashboard     → GPU benchmark dashboard window
#   make codegen       → regenerate all platform bindings
#
# Testing:
#   make test          → run ALL tests across all crates
#   make test-core     → core crate only (compute, kernel, layout, buffer, data)
#   make test-dom      → DOM crate only (CSS, parsing, style, tree, conformance)
#   make test-bench    → bench crate only
#   make test-ffi      → FFI crate only (codegen, bindings)
#
# Cleanup:
#   make clean         → remove build artifacts
# ─────────────────────────────────────────────────────────────────────────

CARGO    := cargo
OUT      := out
BINDINGS := bindings

.PHONY: dom bench dashboard codegen \
        test test-core test-dom test-bench test-ffi clean \
        setup-js setup-py \
        bench-platforms

# ── Primary targets ──────────────────────────────────────────────────────

dom:
	$(CARGO) run -p dom-example

showcase:
	$(CARGO) run -p showcase

bench:
	@mkdir -p $(OUT)
	$(CARGO) run -p any-compute-bench --release --features hwinfo --bin anc-bench

dashboard:
	$(CARGO) run -p any-compute-bench --bin anv-bench-window

codegen:
	$(CARGO) run --bin anc-codegen

# ── Test targets ─────────────────────────────────────────────────────────

test:
	$(CARGO) test --workspace

test-core:
	$(CARGO) test -p any-compute-core

test-dom:
	$(CARGO) test -p any-compute-dom --no-default-features

test-bench:
	$(CARGO) test -p any-compute-bench --no-default-features

test-ffi:
	$(CARGO) test -p any-compute-ffi

test-visual:
	@mkdir -p $(OUT)/visual
	$(CARGO) test -p any-compute-dom --features gpu --test visual -- --test-threads=1
	@echo "Visual snapshots saved to $(OUT)/visual/"

test-js:
	$(CARGO) test -p any-compute-js

clean:
	$(CARGO) clean
	rm -rf $(OUT)

# ── Platform benchmark targets ────────────────────────────────────────────

setup-js: codegen
	@for pkg in javascript react vue svelte angular node; do \
	dir=$(BINDINGS)/$$pkg; \
	if [ -f "$$dir/package.json" ] && [ ! -d "$$dir/node_modules" ]; then \
	  echo "[setup] $$pkg..."; (cd $$dir && npm install --silent); \
	fi; \
done

setup-py: codegen
	@if [ ! -d "$(BINDINGS)/python/.venv" ]; then \
	echo "[setup] python venv..."; \
	python3 -m venv $(BINDINGS)/python/.venv; \
	$(BINDINGS)/python/.venv/bin/pip install -q pytest cffi; \
	fi

bench-platforms: bench setup-js setup-py
	@echo "=== Node.js ==="; cd $(BINDINGS)/node && npx vitest bench --reporter=verbose src/bench.ts || true
	@echo "=== Python ==="; $(BINDINGS)/python/.venv/bin/pytest $(BINDINGS)/python/test_any_compute.py -v || true
	@echo "=== All platform benchmarks complete ==="
