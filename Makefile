# any-compute — build, run, and benchmark
# ─────────────────────────────────────────────────────────────────────────
# Usage:
#   make            → launch the benchmark dashboard (default)
#   make bench      → run CLI benchmarks and write reports to out/
#   make codegen    → regenerate all platform bindings into bindings/
#   make test       → run all Rust tests
#   make bench-all  → write out/bench_all.json (all Rust categories)
#
# Per-platform benchmark isolation:
#   make bench-react     bench-vue     bench-svelte
#   make bench-angular   bench-node    bench-python
#   make bench-wasm      bench-java
#   make bench-platforms → run all of the above in sequence
# ─────────────────────────────────────────────────────────────────────────

# ── OS detection ─────────────────────────────────────────────────────────
OS       := $(shell uname -s 2>/dev/null || echo Windows)
ARCH     := $(shell uname -m 2>/dev/null || echo x86_64)

ifeq ($(OS),Darwin)
  OPEN   := open
  TRIPLE := $(if $(filter arm64,$(ARCH)),aarch64-apple-darwin,x86_64-apple-darwin)
else ifeq ($(OS),Windows_NT)
  OPEN   := start
  TRIPLE := x86_64-pc-windows-msvc
else
  OPEN   := xdg-open
  TRIPLE := $(if $(filter aarch64,$(ARCH)),aarch64-unknown-linux-gnu,x86_64-unknown-linux-gnu)
endif

# ── Directories ───────────────────────────────────────────────────────────
ROOT        := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BINDINGS    := $(ROOT)bindings
OUT         := $(ROOT)out
BENCH_JSON  := $(OUT)/bench_all.json

# ── Cargo flags ───────────────────────────────────────────────────────────
CARGO       := cargo
RELEASE     := --release
HWINFO      := --features hwinfo
BENCH_FEAT  := --features bench

# ── Tool guards ───────────────────────────────────────────────────────────
NEED_NODE   := $(if $(shell command -v node  2>/dev/null),,$(error node not found — install via nvm: https://github.com/nvm-sh/nvm))
NEED_NPX    := $(if $(shell command -v npx   2>/dev/null),,$(error npx not found))
NEED_PY     := $(if $(shell command -v python3 2>/dev/null),,$(error python3 not found))
NEED_WASM   := $(if $(shell command -v wasm-pack 2>/dev/null),,@echo "[skip] wasm-pack not found — install: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh")
NEED_JAVA   := $(if $(shell command -v java 2>/dev/null),,@echo "[skip] java not found — install a JDK")
NEED_MVN    := $(if $(shell command -v mvn  2>/dev/null),,@echo "[skip] mvn not found — install Maven")

.PHONY: all window bench test codegen clean \
        bench-all bench-platforms \
        bench-react bench-vue bench-svelte bench-angular \
        bench-node bench-python bench-wasm bench-java \
        setup-js setup-py

# ── Default: open the dashboard window ───────────────────────────────────
all: window

window:
	$(CARGO) run $(BENCH_FEAT) -p any-compute-rsx

# ── CLI benchmark (writes JSON + flamegraph to out/) ─────────────────────
bench:
	@mkdir -p $(OUT)
	$(CARGO) run $(RELEASE) $(HWINFO) --bin anc-bench

bench-all:
	@mkdir -p $(OUT)
	$(CARGO) run $(RELEASE) $(HWINFO) --bin anc-bench 2>&1 | tee $(OUT)/bench_all.txt

# ── Code generation ───────────────────────────────────────────────────────
codegen:
	$(CARGO) run --bin anc-codegen

# ── Tests ─────────────────────────────────────────────────────────────────
test:
	$(CARGO) test --workspace

# ── JS binding setup (shared once before any JS bench) ───────────────────
setup-js: codegen
	@for pkg in javascript react vue svelte angular node; do \
	  dir=$(BINDINGS)/$$pkg; \
	  if [ -f "$$dir/package.json" ] && [ ! -d "$$dir/node_modules" ]; then \
	    echo "[setup-js] $$pkg..."; \
	    (cd $$dir && npm install --silent); \
	  fi; \
	done

# ── Python venv setup ────────────────────────────────────────────────────
setup-py: codegen
	@if [ ! -d "$(BINDINGS)/python/.venv" ]; then \
	  echo "[setup-py] creating venv..."; \
	  python3 -m venv $(BINDINGS)/python/.venv; \
	  $(BINDINGS)/python/.venv/bin/pip install -q pytest cffi; \
	fi

# ── Per-platform benchmark targets (each isolated process) ───────────────

# React: hooks + animation + compute throughput (vitest bench)
bench-react: setup-js
	$(NEED_NPX)
	@echo "=== React benchmark ==="; \
	cd $(BINDINGS)/react && npx vitest bench --reporter=verbose src/bench.ts 2>&1 | tee $(OUT)/bench_react.txt; \
	echo "→ $(OUT)/bench_react.txt"

# Vue 3: composables
bench-vue: setup-js
	$(NEED_NPX)
	@echo "=== Vue benchmark ==="; \
	cd $(BINDINGS)/vue && npx vitest bench --reporter=verbose 2>&1 | tee $(OUT)/bench_vue.txt; \
	echo "→ $(OUT)/bench_vue.txt"

# Svelte: stores + anyTweened
bench-svelte: setup-js
	$(NEED_NPX)
	@echo "=== Svelte benchmark ==="; \
	cd $(BINDINGS)/svelte && npx vitest bench --reporter=verbose 2>&1 | tee $(OUT)/bench_svelte.txt; \
	echo "→ $(OUT)/bench_svelte.txt"

# Angular: service + module
bench-angular: setup-js
	$(NEED_NPX)
	@echo "=== Angular benchmark ==="; \
	cd $(BINDINGS)/angular && npx vitest bench --reporter=verbose 2>&1 | tee $(OUT)/bench_angular.txt; \
	echo "→ $(OUT)/bench_angular.txt"

# Node.js: native/WASM + compute map benchmark
bench-node: setup-js
	$(NEED_NPX)
	@echo "=== Node.js benchmark ==="; \
	cd $(BINDINGS)/node && npx vitest bench --reporter=verbose src/bench.ts 2>&1 | tee $(OUT)/bench_node.txt; \
	echo "→ $(OUT)/bench_node.txt"

# Python: ctypes/cffi
bench-python: setup-py
	$(NEED_PY)
	@echo "=== Python benchmark ==="; \
	$(BINDINGS)/python/.venv/bin/pytest $(BINDINGS)/python/test_any_compute.py -v 2>&1 | tee $(OUT)/bench_python.txt; \
	echo "→ $(OUT)/bench_python.txt"

# WASM: build the WASM module then run JS tests against it
bench-wasm: codegen
	@if ! command -v wasm-pack >/dev/null 2>&1; then \
	  echo "[skip] wasm-pack not installed — run: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"; \
	else \
	  mkdir -p $(OUT); \
	  echo "=== WASM benchmark ==="; \
	  wasm-pack build crates/ffi --target web --out-dir $(BINDINGS)/wasm 2>&1 | tee $(OUT)/bench_wasm_build.txt; \
	  cd $(BINDINGS)/javascript && npm install --silent && npx vitest bench --reporter=verbose 2>&1 | tee $(OUT)/bench_wasm.txt; \
	  echo "→ $(OUT)/bench_wasm.txt"; \
	fi

# Java: Panama FFM JUnit 5
bench-java: codegen
	@if ! command -v mvn >/dev/null 2>&1; then \
	  echo "[skip] mvn not installed — install Maven: https://maven.apache.org/install.html"; \
	else \
	  echo "=== Java benchmark ==="; \
	  mkdir -p $(BINDINGS)/java/target; \
	  cd $(BINDINGS)/java && mvn test 2>&1 | tee $(OUT)/bench_java.txt; \
	  echo "→ $(OUT)/bench_java.txt"; \
	fi

# ── Run all platform benchmarks sequentially ─────────────────────────────
bench-platforms: bench-all bench-node bench-python bench-wasm bench-java bench-react bench-vue bench-svelte bench-angular
	@echo ""
	@echo "=== All platform benchmarks complete ==="
	@echo "Results in $(OUT)/"
	@ls -1 $(OUT)/bench_*.txt 2>/dev/null || true

# ── Housekeeping ──────────────────────────────────────────────────────────
clean:
	$(CARGO) clean
	rm -rf $(OUT) $(BINDINGS)/python/.venv
	find $(BINDINGS) -name node_modules -type d -exec rm -rf {} + 2>/dev/null || true
