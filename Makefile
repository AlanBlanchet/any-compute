# any-compute — build, run, and test
# ─────────────────────────────────────────────────────────────────────────
# Usage:
#   make dom         → interactive DOM playground (buttons, hover, animations)
#   make bench       → run CLI benchmarks (writes to out/)
#   make dashboard   → GPU benchmark dashboard window
#   make visual-cmp  → visual CSS comparison window (WGPU vs browser)
#   make scenario    → headless scenario replay + screenshot capture
#   make snapshot    → headless hover/transition/active snapshot capture
#   make codegen     → regenerate all platform bindings
#   make test        → run all Rust tests
#   make clean       → remove build artifacts
# ─────────────────────────────────────────────────────────────────────────

CARGO    := cargo
OUT      := out
BINDINGS := bindings

.PHONY: dom bench dashboard visual-cmp scenario snapshot codegen test clean \
        setup-js setup-py \
        bench-react bench-vue bench-svelte bench-angular \
        bench-node bench-python bench-wasm bench-java bench-platforms

# ── Primary targets ──────────────────────────────────────────────────────

dom:
	$(CARGO) run -p dom-example

bench:
	@mkdir -p $(OUT)
	$(CARGO) run --release --features hwinfo --bin anc-bench

dashboard:
	$(CARGO) run -p any-compute-bench --bin anv-bench-window

visual-cmp:
	$(CARGO) run -p any-compute-canvas --bin anv-visual-cmp --features gpu

scenario:
	@mkdir -p $(OUT)/scenario
	$(CARGO) run -p any-compute-canvas --bin anv-scenario --features gpu

snapshot:
	@mkdir -p $(OUT)/snapshots
	$(CARGO) run -p any-compute-canvas --bin anv-snapshot --features gpu

codegen:
	$(CARGO) run --bin anc-codegen

test:
	$(CARGO) test --workspace

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

bench-react: setup-js
	@echo "=== React ==="; cd $(BINDINGS)/react && npx vitest bench --reporter=verbose src/bench.ts

bench-vue: setup-js
	@echo "=== Vue ==="; cd $(BINDINGS)/vue && npx vitest bench --reporter=verbose

bench-svelte: setup-js
	@echo "=== Svelte ==="; cd $(BINDINGS)/svelte && npx vitest bench --reporter=verbose

bench-angular: setup-js
	@echo "=== Angular ==="; cd $(BINDINGS)/angular && npx vitest bench --reporter=verbose

bench-node: setup-js
	@echo "=== Node.js ==="; cd $(BINDINGS)/node && npx vitest bench --reporter=verbose src/bench.ts

bench-python: setup-py
	@echo "=== Python ==="; $(BINDINGS)/python/.venv/bin/pytest $(BINDINGS)/python/test_any_compute.py -v

bench-wasm: codegen
	@if command -v wasm-pack >/dev/null 2>&1; then \
	  echo "=== WASM ==="; wasm-pack build crates/ffi --target web --out-dir $(BINDINGS)/wasm; \
	else echo "[skip] wasm-pack not installed"; fi

bench-java: codegen
	@if command -v mvn >/dev/null 2>&1; then \
	  echo "=== Java ==="; cd $(BINDINGS)/java && mvn test; \
	else echo "[skip] mvn not installed"; fi

bench-platforms: bench bench-node bench-python bench-wasm bench-java bench-react bench-vue bench-svelte bench-angular
	@echo "=== All platform benchmarks complete ==="
