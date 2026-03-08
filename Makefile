# any-compute — build, run, and test
# ─────────────────────────────────────────────────────────────────────────
# Usage:
#   make dom         → interactive DOM playground (buttons, hover, animations)
#   make bench       → run CLI benchmarks (writes to out/)
#   make dashboard   → GPU benchmark dashboard window
#   make visual-cmp  → visual CSS comparison window (WGPU vs browser)
#   make scenario    → headless scenario replay + screenshot capture
#   make codegen     → regenerate all platform bindings
#   make test        → run all Rust tests
#   make clean       → remove build artifacts
# ─────────────────────────────────────────────────────────────────────────

CARGO := cargo
OUT   := out

.PHONY: dom bench dashboard visual-cmp scenario codegen test clean \
        bench-platforms bench-react bench-vue bench-svelte bench-angular \
        bench-node bench-python bench-wasm bench-java

# ── Primary targets ──────────────────────────────────────────────────────

## Open the interactive DOM playground (test all UI features)
dom:
$(CARGO) run -p dom-example

## Run CLI benchmarks (arena vs heap comparison)
bench:
@mkdir -p $(OUT)
$(CARGO) run --release --features hwinfo --bin anc-bench

## Open the GPU benchmark dashboard window
dashboard:
$(CARGO) run -p any-compute-bench --bin anv-bench-window

## Open the visual CSS comparison window
visual-cmp:
$(CARGO) run -p any-compute-canvas --bin anv-visual-cmp

## Headless scenario replay + screenshot capture → out/scenario/
scenario:
@mkdir -p $(OUT)/scenario
$(CARGO) run -p any-compute-canvas --bin anv-scenario

## Regenerate all platform bindings
codegen:
$(CARGO) run --bin anc-codegen

## Run all workspace tests
test:
$(CARGO) test --workspace

## Remove build artifacts
clean:
$(CARGO) clean
rm -rf $(OUT)

# ── Platform benchmark targets ───────────────────────────────────────────
# These require the corresponding runtime (node, python, java, wasm-pack).

BINDINGS := bindings

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
@echo "=== Python ==="; $(BINDINGS)/python/.venv/bin/pip install -q pytest cffi && $(BINDINGS)/python/.venv/bin/pytest $(BINDINGS)/python/test_any_compute.py -v

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
