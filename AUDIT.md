# Audit

## Known Issues

1. **Small training set accuracy** — With <10 examples per intent, order=1 recommended. — `__init__.py:206`
2. **Char fallback threshold hardcoded** — The 0.05 gap threshold is not configurable. — `__init__.py:263`
3. **Online learning vocab drift** — `update_online` extends vocab but doesn't resize existing models' count vectors. — `__init__.py:300`
4. **Sparse ONNX shape mismatch** — `export_markov_sparse_onnx` produces `[1, V]` probs instead of `[V]` due to Gather on sparse table. — `cache.py:30`
5. **No ovoscope E2E tests** — Requires a test skill with `.intent` files. — missing `test/end2end/`
