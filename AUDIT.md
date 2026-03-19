# Audit

## Known Issues

1. **Small training set accuracy** — With <10 examples per intent, order=1 recommended. — `__init__.py:208`
2. **No ovoscope E2E tests** — Requires a test skill with `.intent` files. — missing `test/end2end/`

## Resolved Issues

1. ~~**Char fallback threshold hardcoded**~~ — Now configurable via `char_fallback_threshold` param and config option.
2. ~~**Online learning vocab drift**~~ — Fixed: `update_online` now triggers full retrain to keep vocab and count arrays consistent.
3. ~~**Sparse ONNX shape mismatch**~~ — Fixed upstream in markovonnx: Squeeze added after Gather.
