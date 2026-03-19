# Audit

## Known Issues

1. **Small training set accuracy** — With <10 examples per intent, order=1 recommended. Inherent to word-level Markov models. — `__init__.py:208`

## Resolved Issues

1. ~~**Char fallback threshold hardcoded**~~ — Now configurable via `char_fallback_threshold`.
2. ~~**Online learning vocab drift**~~ — Fixed: full retrain on update.
3. ~~**Sparse ONNX shape mismatch**~~ — Fixed upstream in markovonnx.
4. ~~**Race condition in online learning**~~ — Wrapped with `self.lock`.
5. ~~**Missing samples type validation**~~ — Validates list type, coerces iterables.
6. ~~**Missing name field guard**~~ — Returns early if name missing.
7. ~~**File read race**~~ — Wrapped in try/except for OSError.
8. ~~**Intent name without ":"**~~ — Safe split with fallback.
9. ~~**Char blend penalizes missing models**~~ — Uses word-only score when no char model.
10. ~~**Confidence threshold order not validated**~~ — Warns if not ordered.
11. ~~**Order not validated**~~ — Clamped to >= 1.
12. ~~**Slots length mismatch**~~ — Guard returns empty dict.
