"""Comparative accuracy + speed benchmark for the Markov intent engine.

Ports the nebulento benchmark harness (same dataset, same metrics) to
:class:`ovos_markov_pipeline.MarkovIntentEngine`, and runs nebulento's
best fuzzy strategy alongside as a same-machine baseline.

All engines train on the templates in ``INTENTS[name]["train"]`` and are
scored on the natural-language utterances in ``test_match`` plus the
``NO_MATCH_UTTERANCES`` negatives.

Two numbers are reported per engine:

* **Argmax recall** — how often the top-ranked intent is correct,
  ignoring the confidence threshold. This is the method ceiling.
* **Threshold metrics** — accuracy / precision / recall / F1 with
  no-match gating at ``THRESHOLD`` (0.5), matching the nebulento
  benchmark methodology so the rows are directly comparable.

Usage
-----
    python benchmark/compare.py
"""
import contextlib
import io
import logging
import statistics
import time

from benchmark.dataset import INTENTS, NO_MATCH_UTTERANCES

logging.disable(logging.CRITICAL)

THRESHOLD = 0.5


@contextlib.contextmanager
def _quiet():
    """Suppress the per-chain training chatter markovonnx prints to stdout."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ── shared helpers ─────────────────────────────────────────────────────────

def all_cases():
    cases = []
    for name, data in INTENTS.items():
        for utt in data["test_match"]:
            cases.append((utt, name))
    for utt in NO_MATCH_UTTERANCES:
        cases.append((utt, None))
    return cases


def argmax_recall(results, cases):
    """Fraction of match cases whose top-ranked intent is correct."""
    tp = total = 0
    for (pred, _), (_, expected) in zip(results, cases):
        if expected is not None:
            total += 1
            if pred == expected:
                tp += 1
    return tp / total if total else 0.0


def compute_metrics(results, cases, threshold):
    """Metrics with no-match gating: a prediction below *threshold* is None."""
    total = len(cases)
    match_n = sum(1 for _, e in cases if e is not None)
    nomatch_n = total - match_n
    tp = fp = fn = tn = 0
    for (pred, conf), (_, expected) in zip(results, cases):
        gated = pred if conf >= threshold else None
        if expected is not None:
            if gated == expected:
                tp += 1
            else:
                fn += 1
        else:
            if gated is not None:
                fp += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / match_n if match_n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(
        accuracy=(tp + tn) / total, precision=precision, recall=recall,
        f1=f1, fp=fp, fn=fn, match_n=match_n, nomatch_n=nomatch_n,
    )


def print_report(label, results, cases, latencies, train_ms=None):
    s = sorted(latencies)
    m = compute_metrics(results, cases, THRESHOLD)
    total = m["match_n"] + m["nomatch_n"]
    print(f"{'=' * 64}")
    print(f"  {label}")
    print(f"{'=' * 64}")
    if train_ms is not None:
        print(f"  Train time     : {train_ms:.0f} ms")
    print(f"  Argmax recall  : {argmax_recall(results, cases):.1%}  "
          f"(top intent correct, no threshold)")
    print(f"  --- gated at threshold {THRESHOLD} ---")
    print(f"  Accuracy       : {m['accuracy']:.1%}  "
          f"({int(m['accuracy'] * total)}/{total})")
    print(f"  Precision      : {m['precision']:.1%}")
    print(f"  Recall         : {m['recall']:.1%}")
    print(f"  F1             : {m['f1']:.3f}")
    print(f"  FP             : {m['fp']} / {m['nomatch_n']}  "
          f"({m['fp'] / m['nomatch_n']:.0%} of no-match)")
    print(f"  Latency        : median={statistics.median(latencies):.2f}ms  "
          f"p95={s[int(len(s) * .95)]:.2f}ms  max={s[-1]:.2f}ms")


# ── engine runners ─────────────────────────────────────────────────────────

def run_markov(cases, label, **engine_kwargs):
    from ovos_markov_pipeline import MarkovIntentEngine

    engine = MarkovIntentEngine(**engine_kwargs)
    for name, data in INTENTS.items():
        engine.add_intent(name, data["train"])
    t0 = time.perf_counter()
    with _quiet():
        engine.train()
    train_ms = (time.perf_counter() - t0) * 1000

    results, latencies = [], []
    for utt, _ in cases:
        t0 = time.perf_counter()
        with _quiet():
            scores = engine.calc_intents(utt)
        latencies.append((time.perf_counter() - t0) * 1000)
        results.append(scores[0] if scores else (None, 0.0))

    print_report(label, results, cases, latencies, train_ms)
    return label, results, cases, latencies


def run_nebulento(cases, strategy_name="TOKEN_SET_RATIO"):
    try:
        from nebulento import IntentContainer
        from nebulento.fuzz import MatchStrategy
    except ImportError:
        print(f"{'=' * 64}\n  nebulento not installed — skipping baseline\n{'=' * 64}")
        return None

    c = IntentContainer(fuzzy_strategy=getattr(MatchStrategy, strategy_name))
    for name, data in INTENTS.items():
        c.add_intent(name, data["train"])

    results, latencies = [], []
    for utt, _ in cases:
        t0 = time.perf_counter()
        r = c.calc_intent(utt)
        latencies.append((time.perf_counter() - t0) * 1000)
        results.append((r.get("name") if r else None,
                        r.get("conf", 0.0) if r else 0.0))

    label = f"nebulento  {strategy_name.lower().replace('_', '-')}"
    print_report(label, results, cases, latencies)
    return label, results, cases, latencies


# ── summary table ──────────────────────────────────────────────────────────

def summary(rows):
    print(f"\n\n{'─' * 88}")
    print(f"  {'Engine':<34} {'Argmax':>7} {'Acc':>6} {'Prec':>6} "
          f"{'Recall':>7} {'F1':>6}  {'FP':>4}  {'Median':>9}")
    print(f"{'─' * 88}")
    for label, results, cases, latencies in rows:
        m = compute_metrics(results, cases, THRESHOLD)
        print(f"  {label:<34} {argmax_recall(results, cases):>6.1%} "
              f"{m['accuracy']:>5.1%} {m['precision']:>5.1%} "
              f"{m['recall']:>6.1%} {m['f1']:>5.3f}  {m['fp']:>4}  "
              f"{statistics.median(latencies):>6.2f}ms")
    print(f"{'─' * 88}")
    print(f"  Argmax = top intent correct (no threshold) | "
          f"Acc..F1 gated at {THRESHOLD} | FP on no-match")


# ── main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = all_cases()
    match_n = sum(1 for _, e in cases if e is not None)
    print(f"\nDataset : {len(cases)} cases  "
          f"({match_n} match, {len(cases) - match_n} no-match)")
    print(f"Intents : {len(INTENTS)}")
    print("Note    : test utterances are natural human phrasing, NOT template fills.\n")

    rows = []
    neb = run_nebulento(cases)
    if neb:
        rows.append(neb)

    rows.append(run_markov(
        cases, "markov  order=1",
        order=1, kneser_ney=True, backoff=True))
    rows.append(run_markov(
        cases, "markov  order=2  (default)",
        order=2, kneser_ney=True, backoff=True))
    rows.append(run_markov(
        cases, "markov  order=2  char_fallback",
        order=2, kneser_ney=True, backoff=True, char_fallback=True))

    summary(rows)
