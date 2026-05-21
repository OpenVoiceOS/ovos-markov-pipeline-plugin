"""Comparative accuracy + speed benchmark for the Markov intent engine.

Ports the nebulento benchmark harness (same dataset, same metrics) to
:class:`ovos_markov_pipeline.MarkovIntentEngine`, and runs nebulento's
best fuzzy strategy alongside as a same-machine baseline.

All engines train on the templates in ``INTENTS[name]["train"]`` and are
scored on the natural-language utterances in ``test_match`` plus the
``NO_MATCH_UTTERANCES`` negatives.

Confidence variants
-------------------
The Markov engine's shipped confidence is an *absolute* transform of one
intent's perplexity: ``conf = 1 / (1 + log(ppx))``. The ``relative``
runs instead rescore confidence as a softmax posterior over every
intent's log-likelihood, so the number reflects how far the winning
intent beat the rest. Argmax is unchanged (softmax is monotonic), so
this only moves the threshold-gated metrics, not argmax recall.

Reported per engine
-------------------
* **Argmax recall** — top-ranked intent correct, no threshold (ceiling).
* **AUC** — how well the confidence separates correct argmax matches
  from wrong / no-match cases.
* **F1 @0.5** — gated at the fixed nebulento threshold.
* **F1 @best** — gated at the F1-optimal threshold (swept per engine).

Usage
-----
    python benchmark/compare.py
"""
import contextlib
import io
import logging
import math
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


def relative_conf(scores, beta):
    """Rescore an intent ranking as a softmax posterior.

    ``scores`` is the engine's full ``[(name, conf), ...]`` list. Each
    shipped confidence ``conf = 1/(1+log(ppx))`` is inverted back to a
    log-likelihood ``ll = -log(ppx) = 1 - 1/conf``; a softmax over those
    (sharpened by ``beta``) yields a posterior per intent. Order is
    preserved — softmax is monotonic — so the argmax never changes.
    """
    lls = [(name, 1.0 - 1.0 / min(max(c, 1e-6), 1.0)) for name, c in scores]
    mx = max(ll for _, ll in lls)
    exps = [(name, math.exp(beta * (ll - mx))) for name, ll in lls]
    z = sum(e for _, e in exps) or 1.0
    return [(name, e / z) for name, e in exps]


def argmax_recall(results, cases):
    """Fraction of match cases whose top-ranked intent is correct."""
    tp = total = 0
    for (pred, _), (_, expected) in zip(results, cases):
        if expected is not None:
            total += 1
            if pred == expected:
                tp += 1
    return tp / total if total else 0.0


def auc(results, cases):
    """ROC-AUC of the confidence separating correct-argmax from the rest."""
    pos, neg = [], []
    for (pred, conf), (_, expected) in zip(results, cases):
        (pos if (expected is not None and pred == expected) else neg).append(conf)
    if not pos or not neg:
        return 0.0
    wins = ties = 0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def compute_metrics(results, cases, threshold):
    """Metrics with no-match gating: a prediction below *threshold* is None."""
    total = len(cases)
    match_n = sum(1 for _, e in cases if e is not None)
    tp = fp = fn = tn = 0
    for (pred, conf), (_, expected) in zip(results, cases):
        gated = pred if conf >= threshold else None
        if expected is not None:
            tp, fn = (tp + 1, fn) if gated == expected else (tp, fn + 1)
        else:
            fp, tn = (fp + 1, tn) if gated is not None else (fp, tn + 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / match_n if match_n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(
        accuracy=(tp + tn) / total, precision=precision, recall=recall,
        f1=f1, fp=fp, fn=fn, match_n=match_n, nomatch_n=total - match_n)


def best_f1(results, cases):
    """Sweep the threshold; return (best_f1, threshold, metrics-at-best)."""
    best = (0.0, 0.0, compute_metrics(results, cases, 0.0))
    for i in range(101):
        thr = i / 100
        m = compute_metrics(results, cases, thr)
        if m["f1"] > best[0]:
            best = (m["f1"], thr, m)
    return best


def print_report(label, results, cases, latencies, train_ms=None):
    s = sorted(latencies)
    m05 = compute_metrics(results, cases, THRESHOLD)
    bf1, bthr, mb = best_f1(results, cases)
    total = m05["match_n"] + m05["nomatch_n"]
    print(f"{'=' * 64}")
    print(f"  {label}")
    print(f"{'=' * 64}")
    if train_ms is not None:
        print(f"  Train time     : {train_ms:.0f} ms")
    print(f"  Argmax recall  : {argmax_recall(results, cases):.1%}  "
          f"(top intent correct, no threshold)")
    print(f"  Confidence AUC : {auc(results, cases):.3f}")
    print(f"  F1 @0.5        : {m05['f1']:.3f}  "
          f"(acc {m05['accuracy']:.1%}, recall {m05['recall']:.1%}, "
          f"FP {m05['fp']}/{m05['nomatch_n']})")
    print(f"  F1 @best       : {bf1:.3f}  at threshold {bthr:.2f}  "
          f"(acc {mb['accuracy']:.1%}, recall {mb['recall']:.1%}, "
          f"prec {mb['precision']:.1%}, FP {mb['fp']}/{mb['nomatch_n']})")
    print(f"  Latency        : median={statistics.median(latencies):.2f}ms  "
          f"p95={s[int(len(s) * .95)]:.2f}ms  max={s[-1]:.2f}ms")


# ── engine runners ─────────────────────────────────────────────────────────

def run_markov(cases, label, beta=None, **engine_kwargs):
    """Run MarkovIntentEngine. If *beta* is set, rescore confidence
    relatively via :func:`relative_conf` with that softmax temperature."""
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
        if beta is not None and scores:
            scores = relative_conf(scores, beta)
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
    print(f"\n\n{'─' * 92}")
    print(f"  {'Engine':<34} {'Argmax':>7} {'AUC':>6} {'F1@0.5':>7} "
          f"{'F1@best':>8} {'(thr)':>7}  {'Median':>9}")
    print(f"{'─' * 92}")
    for label, results, cases, latencies in rows:
        m05 = compute_metrics(results, cases, THRESHOLD)
        bf1, bthr, _ = best_f1(results, cases)
        print(f"  {label:<34} {argmax_recall(results, cases):>6.1%} "
              f"{auc(results, cases):>6.3f} {m05['f1']:>7.3f} "
              f"{bf1:>8.3f} {bthr:>7.2f}  "
              f"{statistics.median(latencies):>6.2f}ms")
    print(f"{'─' * 92}")
    print("  Argmax = top intent correct (no threshold)")
    print("  F1@0.5 = gated at fixed 0.5 | F1@best = gated at F1-optimal threshold")


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
        cases, "markov  order=1  absolute conf",
        order=1, kneser_ney=True, backoff=True))
    rows.append(run_markov(
        cases, "markov  order=1  relative b=3",
        beta=3.0, order=1, kneser_ney=True, backoff=True))
    rows.append(run_markov(
        cases, "markov  order=2  char_fb  absolute",
        order=2, kneser_ney=True, backoff=True, char_fallback=True))
    rows.append(run_markov(
        cases, "markov  order=2  char_fb  relative b=3",
        beta=3.0, order=2, kneser_ney=True, backoff=True, char_fallback=True))

    summary(rows)
