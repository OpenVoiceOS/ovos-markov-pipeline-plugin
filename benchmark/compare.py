"""
Comparative accuracy + speed benchmark across intent engines.

Engines
-------
padaos      – regex-based matcher                       (baseline)
padatious   – neural-network matcher, requires training (baseline)
nebulento   – fuzzy string matching engine              (baseline)
markov      – this repo's Markov chain perplexity engine, in three shapes:
              flat / domain (parallel) / hierarchical (two-stage)

The three ``markov`` rows are the subject of this benchmark; padaos,
padatious and nebulento are fixed external baselines so results are
comparable across the OVOS intent-engine family.

Every engine here is a template / sample matcher: it trains on example
sentences, not keyword vocabularies. They are evaluated on two OpenVoiceOS
datasets — ``intents-for-eval`` and ``massive`` — each engine training on the
``<lang>-templates`` config and evaluating on ``<lang>-test``. See
``benchmark/dataset.py``.

Slots
-----
padaos, padatious and nebulento register every ``{slot}`` placeholder's
example values as a named entity. The Markov engine has no entity API: it
keeps ``{slot}`` placeholders verbatim as features, so the markov rows train
on the raw templates. See ``docs/benchmark.md``.

Usage
-----
    python benchmark/compare.py                  # both datasets
    python benchmark/compare.py intents-for-eval # one dataset
"""
import sys
import time
import tempfile
import statistics
import logging
import random
import re
from collections import defaultdict

from benchmark.dataset import DATASETS, load

_SLOT_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _fill_slots(template, entities, max_fills=4, rng=None):
    """Expand a template's ``{slot}`` placeholders with concrete values.

    Each engine here is a sample matcher: a literal ``{song}`` token in
    training never matches a real song name at eval time. Substituting
    concrete values from ``entities`` gives the engine realistic training
    sentences. A small per-template cap keeps the training set bounded.
    """
    rng = rng or random.Random(0)
    slots = _SLOT_RE.findall(template)
    if not slots:
        return [template]
    out, seen = [], set()
    for _ in range(max_fills):
        s = template
        for slot in slots:
            values = entities.get(slot) or [slot]
            s = s.replace("{" + slot + "}", rng.choice(values), 1)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _filled_train(bundle, name):
    """Return slot-filled training sentences for a single intent."""
    rng = random.Random(hash(name) & 0xFFFFFFFF)
    out = []
    for tmpl in bundle.intents[name]["train"]:
        out.extend(_fill_slots(tmpl, bundle.entities, max_fills=4, rng=rng))
    return out


def fbeta(precision, recall, beta=0.5):
    """F_β score. β<1 weights precision over recall — appropriate for voice
    assistants where a false positive (wrong skill fires) is unrecoverable
    while a false negative falls through to fallback handlers (LLM, etc.).
    """
    b2 = beta * beta
    denom = b2 * precision + recall
    return ((1 + b2) * precision * recall / denom) if denom else 0.0


def recall_at_precision(results_no_thresh, cases, p_floor=0.99, step=0.01):
    """Sweep the threshold and return the max recall achievable while
    keeping precision >= ``p_floor`` (and the threshold that gets there).
    """
    best_r, best_t = 0.0, None
    t = 0.0
    while t <= 1.0 + 1e-9:
        thresholded = [
            (lbl if c >= t else None, c) for (lbl, c) in results_no_thresh
        ]
        m = compute_metrics(thresholded, cases)
        if m["precision"] >= p_floor and m["recall"] > best_r:
            best_r, best_t = m["recall"], round(t, 4)
        t += step
    return best_r, best_t


def calibrate_threshold(results_no_thresh, cases, step=0.01, metric="f0.5"):
    """Sweep threshold and return (best_threshold, best_metric_value, best_metrics).

    ``metric`` selects the scalar to maximise:
      - ``"f1"``   — standard F1.
      - ``"f0.5"`` — F_β=0.5 (precision-weighted; OVOS default).

    ``results_no_thresh`` is a list of ``(predicted_label, conf)`` where
    ``predicted_label`` is the engine's argmax (never None) and ``conf`` is its
    score. We re-apply the threshold ourselves: if ``conf < t`` we treat the
    case as a no-match.
    """
    def score(m):
        return m["f1"] if metric == "f1" else fbeta(m["precision"], m["recall"], 0.5)

    best = (0.0, -1.0, None)
    t = 0.0
    while t <= 1.0 + 1e-9:
        thresholded = [
            (lbl if c >= t else None, c) for (lbl, c) in results_no_thresh
        ]
        m = compute_metrics(thresholded, cases)
        s = score(m)
        if s > best[1]:
            best = (round(t, 4), s, m)
        t += step
    return best

logging.disable(logging.CRITICAL)

_CI_MODE = "--ci" in sys.argv


# ── shared helpers ─────────────────────────────────────────────────────────

def all_cases(bundle):
    """Flatten a :class:`~benchmark.dataset.Bundle` into ``(utterance, expected)``."""
    cases = []
    for name, data in bundle.intents.items():
        for utt in data["test_match"]:
            cases.append((utt, name))
    for utt in bundle.no_match:
        cases.append((utt, None))
    return cases


def compute_metrics(results, cases):
    total     = len(cases)
    match_n   = sum(1 for _, e in cases if e is not None)
    nomatch_n = total - match_n
    tp = fp = fn = tn = 0
    per_tp = defaultdict(int)
    per_fn = defaultdict(int)
    per_fp = defaultdict(int)
    wrong  = []
    for (predicted, conf), (utt, expected) in zip(results, cases):
        if expected is not None:
            if predicted == expected:
                tp += 1
                per_tp[expected] += 1
            else:
                fn += 1
                per_fn[expected] += 1
                wrong.append((utt, expected, predicted, conf))
        else:
            if predicted is not None:
                fp += 1
                per_fp[predicted] += 1
                wrong.append((utt, expected, predicted, conf))
            else:
                tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / match_n   if match_n   else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(
        accuracy=(tp + tn) / total if total else 0.0,
        precision=precision, recall=recall, f1=f1,
        fp=fp, fn=fn, match_n=match_n, nomatch_n=nomatch_n,
        per_tp=per_tp, per_fn=per_fn, per_fp=per_fp, wrong=wrong,
    )


def _stats_lines(label, metrics, latencies, intents, train_ms=None):
    s = sorted(latencies)
    total = metrics['match_n'] + metrics['nomatch_n']
    nomatch_n = metrics['nomatch_n']
    match_n = metrics['match_n']
    fp_pct = f"  ({metrics['fp']/nomatch_n:.0%} of no-match)" if nomatch_n else ""
    fn_pct = f"  ({metrics['fn']/match_n:.0%} of match)" if match_n else ""
    lines = [
        f"{'='*64}",
        f"  {label}",
        f"{'='*64}",
    ]
    if train_ms is not None:
        lines.append(f"  Train time: {train_ms:.0f} ms")
    lines += [
        f"  Accuracy  : {metrics['accuracy']:.1%}  ({int(metrics['accuracy']*total)}/{total})",
        f"  Precision : {metrics['precision']:.1%}",
        f"  Recall    : {metrics['recall']:.1%}",
        f"  F1        : {metrics['f1']:.3f}",
        f"  FP        : {metrics['fp']} / {nomatch_n}{fp_pct}",
        f"  FN        : {metrics['fn']} / {match_n}{fn_pct}",
        f"  Latency   : median={statistics.median(latencies):.2f}ms  "
        f"p95={s[int(len(s)*.95)]:.2f}ms  max={s[-1]:.2f}ms",
    ]
    issues = sorted(set(metrics['per_fn']) | set(metrics['per_fp']))
    if issues:
        lines.append("")
        lines.append("  Per-intent (issues only):")
        for i in sorted(intents):
            fn = metrics['per_fn'].get(i, 0)
            fp = metrics['per_fp'].get(i, 0)
            tp = metrics['per_tp'].get(i, 0)
            if fn or fp:
                rec = tp / (tp + fn) if (tp + fn) else 0
                lines.append(f"    {i:<28}  recall={rec:.0%}  fn={fn}  fp={fp}")
    return lines


def print_report(label, metrics, latencies, intents, train_ms=None):
    lines = _stats_lines(label, metrics, latencies, intents, train_ms)
    if _CI_MODE:
        acc = metrics['accuracy']
        fp  = metrics['fp']
        med = statistics.median(latencies)
        print("<details>")
        print(f"<summary><b>{label}</b> &mdash; acc {acc:.1%} &middot; "
              f"FP {fp} &middot; median {med:.2f}ms</summary>")
        print()
        print("```text")
        for line in lines:
            print(line)
        print("```")
        print()
        print("</details>")
        print()
    else:
        for line in lines:
            print(line)


# ── baseline engine runners ────────────────────────────────────────────────

def run_padaos(bundle, cases):
    import padaos
    c = padaos.IntentContainer()
    for entity_name, samples in bundle.entities.items():
        c.add_entity(entity_name, samples)
    for name, data in bundle.intents.items():
        c.add_intent(name, data["train"])
    t0 = time.perf_counter()
    c.compile()
    train_ms = (time.perf_counter() - t0) * 1000

    results, latencies = [], []
    for utt, _ in cases:
        q = utt.lower().strip()
        t0 = time.perf_counter()
        r  = c.calc_intent(q)
        latencies.append((time.perf_counter() - t0) * 1000)
        results.append((r.get("name"), 1.0 if r.get("name") else 0.0))

    m = compute_metrics(results, cases)
    print_report("padaos  (regex, no fuzz)", m, latencies, bundle.intents, train_ms)
    return m, statistics.median(latencies), statistics.mean(latencies), train_ms, None


def run_padatious(bundle, cases, threshold=0.5):
    from padatious import IntentContainer as PC
    with tempfile.TemporaryDirectory() as d:
        c = PC(cache_dir=d)
        for entity_name, samples in bundle.entities.items():
            c.add_entity(entity_name, samples)
        for name, data in bundle.intents.items():
            c.add_intent(name, data["train"])
        t0 = time.perf_counter()
        c.train(single_thread=True, debug=False)
        train_ms = (time.perf_counter() - t0) * 1000

        raw, latencies = [], []
        for utt, _ in cases:
            t0 = time.perf_counter()
            r  = c.calc_intent(utt.lower().strip())
            latencies.append((time.perf_counter() - t0) * 1000)
            raw.append((r.name if r else None, r.conf if r else 0.0))

    results = [(lbl if c >= threshold else None, c) for (lbl, c) in raw]
    m = compute_metrics(results, cases)
    print_report(f"padatious  (neural, threshold={threshold})", m, latencies,
                 bundle.intents, train_ms)
    return m, statistics.median(latencies), statistics.mean(latencies), train_ms, raw


def run_nebulento(bundle, cases, threshold=0.5):
    from nebulento import IntentContainer
    c = IntentContainer()
    for entity_name, samples in bundle.entities.items():
        c.add_entity(entity_name, samples)
    for name, data in bundle.intents.items():
        try:
            c.add_intent(name, data["train"])
        except Exception as e:
            print(f"[SKIP] nebulento — registration failed: {e}")
            return None

    raw, latencies = [], []
    for utt, _ in cases:
        t0 = time.perf_counter()
        r  = c.calc_intent(utt)
        latencies.append((time.perf_counter() - t0) * 1000)
        raw.append((r.get("name") if r else None, r.get("conf", 0.0) if r else 0.0))

    results = [(lbl if c >= threshold else None, c) for (lbl, c) in raw]
    m = compute_metrics(results, cases)
    print_report("nebulento  (fuzzy, damerau-levenshtein)", m, latencies, bundle.intents)
    return m, statistics.median(latencies), statistics.mean(latencies), None, raw


# ── markov engine runners (the subject of this benchmark) ──────────────────

# Engine ships with order=2, kneser_ney=True, smoothing=1e-5 — the grid sweep
# in benchmark/grid.py shows that's the *worst* combination on intents-for-eval.
# Defaults here override to the grid winner so the benchmark reports the
# engine's real ceiling, not its mis-tuned default.
_MARKOV_DEFAULTS = dict(order=1, smoothing=1e-3, kneser_ney=False, backoff=True)


def run_markov_flat(bundle, cases, threshold=0.5, config=None, label_extra=""):
    from ovos_markov_pipeline import MarkovIntentEngine
    cfg = dict(_MARKOV_DEFAULTS, **(config or {}))
    engine = MarkovIntentEngine(**cfg)
    for name in bundle.intents:
        engine.add_intent(name, _filled_train(bundle, name))
    t0 = time.perf_counter()
    engine.train()
    train_ms = (time.perf_counter() - t0) * 1000

    raw, latencies = [], []
    for utt, _ in cases:
        t0 = time.perf_counter()
        scores = engine.calc_intents(utt)
        latencies.append((time.perf_counter() - t0) * 1000)
        if scores:
            raw.append((scores[0][0], scores[0][1]))
        else:
            raw.append((None, 0.0))

    results = [(lbl if c >= threshold else None, c) for (lbl, c) in raw]
    m = compute_metrics(results, cases)
    print_report(f"markov  flat{label_extra}  (threshold={threshold})", m, latencies,
                 bundle.intents, train_ms)
    return m, statistics.median(latencies), statistics.mean(latencies), train_ms, raw


def run_markov_domain(bundle, cases, threshold=0.5, config=None):
    from ovos_markov_pipeline import DomainMarkovIntentEngine
    cfg = dict(_MARKOV_DEFAULTS, **(config or {}))
    engine = DomainMarkovIntentEngine(**cfg)
    for name, data in bundle.intents.items():
        engine.register_domain_intent(data["domain"], name, _filled_train(bundle, name))
    t0 = time.perf_counter()
    engine.train()
    train_ms = (time.perf_counter() - t0) * 1000

    raw, latencies = [], []
    for utt, _ in cases:
        t0 = time.perf_counter()
        scores = engine.calc_intents(utt)
        latencies.append((time.perf_counter() - t0) * 1000)
        if scores:
            raw.append((scores[0][0], scores[0][1]))
        else:
            raw.append((None, 0.0))

    results = [(lbl if c >= threshold else None, c) for (lbl, c) in raw]
    m = compute_metrics(results, cases)
    print_report(f"markov  domain  (parallel, threshold={threshold})", m,
                 latencies, bundle.intents, train_ms)
    return m, statistics.median(latencies), statistics.mean(latencies), train_ms, raw


def run_markov_hierarchical(bundle, cases, threshold=0.5, domain_threshold=0.0, config=None):
    from ovos_markov_pipeline import HierarchicalMarkovIntentEngine
    cfg = dict(_MARKOV_DEFAULTS, **(config or {}))
    engine = HierarchicalMarkovIntentEngine(domain_threshold=domain_threshold, **cfg)
    for name, data in bundle.intents.items():
        engine.register_domain_intent(data["domain"], name, _filled_train(bundle, name))
    t0 = time.perf_counter()
    engine.train()
    train_ms = (time.perf_counter() - t0) * 1000

    raw, latencies = [], []
    for utt, _ in cases:
        t0 = time.perf_counter()
        scores = engine.calc_intents(utt)
        latencies.append((time.perf_counter() - t0) * 1000)
        if scores:
            raw.append((scores[0][0], scores[0][1]))
        else:
            raw.append((None, 0.0))

    results = [(lbl if c >= threshold else None, c) for (lbl, c) in raw]
    m = compute_metrics(results, cases)
    print_report(f"markov  hierarchical  (two-stage, threshold={threshold})", m,
                 latencies, bundle.intents, train_ms)
    return m, statistics.median(latencies), statistics.mean(latencies), train_ms, raw


# ── summary table ──────────────────────────────────────────────────────────

def summary(title, rows):
    """rows: list of (label, metrics, median_lat_ms, mean_lat_ms, train_ms_or_None)"""
    if _CI_MODE:
        print(f"## {title}\n")
        print("| Engine | Acc | Prec | Recall | F1 | FP | Median |")
        print("|---|---|---|---|---|---|---|")
        for label, m, median_lat, mean_lat, _ in rows:
            print(f"| {label} | {m['accuracy']:.1%} | {m['precision']:.1%} | "
                  f"{m['recall']:.1%} | {m['f1']:.3f} | {m['fp']} | {median_lat:.2f}ms |")
        print()
        print("_FP = false positives on no-match_")
    else:
        print(f"\n\n{'─'*84}")
        print(f"  {title}")
        print(f"  {'Engine':<36} {'Acc':>6} {'Prec':>6} {'Recall':>7} {'F1':>6}  "
              f"{'FP':>4}  {'Median':>8}  {'Mean':>8}")
        print(f"{'─'*84}")
        for label, m, median_lat, mean_lat, train_ms in rows:
            print(f"  {label:<36} {m['accuracy']:>5.1%} {m['precision']:>5.1%} "
                  f"{m['recall']:>6.1%} {m['f1']:>5.3f}  {m['fp']:>4}  "
                  f"{median_lat:>6.2f}ms  {mean_lat:>6.2f}ms")
        print(f"{'─'*84}")
        print("  FP = false positives on no-match | Median/Mean = query latency in ms")


# ── main ───────────────────────────────────────────────────────────────────

def run_dataset(name):
    bundle = load(name)
    cases = all_cases(bundle)
    match_n = sum(1 for _, e in cases if e is not None)
    print(f"\nDataset : {bundle.repo}  ({bundle.lang})")
    print(f"Cases   : {len(cases)}  ({match_n} match, {len(cases)-match_n} no-match)")
    print(f"Intents : {len(bundle.intents)}  across {len(bundle.domains)} domains")
    print("Splits  : " + ", ".join(f"{k}={len(v)}" for k, v in bundle.splits.items()))

    rows = []
    cal_rows = []  # (label, default_thr, default_f1, default_fp, opt_thr, opt_f1, opt_fp)

    # baselines — fixed across the OVOS intent-engine family
    m, lat, mean_lat, tr, raw = run_padaos(bundle, cases)
    rows.append(("padaos  (regex)", m, lat, mean_lat, tr))
    # padaos has no conf knob — skip calibration

    m, lat, mean_lat, tr, raw = run_padatious(bundle, cases, threshold=0.5)
    rows.append(("padatious  neural  threshold=0.5", m, lat, mean_lat, tr))
    cal_rows.append(_calibrate_row("padatious", raw, cases, 0.5, m))

    neb = run_nebulento(bundle, cases, threshold=0.5)
    if neb is not None:
        m, lat, mean_lat, tr, raw = neb
        rows.append(("nebulento  damerau-levenshtein", m, lat, mean_lat, tr))
        cal_rows.append(_calibrate_row("nebulento", raw, cases, 0.5, m))

    # subject — this repo's Markov engine in three shapes (default config)
    m, lat, mean_lat, tr, raw = run_markov_flat(bundle, cases, threshold=0.5)
    rows.append(("markov  flat", m, lat, mean_lat, tr))
    cal_rows.append(_calibrate_row("markov flat", raw, cases, 0.5, m))

    m, lat, mean_lat, tr, raw = run_markov_domain(bundle, cases, threshold=0.5)
    rows.append(("markov  domain  (parallel)", m, lat, mean_lat, tr))
    cal_rows.append(_calibrate_row("markov domain", raw, cases, 0.5, m))

    m, lat, mean_lat, tr, raw = run_markov_hierarchical(bundle, cases, threshold=0.5)
    rows.append(("markov  hierarchical  (two-stage)", m, lat, mean_lat, tr))
    cal_rows.append(_calibrate_row("markov hierarchical", raw, cases, 0.5, m))

    summary(f"{name}  —  {bundle.repo}", rows)
    _print_calibration_table(cal_rows)


def _calibrate_row(label, raw, cases, default_thr, default_metrics):
    """Compute the calibration row for one engine.

    Returns a tuple holding the default and F_0.5-optimal thresholds plus the
    metrics that matter to a voice assistant: F1, F_0.5 (precision-weighted),
    FP count, and Rec@P>=0.99 (the recall achievable while keeping at most 1%
    false positives — the operating point a production OVOS install would use).
    """
    df1 = default_metrics["f1"]
    dfp = default_metrics["fp"]
    df05 = fbeta(default_metrics["precision"], default_metrics["recall"], 0.5)
    opt_thr, _, opt_metrics = calibrate_threshold(raw, cases, step=0.01,
                                                  metric="f0.5")
    of1 = opt_metrics["f1"]
    ofp = opt_metrics["fp"]
    of05 = fbeta(opt_metrics["precision"], opt_metrics["recall"], 0.5)
    rec_at_p, rec_thr = recall_at_precision(raw, cases, p_floor=0.99, step=0.01)
    return (label, default_thr, df1, df05, dfp,
            opt_thr, of1, of05, ofp, rec_at_p, rec_thr)


def _print_calibration_table(rows):
    print(f"\n{'─'*108}")
    print("  Per-engine threshold calibration  (sweep 0..1 step 0.01, max F_0.5)")
    print(f"  {'Engine':<24} {'def_thr':>7} {'def_F1':>7} {'defF.5':>7} {'def_FP':>6}"
          f"  {'opt_thr':>7} {'opt_F1':>7} {'optF.5':>7} {'opt_FP':>6}"
          f"  {'R@P99':>6} {'thr':>5}")
    print(f"{'─'*108}")
    for r in rows:
        (label, dthr, df1, df05, dfp,
         othr, of1, of05, ofp, rec_at_p, rec_thr) = r
        rec_t = f"{rec_thr:.2f}" if rec_thr is not None else "--"
        print(f"  {label:<24} {dthr:>7.2f} {df1:>7.3f} {df05:>7.3f} {dfp:>6d}"
              f"  {othr:>7.2f} {of1:>7.3f} {of05:>7.3f} {ofp:>6d}"
              f"  {rec_at_p:>6.1%} {rec_t:>5}")
    print(f"{'─'*108}")
    print("  F_0.5 (β=0.5) weights precision 2x recall — the right summary metric")
    print("  for OVOS, where a wrong intent is unrecoverable but a missed intent")
    print("  falls through to fallback handlers. R@P99 = max recall achievable")
    print("  with the threshold tuned to keep precision >= 99%.")


if __name__ == "__main__":
    selected = [a for a in sys.argv[1:] if not a.startswith("-")]
    targets = selected or list(DATASETS)
    for dataset_name in targets:
        if dataset_name not in DATASETS:
            print(f"unknown dataset {dataset_name!r}; choose from {list(DATASETS)}")
            continue
        run_dataset(dataset_name)
