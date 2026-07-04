"""Sweep markov engine configs on intents-for-eval, printing F1 (default + calibrated)."""
import sys
import logging
import itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.disable(logging.CRITICAL)

from benchmark.dataset import load
from benchmark.compare import (
    all_cases,
    compute_metrics,
    calibrate_threshold,
    _filled_train,
)


def run_one(bundle, cases, order, smoothing, kneser_ney, backoff):
    from ovos_markov_pipeline import MarkovIntentEngine
    engine = MarkovIntentEngine(
        order=order, smoothing=smoothing,
        kneser_ney=kneser_ney, backoff=backoff,
    )
    for name in bundle.intents:
        engine.add_intent(name, _filled_train(bundle, name))
    engine.train()
    raw = []
    for utt, _ in cases:
        scores = engine.calc_intents(utt)
        if scores:
            raw.append((scores[0][0], scores[0][1]))
        else:
            raw.append((None, 0.0))
    # default 0.5
    res05 = [(l if c >= 0.5 else None, c) for (l, c) in raw]
    m05 = compute_metrics(res05, cases)
    opt_thr, opt_f1, opt_fp, _ = calibrate_threshold(raw, cases, step=0.02)
    return m05["f1"], m05["fp"], opt_thr, opt_f1, opt_fp


def main():
    bundle = load("intents-for-eval")
    cases = all_cases(bundle)
    print(f"order  kn  bo  smooth   def_F1  def_FP  opt_thr  opt_F1  opt_FP")
    print("-" * 72)
    orders = [1, 2, 3, 4]
    smoothings = [1e-3, 1e-5, 1e-7]
    kns = [True, False]
    bos = [True, False]
    rows = []
    for order, kn, bo, sm in itertools.product(orders, kns, bos, smoothings):
        try:
            df1, dfp, othr, of1, ofp = run_one(bundle, cases, order, sm, kn, bo)
        except Exception as e:
            print(f"{order:>5}  {str(kn)[0]}   {str(bo)[0]}   {sm:.0e}  ERROR {e}")
            continue
        rows.append((of1, order, kn, bo, sm, df1, dfp, othr, ofp))
        print(f"{order:>5}  {str(kn)[0]}   {str(bo)[0]}   {sm:.0e}  {df1:>6.3f}  {dfp:>6d}  {othr:>6.2f}  {of1:>6.3f}  {ofp:>6d}")
    rows.sort(reverse=True)
    print("\nTOP 5 by opt_F1:")
    for r in rows[:5]:
        of1, order, kn, bo, sm, df1, dfp, othr, ofp = r
        print(f"  order={order} kn={kn} backoff={bo} smooth={sm:.0e}  "
              f"opt_F1={of1:.3f} @ thr={othr:.2f} (fp={ofp})  def_F1={df1:.3f}")


if __name__ == "__main__":
    main()
