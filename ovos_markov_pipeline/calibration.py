"""Confidence calibration for perplexity-based scoring.

Provides tools to evaluate and tune the perplexity-to-confidence mapping
against labelled evaluation data.
"""

from typing import Dict, List, Tuple

from ovos_markov_pipeline import MarkovIntentEngine


def evaluate(
    engine: MarkovIntentEngine,
    eval_data: List[Tuple[str, str]],
) -> Dict[str, float]:
    """Evaluate intent classification accuracy on labelled data.

    Args:
        engine: A trained :class:`MarkovIntentEngine`.
        eval_data: List of ``(utterance, expected_intent)`` pairs.

    Returns:
        Dict with ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``avg_confidence_correct``, ``avg_confidence_incorrect``.
    """
    correct = 0
    total = len(eval_data)
    conf_correct: List[float] = []
    conf_incorrect: List[float] = []

    # Per-intent TP/FP/FN
    tp: Dict[str, int] = {}
    fp: Dict[str, int] = {}
    fn: Dict[str, int] = {}

    for utterance, expected in eval_data:
        scores = engine.calc_intents(utterance)
        if not scores:
            fn[expected] = fn.get(expected, 0) + 1
            continue

        predicted, conf = scores[0]
        if predicted == expected:
            correct += 1
            conf_correct.append(conf)
            tp[expected] = tp.get(expected, 0) + 1
        else:
            conf_incorrect.append(conf)
            fp[predicted] = fp.get(predicted, 0) + 1
            fn[expected] = fn.get(expected, 0) + 1

    accuracy = correct / max(total, 1)

    # Macro-averaged precision, recall, F1
    all_intents = set(list(tp.keys()) + list(fp.keys()) + list(fn.keys()))
    precisions: List[float] = []
    recalls: List[float] = []
    for intent in all_intents:
        t = tp.get(intent, 0)
        f = fp.get(intent, 0)
        n = fn.get(intent, 0)
        p = t / max(t + f, 1)
        r = t / max(t + n, 1)
        precisions.append(p)
        recalls.append(r)

    avg_p = sum(precisions) / max(len(precisions), 1)
    avg_r = sum(recalls) / max(len(recalls), 1)
    f1 = 2 * avg_p * avg_r / max(avg_p + avg_r, 1e-10)

    return {
        "accuracy": accuracy,
        "precision": avg_p,
        "recall": avg_r,
        "f1": f1,
        "avg_confidence_correct": (sum(conf_correct) / len(conf_correct) if conf_correct else 0.0),
        "avg_confidence_incorrect": (
            sum(conf_incorrect) / len(conf_incorrect) if conf_incorrect else 0.0
        ),
        "total": total,
        "correct": correct,
    }


def find_optimal_thresholds(
    engine: MarkovIntentEngine,
    eval_data: List[Tuple[str, str]],
    steps: int = 20,
) -> Dict[str, float]:
    """Search for confidence thresholds that maximize F1 at each tier.

    Args:
        engine: A trained :class:`MarkovIntentEngine`.
        eval_data: List of ``(utterance, expected_intent)`` pairs.
        steps: Number of threshold values to test.

    Returns:
        Dict with ``best_threshold``, ``best_f1``, ``precision_at_best``,
        ``recall_at_best``.
    """
    best_f1 = 0.0
    best_threshold = 0.5
    best_p = 0.0
    best_r = 0.0

    for i in range(steps + 1):
        threshold = i / steps

        correct = 0
        predicted_count = 0
        actual_count = len(eval_data)

        for utterance, expected in eval_data:
            scores = engine.calc_intents(utterance)
            if not scores or scores[0][1] < threshold:
                continue
            predicted_count += 1
            if scores[0][0] == expected:
                correct += 1

        p = correct / max(predicted_count, 1)
        r = correct / max(actual_count, 1)
        f1 = 2 * p * r / max(p + r, 1e-10)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_p = p
            best_r = r

    return {
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "precision_at_best": best_p,
        "recall_at_best": best_r,
    }
