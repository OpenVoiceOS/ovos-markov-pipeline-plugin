#!/usr/bin/env python3
"""Benchmark ovos-markov-pipeline-plugin on the utterance_tags dataset.

Dataset: utterance_tags_v0.2.csv — 5504 utterances across 8+ intent classes
(COMMAND:ACTION, QUESTION:YESNO, QUESTION:QUERY, etc.)

Evaluates:
1. Accuracy, precision, recall, F1 at different train/test splits
2. Effect of order (1, 2, 3) on accuracy
3. Effect of Kneser-Ney vs Laplace smoothing
4. Effect of backoff
5. Confidence threshold calibration
6. Training time and inference throughput
"""

import csv
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# Adjust path if needed
DATA_PATH = (
    Path("/home/miro/PycharmProjects")
    / "Machine Learning Workspace"
    / "guided-categorical-embeddings"
    / "examples"
    / "questions_experiment_1"
    / "utterance_tags_v0.2.csv"
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ovos_markov_pipeline import MarkovIntentEngine, _ppx_to_confidence
from ovos_markov_pipeline.calibration import evaluate, find_optimal_thresholds


def load_data(path: str) -> List[Tuple[str, str]]:
    """Load (utterance, tag) pairs from CSV."""
    data: List[Tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                tag, sentence = row[0].strip(), row[1].strip()
                if tag and sentence:
                    data.append((sentence, tag))
    return data


def split_data(
    data: List[Tuple[str, str]], train_ratio: float = 0.8, seed: int = 42
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Stratified train/test split."""
    rng = random.Random(seed)
    by_tag: Dict[str, List[str]] = {}
    for sentence, tag in data:
        by_tag.setdefault(tag, []).append(sentence)

    train: List[Tuple[str, str]] = []
    test: List[Tuple[str, str]] = []
    for tag, sentences in by_tag.items():
        rng.shuffle(sentences)
        split = max(1, int(len(sentences) * train_ratio))
        for s in sentences[:split]:
            train.append((s, tag))
        for s in sentences[split:]:
            test.append((s, tag))

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def train_engine(
    train_data: List[Tuple[str, str]],
    order: int = 2,
    kneser_ney: bool = True,
    backoff: bool = True,
) -> MarkovIntentEngine:
    """Train a MarkovIntentEngine from (utterance, tag) pairs."""
    by_tag: Dict[str, List[str]] = {}
    for sentence, tag in train_data:
        by_tag.setdefault(tag, []).append(sentence)

    engine = MarkovIntentEngine(
        order=order,
        kneser_ney=kneser_ney,
        backoff=backoff,
        smoothing=1e-5,
    )
    for tag, sentences in by_tag.items():
        engine.add_intent(tag, sentences)
    engine.train()
    return engine


def benchmark_config(
    train_data: List[Tuple[str, str]],
    test_data: List[Tuple[str, str]],
    order: int,
    kneser_ney: bool,
    backoff: bool,
) -> Dict:
    """Train and evaluate a single configuration."""
    t0 = time.perf_counter()
    engine = train_engine(train_data, order=order, kneser_ney=kneser_ney, backoff=backoff)
    train_time = time.perf_counter() - t0

    # Evaluate (use subset for speed)
    eval_subset = test_data[:500] if len(test_data) > 500 else test_data
    metrics = evaluate(engine, eval_subset)

    # Throughput
    n_bench = min(200, len(test_data))
    t0 = time.perf_counter()
    for sentence, _ in test_data[:n_bench]:
        engine.calc_intents(sentence)
    infer_time = time.perf_counter() - t0
    throughput = n_bench / infer_time

    # Calibration (use subset for speed)
    thresholds = find_optimal_thresholds(engine, test_data[:300], steps=10)

    return {
        "order": order,
        "kneser_ney": kneser_ney,
        "backoff": backoff,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "train_time_ms": train_time * 1000,
        "throughput_qps": throughput,
        "best_threshold": thresholds["best_threshold"],
        "best_f1": thresholds["best_f1"],
    }


def main() -> None:
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Please update DATA_PATH in this script.")
        return

    print(f"Loading dataset: {DATA_PATH}")
    data = load_data(str(DATA_PATH))
    print(f"Total samples: {len(data)}")

    # Show class distribution
    tag_counts = Counter(tag for _, tag in data)
    print(f"\nClass distribution ({len(tag_counts)} classes):")
    for tag, count in tag_counts.most_common():
        print(f"  {tag:<30s} {count:>5d}  ({100*count/len(data):>5.1f}%)")

    # Split
    train, test = split_data(data, train_ratio=0.8)
    print(f"\nTrain: {len(train)}, Test: {len(test)}")

    # Benchmark configurations
    configs = [
        # order, kneser_ney, backoff
        (1, False, False),
        (1, True, False),
        (2, False, False),
        (2, True, True),
    ]

    print(f"\n{'Order':<6} {'KN':<5} {'BO':<5} {'Acc':<8} {'P':<8} {'R':<8} "
          f"{'F1':<8} {'Train ms':<10} {'QPS':<8} {'Best θ':<8} {'Best F1':<8}")
    print("-" * 95)

    results = []
    for order, kn, bo in configs:
        r = benchmark_config(train, test, order, kn, bo)
        results.append(r)
        print(
            f"{r['order']:<6d} {str(r['kneser_ney']):<5s} {str(r['backoff']):<5s} "
            f"{r['accuracy']:<8.3f} {r['precision']:<8.3f} {r['recall']:<8.3f} "
            f"{r['f1']:<8.3f} {r['train_time_ms']:<10.1f} {r['throughput_qps']:<8.0f} "
            f"{r['best_threshold']:<8.2f} {r['best_f1']:<8.3f}"
        )

    # Best config
    best = max(results, key=lambda r: r["f1"])
    print(f"\nBest config: order={best['order']}, KN={best['kneser_ney']}, "
          f"backoff={best['backoff']} → F1={best['f1']:.3f}")

    # Per-class analysis with best config
    print(f"\n--- Per-class breakdown (order={best['order']}, "
          f"KN={best['kneser_ney']}, backoff={best['backoff']}) ---")
    engine = train_engine(train, order=best["order"],
                          kneser_ney=best["kneser_ney"],
                          backoff=best["backoff"])

    per_class_correct: Dict[str, int] = {}
    per_class_total: Dict[str, int] = {}
    confusion: Dict[str, Dict[str, int]] = {}

    for sentence, expected in test:
        per_class_total[expected] = per_class_total.get(expected, 0) + 1
        scores = engine.calc_intents(sentence)
        predicted = scores[0][0] if scores else "NONE"
        if predicted == expected:
            per_class_correct[expected] = per_class_correct.get(expected, 0) + 1
        # Confusion
        confusion.setdefault(expected, {})
        confusion[expected][predicted] = confusion[expected].get(predicted, 0) + 1

    print(f"\n{'Class':<30s} {'Correct':<10s} {'Total':<8s} {'Acc':<8s}")
    print("-" * 56)
    for tag in sorted(per_class_total.keys()):
        c = per_class_correct.get(tag, 0)
        t = per_class_total[tag]
        print(f"{tag:<30s} {c:<10d} {t:<8d} {c/t:<8.3f}")

    # Top confusions
    print(f"\nTop confusions:")
    confusions_flat = []
    for true_tag, preds in confusion.items():
        for pred_tag, count in preds.items():
            if pred_tag != true_tag and count > 2:
                confusions_flat.append((count, true_tag, pred_tag))
    confusions_flat.sort(reverse=True)
    for count, true_tag, pred_tag in confusions_flat[:10]:
        print(f"  {true_tag} → {pred_tag}: {count}")


if __name__ == "__main__":
    main()
