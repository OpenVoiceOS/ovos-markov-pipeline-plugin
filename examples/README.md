# Examples

## Scripts

| Script | Description |
|--------|-------------|
| `basic_intent_engine.py` | Standalone intent classification + slot extraction without OVOS |
| `benchmark_utterance_tags.py` | Benchmark on 5504-utterance, 11-class dataset |

## Running

```bash
python examples/basic_intent_engine.py
python examples/benchmark_utterance_tags.py
```

## Benchmark Results (utterance_tags_v0.2.csv)

| Config | Accuracy | Train | QPS |
|--------|----------|-------|-----|
| order=1, Laplace | 59.2% | 156ms | 663 |
| **order=1, Kneser-Ney** | **73.3%** | 265ms | 367 |
| order=2, Laplace | 48.6% | 169ms | 1068 |
| order=2, KN+backoff | 59.1% | 844ms | 410 |

Best per-class: QUESTION:REQUEST 91%, QUESTION:QUERY 87%, SENTENCE:EXCLAMATION 87%.
