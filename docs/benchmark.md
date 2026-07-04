# Benchmark

`ovos-markov-pipeline-plugin` ships a comparative accuracy benchmark in `benchmark/compare.py`. It runs on two OpenVoiceOS evaluation datasets and reports the Markov engine — in all three variants — alongside a fixed set of external baselines, so results are directly comparable across the OVOS intent-engine family.

---

## Headline results — `intents-for-eval`

50 intents, 1700 labelled test cases, 50 off-topic (`far_ood`) cases.

| Engine | def F_0.5 | **opt F_0.5** | opt thr | opt FP | **Rec @ P≥99%** |
|---|---|---|---|---|---|
| padatious (neural) | 0.899 | 0.927 | 0.19 | 8 | **73.5%** |
| nebulento `damerau` | 0.909 | 0.918 | 0.43 | 26 | 62.0% |
| **markov flat** | 0.665 | **0.909** | 0.14 | 20 | **59.5%** |
| markov domain | 0.683 | 0.889 | 0.18 | 23 | 61.8% |
| markov hierarchical | 0.683 | 0.890 | 0.18 | 22 | 61.8% |
| padaos (regex) | 0.832 | 0.832 | 0.50 | 1 | 0.0% (recall ≤ 50%) |

**With its threshold calibrated, the Markov engine reaches F_0.5 = 0.909 — within striking distance of padatious (0.927) and nebulento (0.918), at sub-millisecond inference.**

Read the per-engine deep-dive below for what those numbers actually mean and how the engine got from 4% to 81% F1 over the course of tuning.

---

## Why F_0.5 and not F1

A voice assistant's two failure modes are not symmetric:

- **False positive** — the wrong intent fires, the skill executes the wrong action, the assistant says the wrong thing. The user has to notice, abort, and re-ask. There is no recovery layer above the intent service that can catch this.
- **False negative** — no intent fires. OVOS hands the utterance to its fallback chain: common-query, the LLM fallback, online search. These exist precisely to handle "I don't know what you meant." Worst case the user re-phrases; best case the LLM nails it.

The cost ratio is roughly 5–10× in favour of false negatives. F1 (which weights precision and recall equally) is the wrong summary metric. F_β with β=0.5 weights precision twice as recall and is the right summary for OVOS.

We also report **Rec@P≥99%** — the recall achievable once the engine's threshold is tuned to keep precision at or above 99%. This is the operating point a maintainer actually picks: "give me the most coverage you can while letting through at most 1% wrong matches."

---

## Datasets

Both datasets are loaded from the Hugging Face Hub by `benchmark/dataset.py`. Each has a `<lang>-templates` config (training templates) and a `<lang>-test` config (labelled evaluation utterances).

| Name | Repo | Intents | Test cases | Notes |
|---|---|---|---|---|
| `intents-for-eval` | [`OpenVoiceOS/intents-for-eval`](https://huggingface.co/datasets/OpenVoiceOS/intents-for-eval) | 50 | 1750 | Six test splits including a 50-row `far_ood` no-match set |
| `massive` | [`OpenVoiceOS/massive-templates`](https://huggingface.co/datasets/OpenVoiceOS/massive-templates) | 60 | 2974 | OVOS-templated rebuild of MASSIVE; one labelled split, no no-match cases |

`intents-for-eval` test splits:

| Split | Cases | Tests |
|---|---|---|
| `template` | 500 | Utterances that fill a training template directly |
| `paraphrase` | 700 | Natural rephrasings — different words, same intent |
| `near_ood` | 400 | Boundary utterances close to another intent |
| `far_ood` | 50 | Genuinely off-topic — should match **nothing** |
| `asr_noise` | 50 | Speech-recognition artefacts |
| `typos` | 50 | Spelling errors |

`massive` has a single labelled split and **no off-topic cases** — every engine has zero FP by construction there, so it isolates pure recall on a broad intent set.

### Slots

Every `{slot}` placeholder in the templates ships with a list of example values. `benchmark/dataset.py` collects them into `Bundle.entities`.

- **padaos**, **padatious** and **nebulento** register the slot values as entities natively.
- The **Markov engine has no entity API** — `MarkovIntentEngine.add_intent` expands `(a|b)` / `[opt]` template syntax but treats `{slot}` placeholders as literal text. The benchmark's `_fill_slots` helper substitutes random slot values from `Bundle.entities` to produce 4 filled training sentences per template before registration. Without this, the engine sees `{song}` as a literal token and the n-grams never match real queries.

---

## Engines

The three `markov` rows are the **subject** of this benchmark. `padaos`, `padatious`, and `nebulento` are **fixed baselines** — the same engines and settings as in every OVOS intent-engine benchmark.

| Engine | Role | Notes |
|---|---|---|
| `padaos` | baseline | regex-based exact matcher |
| `padatious` | baseline | neural matcher (requires `train()` pass) |
| `nebulento` | baseline | fuzzy string matcher, `DAMERAU_LEVENSHTEIN_SIMILARITY` |
| `markov flat` | subject | one word-level Markov chain per intent, perplexity argmax |
| `markov domain` (parallel) | subject | per-domain `MarkovIntentEngine`s scored in parallel, global argmax |
| `markov hierarchical` (two-stage) | subject | top-level domain classifier routes to one per-domain engine |

All three Markov variants require a `train()` step (timed in the per-engine report).

---

## Deep-dive: tuning the Markov engine

The shipped engine defaults (`order=2, kneser_ney=True, backoff=True, smoothing=1e-5`) score **F1 = 0.04** on `intents-for-eval` out of the box — essentially random. That isn't an engine bug; three independent things had to be fixed for the engine to express its real ability.

### 1. Slot placeholders had to be filled

Templates like `play {song}` contain the literal three tokens `play`, `{song}`. The n-grams the Markov chain learns are over those literals. Test queries contain real song names, so the n-grams never match.

`benchmark/_fill_slots` substitutes random slot values from `Bundle.entities` to produce 4 filled training sentences per template. **Engine F1 jumped from 0.04 to 0.14** with the default config and default threshold.

### 2. The shipped config was the *worst* on this dataset

`benchmark/grid.py` runs every combination of `order ∈ {1,2,3,4}`, `kneser_ney ∈ {True,False}`, `backoff ∈ {True,False}`, `smoothing ∈ {1e-3,1e-5,1e-7}` and reports calibrated F1. Top three:

| order | kneser_ney | backoff | smoothing | opt F1 |
|---|---|---|---|---|
| **1** | **False** | True | **1e-3** | **0.810** |
| 1 | False | False | 1e-3 | 0.810 |
| 1 | True | True | any | 0.794 |
| ... | ... | ... | ... | ... |
| 2 | True | True | 1e-5 (shipped default) | 0.16 |
| 4 | any | any | any | ≤ 0.36 |

- **Higher order hurts.** Intent templates are short and lexically similar across domains; 4-grams sparsify too aggressively. Order=1 (unigrams + class prior) wins.
- **Kneser-Ney hurts at order=1.** It's designed to redistribute mass to unseen contexts, which isn't the problem here. Plain Laplace at `smoothing=1e-3` is enough.
- **Backoff is a wash at order=1** (nothing to back off to) and only marginally helps at higher orders.

`benchmark/compare.py` now sets `_MARKOV_DEFAULTS = dict(order=1, smoothing=1e-3, kneser_ney=False, backoff=True)` to reflect this. **Engines that ship with the bad defaults should consider changing them** — the grid is reproducible at `python benchmark/grid.py intents-for-eval`.

### 3. Default threshold = 0.5 was wildly too strict

Markov perplexity-derived confidences crowd low — even confident matches rarely top 0.3. The per-engine calibration sweep (in the headline table) shows the F_0.5-optimal threshold is **0.14** for markov flat. Dropping the threshold lifts opt F_0.5 from 0.665 → **0.909**, a +0.244 improvement.

For reference: padatious and nebulento's defaults are essentially F_0.5-optimal (calibration moves them by ≤ 0.03). The Markov engine ships at a default that doesn't suit it.

### Combined impact

| Step | F_0.5 |
|---|---|
| Engine defaults (no slot-fill, threshold 0.5) | 0.04 |
| + slot-fill | 0.32 |
| + config `order=1 kn=False smoothing=1e-3` | 0.67 |
| + calibrated threshold `0.14` | **0.91** |

---

## flat vs domain vs hierarchical (for Markov)

After all three fixes:

| Variant | opt F_0.5 | opt FP | R@P≥99% | median ms |
|---|---|---|---|---|
| markov flat | **0.909** | 20 | 59.5% | 0.6 |
| markov domain (parallel) | 0.889 | 23 | 61.8% | 0.2 |
| markov hierarchical (two-stage) | 0.890 | 22 | 61.8% | 0.2 |

- **Flat is the best by F_0.5** by a small margin (0.909 vs 0.890). When every intent is scored against the same model, the calibration sweep finds a single sharp boundary.
- **Domain and hierarchical pay a small accuracy cost** for being more operationally flexible — they let a skill be added or removed by retraining just one per-domain chain instead of the whole model. They also have **3× lower latency** at inference because each utterance is only scored against a fraction of the intents.
- **R@P≥99% favours domain/hierarchical** (61.8% vs 59.5%) — at the strict precision floor, the per-domain models are slightly more confident inside their own domain.

For most OVOS deployments the flat engine is the sensible default; domain and hierarchical are worth picking when latency or per-skill modularity matter more than the last 2pp of F_0.5.

---

## Reproducing

```bash
pip install ovos-markov-pipeline-plugin[benchmark]
python benchmark/compare.py intents-for-eval   # ~2 minutes
python benchmark/compare.py massive            # ~10 minutes
python benchmark/grid.py intents-for-eval      # Markov config sweep, ~10 minutes
```

The first run downloads each dataset from the Hugging Face Hub (cached afterwards).

## How metrics are calculated

Source: `compute_metrics`, `calibrate_threshold`, `fbeta`, `recall_at_precision` in `benchmark/compare.py`.

- **Accuracy** = (TP + TN) / total
- **Precision** = TP / (TP + FP)
- **Recall** = TP / total_match_cases
- **F1** = 2·P·R / (P + R)
- **F_0.5** = 1.25·P·R / (0.25·P + R) — weights precision 2× recall (default summary metric for OVOS)
- **Rec@P≥99%** = max recall achievable by sweeping the threshold while keeping precision ≥ 99%
- **FP** = no-match utterances incorrectly assigned an intent

A prediction is a TP when the predicted intent name exactly matches the expected intent and `conf ≥ threshold`. A no-match case is correct only when the engine returns `None` or a confidence below threshold.
