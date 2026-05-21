# Entity extraction

Once an intent is matched, a skill often needs the *slot values* inside the
utterance — the duration in "set a timer for **five minutes**", the city in
"weather in **Lisbon**". The `SlotExtractor` class extracts those slots with a
Hidden Markov Model and BIO tagging.

## BIO tagging

BIO ("Begin, Inside, Outside") labels each token of an utterance:

```text
set  a  timer  for  five     minutes
O    O  O      O    B-time   I-time
```

- `B-<slot>` — the first token of a slot value.
- `I-<slot>` — a continuation token of the same slot.
- `O` — a token that is not part of any slot.

Consecutive `B-time` / `I-time` tokens are joined into the slot value
`"five minutes"`.

## How it works

`SlotExtractor` trains one supervised HMM per intent. The hidden states are the
BIO tags; the observations are the words. Given a new tokenized utterance, the
Viterbi algorithm finds the most likely tag sequence, which is then parsed back
into a `{slot_name: value}` dict.

## Usage

```python
from ovos_markov_pipeline.slots import SlotExtractor

extractor = SlotExtractor()

extractor.add_slot_data(
    "timer.intent",
    utterances=[
        ["set", "a", "timer", "for", "five", "minutes"],
        ["start", "a", "ten", "minute", "timer"],
    ],
    bio_tags=[
        ["O", "O", "O", "O", "B-time", "I-time"],
        ["O", "O", "B-time", "I-time", "O"],
    ],
)
extractor.train()

slots = extractor.extract("timer.intent", ["set", "a", "timer", "for", "ten", "minutes"])
print(slots)  # {'time': 'ten minutes'}
```

## API

| Method                                          | Purpose                                              |
| ------------------------------------------------ | ---------------------------------------------------- |
| `add_slot_data(intent_name, utterances, bio_tags)` | Register BIO-tagged training data for an intent.   |
| `train()`                                        | Train one HMM per registered intent.                 |
| `extract(intent_name, tokens)`                   | Return a `{slot: value}` dict for a tokenized input. |
| `remove_intent(intent_name)`                     | Drop an intent's slot data and model.                |

## Notes

- `utterances` and `bio_tags` must be aligned: same number of sequences, and
  each tag sequence the same length as its utterance.
- The HMM allocates one hidden state per distinct tag plus one for unknowns.
- `extract` returns an empty dict for an unknown intent, or when Viterbi
  produces a tag sequence whose length does not match the input — extraction
  never raises on malformed input.
- Tokenize utterances the same way at training and extraction time so the
  observation vocabulary lines up.
