# FAQ

## What is this plugin?
An OVOS pipeline plugin that classifies intents by training one Markov chain per intent and selecting the model with lowest perplexity on the input utterance.

## How does confidence scoring work?
`confidence = 1 / (1 + log(perplexity))`. Lower perplexity means the utterance is more likely under that intent's model, producing higher confidence.

## What order should I use?
Order 1 works best with small training sets (5-20 examples). Order 2 needs 20+ examples per intent.

## Does stemming help?
Yes, set `"stem": true` in config. Uses snowball stemmer to normalize word forms (running→run, dogs→dog). Significant improvement for inflected languages.

## What is the character-level fallback?
When `"char_fallback": true`, the engine trains char-level models alongside word-level. If the top-2 word-level scores are within 0.05 of each other, it blends 60% word + 40% char scores to break the tie.

## What is online learning?
When `"online_learning": true`, high-confidence matches are fed back into the intent model. The matched utterance is added to the intent's training data and the model is incrementally updated — no full retrain needed.

## Does it support entities/slots?
Not yet. Entity extraction via HMM (BIO tagging) is planned.

## How does it compare to Padatious?
Padatious uses template matching with neural scoring. Markov is simpler, trains instantly, no native code deps. Padatious has better accuracy with pattern templates (``{entity}`` syntax).

## How does it compare to Model2Vec?
Model2Vec uses semantic embeddings (300MB+ model). Markov is kilobytes, trains in milliseconds, but doesn't understand semantic similarity — only word co-occurrence patterns.
