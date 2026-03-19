# FAQ

## What is this plugin?
An OVOS pipeline plugin that classifies intents by training one Markov chain per intent and selecting the model with lowest perplexity on the input utterance.

## How does confidence scoring work?
`confidence = 1 / (1 + log(perplexity))`. Lower perplexity means the utterance is more likely under that intent's model, producing higher confidence.

## What order should I use?
Order 1 (unigram context) works best with small training sets (5-20 examples). Order 2 needs 20+ examples per intent for reliable discrimination.

## Does it support entities/slots?
Not currently. This plugin identifies *which* intent matches, not *what* entities are present. Use alongside Adapt (for keyword entities) or a slot-filling plugin.

## How does it compare to Padatious?
Padatious uses template matching with neural network scoring. Markov is simpler (no neural network), trains instantly, and works well for narrow domains. Padatious generally has better accuracy with pattern templates.
