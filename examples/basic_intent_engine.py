#!/usr/bin/env python3
"""Example: Using MarkovIntentEngine standalone (without OVOS).

Demonstrates intent classification, confidence scoring, and slot
extraction without requiring the full OVOS framework.
"""

from ovos_markov_pipeline import MarkovIntentEngine, _ppx_to_confidence
from ovos_markov_pipeline.slots import SlotExtractor
from ovos_markov_pipeline.calibration import evaluate


def main() -> None:
    # 1. Create and train intent engine
    engine = MarkovIntentEngine(order=1, kneser_ney=True, backoff=False)

    engine.add_intent("weather:get_weather", [
        "what is the weather", "what is the weather like",
        "how is the weather today", "tell me the weather",
        "what is the forecast", "is it going to rain",
        "will it rain today", "what is the temperature",
        "how hot is it outside", "what is the weather forecast",
    ])
    engine.add_intent("timer:set_timer", [
        "set a timer for five minutes", "set a timer for ten minutes",
        "start a timer", "set a countdown",
        "timer for five minutes", "remind me in ten minutes",
        "start a countdown", "set an alarm for five minutes",
        "set a timer", "start a five minute timer",
    ])
    engine.add_intent("music:play_music", [
        "play some music", "play jazz music",
        "play rock and roll", "put on some music",
        "play my playlist", "play something relaxing",
        "i want to listen to music", "play the radio",
        "play my favorite songs", "start playing music",
    ])

    engine.train()

    # 2. Classify utterances
    print("--- Intent Classification ---")
    test_utterances = [
        "what is the weather today",
        "set a timer for three minutes",
        "play my playlist",
        "how hot is it outside",
        "start a countdown",
        "play some jazz",
        "random gibberish words here",
    ]

    for utt in test_utterances:
        scores = engine.calc_intents(utt)
        if scores:
            intent, conf = scores[0]
            print(f"  '{utt}' → {intent} ({conf:.3f})")
        else:
            print(f"  '{utt}' → (no match)")

    # 3. Evaluate accuracy
    print("\n--- Evaluation ---")
    eval_data = [
        ("what is the forecast today", "weather:get_weather"),
        ("is it going to rain tomorrow", "weather:get_weather"),
        ("set a timer please", "timer:set_timer"),
        ("start a five minute timer", "timer:set_timer"),
        ("play some relaxing music", "music:play_music"),
        ("put on my favorite playlist", "music:play_music"),
    ]
    metrics = evaluate(engine, eval_data)
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  F1: {metrics['f1']:.3f}")

    # 4. Slot extraction with HMM
    print("\n--- Slot Extraction ---")
    ext = SlotExtractor()
    ext.add_slot_data(
        "timer:set_timer",
        utterances=[
            ["set", "a", "timer", "for", "five", "minutes"],
            ["set", "a", "timer", "for", "ten", "minutes"],
            ["timer", "for", "three", "hours"],
            ["set", "a", "countdown", "for", "two", "minutes"],
        ],
        bio_tags=[
            ["O", "O", "O", "O", "B-time", "I-time"],
            ["O", "O", "O", "O", "B-time", "I-time"],
            ["O", "O", "B-time", "I-time"],
            ["O", "O", "O", "O", "B-time", "I-time"],
        ],
    )
    ext.train()

    for tokens in [
        ["set", "a", "timer", "for", "five", "minutes"],
        ["timer", "for", "three", "hours"],
    ]:
        slots = ext.extract("timer:set_timer", tokens)
        print(f"  {' '.join(tokens)} → {slots}")


if __name__ == "__main__":
    main()
