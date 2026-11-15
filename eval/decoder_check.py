"""How often does the decoder fire on the dev data, and does it help or hurt?

Uses regex only, so no model training is involved and there is no leakage.

    python -m eval.decoder_check
"""
import random
from collections import Counter

from app.detectors.decoder import decode_variants
from app.detectors.regex_layer import regex_detect
from eval.run_eval import DATA_PATH, load_rows, split_rows


def group(method: str) -> str:
    return "caesar" if method.startswith("caesar") else method


def main():
    rows = split_rows(load_rows(DATA_PATH), "dev")
    fired = {1: Counter(), 0: Counter()}  # prompts where each decoder produced a reading
    regex_gain = {1: [], 0: []}  # prompts regex misses as written but catches after decoding
    decoded_examples = []

    for r in rows:
        label, text = r["label"], r["text"]
        variants = decode_variants(text)
        if not variants:
            continue
        fired[label]["any decoder"] += 1
        for method in {group(v.method) for v in variants}:
            fired[label][method] += 1
        decoded_examples.append((label, variants[0].method, text, variants[0].text))
        if not regex_detect(text) and any(regex_detect(v.text) for v in variants):
            regex_gain[label].append((variants[0].method, text, variants[0].text))

    attacks = sum(r["label"] == 1 for r in rows)
    benign = len(rows) - attacks
    print(f"Dev prompts: {len(rows)} (attacks {attacks}, benign {benign})\n")
    print(f"{'decoder':<14}{'attacks':>9}{'benign':>9}")
    for name in ("any decoder", "base64", "url", "leetspeak", "caesar"):
        print(f"{name:<14}{fired[1][name]:>9}{fired[0][name]:>9}")

    print(f"\nRegex catches after decoding, which it missed as written:")
    print(f"  attacks newly caught: {len(regex_gain[1])}")
    print(f"  benign newly blocked (false alarms): {len(regex_gain[0])}")

    rng = random.Random(1)
    for title, items in (("ATTACKS decoded (sample)", [e for e in decoded_examples if e[0] == 1]),
                         ("BENIGN prompts the decoder touched (sample)", [e for e in decoded_examples if e[0] == 0])):
        print(f"\n--- {title} ---")
        for label, method, text, decoded in rng.sample(items, min(8, len(items))):
            print(f"[{method}] {text[:70]!r}\n      -> {decoded[:70]!r}")

    if regex_gain[0]:
        print("\n--- FALSE ALARMS caused by decoding (fix these first) ---")
        for method, text, decoded in regex_gain[0][:10]:
            print(f"[{method}] {text[:80]!r}\n      -> {decoded[:80]!r}")


if __name__ == "__main__":
    main()