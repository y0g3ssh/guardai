"""Look at what the ML layer gets wrong (dev data only, out-of-fold scores).

    python -m eval.errors_ml                       # threshold 0.86
    python -m eval.errors_ml --threshold 0.7 --n 40
    python -m eval.errors_ml --refresh             # recompute the cross-validation scores

Line format:  [score]R text     ("R" means the regex layer also flags it)
"""
import argparse
import random

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from app.detectors.ml_layer import MODEL_PATH, build_pipeline
from app.detectors.regex_layer import regex_detect
from eval.run_eval import DATA_PATH, load_rows, split_rows

CACHE = MODEL_PATH.parent / "oof_proba.npy"  # lives in models/, which is git-ignored


def get_proba(texts, y, refresh):
    """Out-of-fold attack probabilities, cached so repeat runs are instant.
    Use --refresh after changing the dataset or the pipeline in ml_layer.py."""
    if CACHE.exists() and not refresh:
        cached = np.load(CACHE)
        if len(cached) == len(texts):
            return cached
    print("Running 5-fold cross-validation (a minute or two)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(build_pipeline(), texts, y, cv=cv, method="predict_proba")[:, 1]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE, proba)
    return proba


def show(title, idx, texts, proba, regex_pred, limit):
    print(f"\n--- {title} ({len(idx)} total, showing {min(limit, len(idx))}) ---")
    for i in idx[:limit]:
        flag = "R" if regex_pred[i] else " "
        text = texts[i][:150].replace("\n", " ")
        print(f"[{proba[i]:.2f}]{flag} {text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.86, help="ML block threshold")
    parser.add_argument("--low", type=float, default=0.3, help="below this the ML layer is 'confident it is safe'")
    parser.add_argument("--n", type=int, default=25, help="how many examples to print per group")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    dev = split_rows(load_rows(DATA_PATH), "dev")
    texts = [r["text"] for r in dev]
    y = np.array([r["label"] for r in dev])

    proba = get_proba(texts, y, args.refresh)
    regex_pred = np.array([int(regex_detect(t)) for t in texts])
    blocked = proba >= args.threshold

    # 1) Benign prompts the ML layer would block, most confident first.
    false_alarms = np.where((y == 0) & blocked)[0]
    false_alarms = false_alarms[np.argsort(-proba[false_alarms])]

    # 2) Attacks it lets through: "confident misses" are the worrying ones,
    #    "near misses" are in the uncertain band an LLM layer could settle.
    confident_misses = [int(i) for i in np.where((y == 1) & (proba < args.low))[0]]
    near_misses = [int(i) for i in np.where((y == 1) & (proba >= args.low) & ~blocked)[0]]
    random.Random(1).shuffle(confident_misses)
    random.Random(1).shuffle(near_misses)

    # Summary numbers
    band = (proba >= args.low) & (proba < args.threshold)
    print(f"Threshold {args.threshold} | uncertain band {args.low} to {args.threshold}")
    print(f"Prompts in the uncertain band: {band.sum()} of {len(y)} ({band.mean():.1%})")
    print(f"  attacks in band: {(band & (y == 1)).sum()} | benign in band: {(band & (y == 0)).sum()}")
    missed = (y == 1) & ~blocked
    print(f"Attacks the ML layer misses: {missed.sum()} | regex flags {(missed & (regex_pred == 1)).sum()} of them")

    show("FALSE ALARMS: benign prompts blocked by ML", false_alarms, texts, proba, regex_pred, args.n)
    show(f"CONFIDENT MISSES: attacks scored below {args.low}", confident_misses, texts, proba, regex_pred, args.n)
    show("NEAR MISSES: attacks in the uncertain band", near_misses, texts, proba, regex_pred, args.n)


if __name__ == "__main__":
    main()