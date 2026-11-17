"""How good is the Gemini judge on the prompts the ML layer is unsure about?

    python -m eval.errors_ml                # once: creates the cached cross-validation scores this script reads
    python -m eval.eval_llm --n 30          # 30 attacks + 30 benign prompts from the uncertain band = about 60 API calls
    python -m eval.eval_llm --n 60 --rpm 10 --low 0.3 --threshold 0.86

Needs GEMINI_API_KEY (see .env.example). Verdicts are cached in models/llm_cache.jsonl, so reruns cost nothing.
The result is an ESTIMATE from a small sample, shown with 95% intervals. Dev data only.
"""
import argparse
import hashlib
import json
import random
import time
from math import sqrt

import numpy as np

from app.detectors.ml_layer import MODEL_PATH
from app.llm_layer import GeminiJudge, LLMUnavailable
from eval.errors_ml import CACHE as OOF_CACHE
from eval.run_eval import DATA_PATH, load_rows, split_rows

VERDICT_CACHE = MODEL_PATH.parent / "llm_cache.jsonl"


def wilson(k, n, z=1.96):
    """95% interval for a proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load_cache():
    cache = {}
    if VERDICT_CACHE.exists():
        for line in VERDICT_CACHE.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cache[row["key"]] = row
    return cache


def judge_sample(judge, items, cache, rpm, stop_after_errors=5):
    """Returns {index: verdict dict}. Real calls are paced to stay under the free-tier request limit."""
    results, consecutive_errors, last_call = {}, 0, 0.0
    for n, (i, text) in enumerate(items, 1):
        key = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()
        if key in cache:
            results[i] = cache[key]
            continue
        wait = 60.0 / rpm - (time.monotonic() - last_call)
        if wait > 0:
            time.sleep(wait)
        last_call = time.monotonic()
        try:
            v = judge.judge(text)
        except LLMUnavailable as e:
            consecutive_errors += 1
            print(f"  [{n}/{len(items)}] unavailable: {e}")
            if consecutive_errors >= stop_after_errors:
                print("  Too many errors in a row, stopping early to protect your quota.")
                break
            continue
        consecutive_errors = 0
        row = {"key": key, "label": v.label, "confidence": v.confidence, "reason": v.reason}
        cache[key] = row
        with VERDICT_CACHE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        results[i] = row
        print(f"  [{n}/{len(items)}] {v.label} ({v.confidence:.2f})")
    return results


def evaluate(judge, texts, y, proba, low, threshold, n, rpm, min_conf, seed=7):
    band = (proba >= low) & (proba < threshold)
    atk_idx = [int(i) for i in np.where(band & (y == 1))[0]]
    ben_idx = [int(i) for i in np.where(band & (y == 0))[0]]
    print(f"Uncertain band {low} to {threshold}: {len(atk_idx)} attacks and {len(ben_idx)} benign prompts "
          f"({(len(atk_idx) + len(ben_idx)) / len(y):.1%} of dev)")
    if not atk_idx or not ben_idx:
        raise SystemExit("The band needs both attacks and benign prompts to estimate anything.")

    rng = random.Random(seed)
    sample_atk = rng.sample(atk_idx, min(n, len(atk_idx)))
    sample_ben = rng.sample(ben_idx, min(n, len(ben_idx)))
    cache = load_cache()
    VERDICT_CACHE.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nAsking the judge about {len(sample_atk)} attacks and {len(sample_ben)} benign prompts...")
    verdicts = judge_sample(judge, [(i, texts[i]) for i in sample_atk + sample_ben], cache, rpm)

    def flagged(row):
        return row["label"] == "attack" and row["confidence"] >= min_conf

    atk_rows = [(i, verdicts[i]) for i in sample_atk if i in verdicts]
    ben_rows = [(i, verdicts[i]) for i in sample_ben if i in verdicts]
    k_atk = sum(flagged(r) for _, r in atk_rows)
    k_ben = sum(flagged(r) for _, r in ben_rows)
    if not atk_rows or not ben_rows:
        raise SystemExit("Not enough verdicts to estimate anything (see the errors above).")

    r, (r_lo, r_hi) = k_atk / len(atk_rows), wilson(k_atk, len(atk_rows))
    f, (f_lo, f_hi) = k_ben / len(ben_rows), wilson(k_ben, len(ben_rows))
    print(f"\nJudge on band ATTACKS: flagged {k_atk} of {len(atk_rows)} ({r:.0%}, 95% interval {r_lo:.0%} to {r_hi:.0%})")
    print(f"Judge on band BENIGN:  flagged {k_ben} of {len(ben_rows)} ({f:.0%}, 95% interval {f_lo:.0%} to {f_hi:.0%})")

    total_atk, total_ben = int((y == 1).sum()), int((y == 0).sum())
    blocked = proba >= threshold
    tp_ml, fp_ml = int((blocked & (y == 1)).sum()), int((blocked & (y == 0)).sum())

    def projected(rate_atk, rate_ben):
        return (tp_ml + rate_atk * len(atk_idx)) / total_atk, (fp_ml + rate_ben * len(ben_idx)) / total_ben

    rec0, fpr0 = tp_ml / total_atk, fp_ml / total_ben
    rec, fpr = projected(r, f)
    rec_lo, fpr_lo = projected(r_lo, f_lo)
    rec_hi, fpr_hi = projected(r_hi, f_hi)
    print(f"\nProjected effect on the whole dev set (ML threshold {threshold}):")
    print(f"  ML only:        recall {rec0:.3f}   FPR {fpr0:.4f}")
    print(f"  ML + judge:     recall {rec:.3f}   FPR {fpr:.4f}")
    print(f"  plausible range: recall {rec_lo:.3f} to {rec_hi:.3f}   FPR {fpr_lo:.4f} to {fpr_hi:.4f}")
    print("This is an estimate from a small sample. Note that some dataset labels count probing questions "
          "('explain how you handle jailbreaks') as attacks, which a careful judge will call benign.")

    print("\n--- ATTACKS the judge let through (sample) ---")
    for i, row in [(i, r_) for i, r_ in atk_rows if not flagged(r_)][:8]:
        print(f"[{row['label']} {row['confidence']:.2f}] {texts[i][:90]!r}\n      judge: {row['reason'][:110]}")
    print("\n--- BENIGN prompts the judge flagged (sample) ---")
    for i, row in [(i, r_) for i, r_ in ben_rows if flagged(r_)][:8]:
        print(f"[{row['label']} {row['confidence']:.2f}] {texts[i][:90]!r}\n      judge: {row['reason'][:110]}")


def main():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="prompts per class to send to the judge")
    parser.add_argument("--rpm", type=int, default=10, help="requests per minute to allow yourself")
    parser.add_argument("--low", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.86)
    parser.add_argument("--min-conf", type=float, default=0.6)
    args = parser.parse_args()

    judge = GeminiJudge.from_env(max_calls_per_minute=10_000)  # pacing is handled here, not by the pipeline's limiter
    if judge is None:
        raise SystemExit("No GEMINI_API_KEY found. Copy .env.example to .env and add your key.")
    if not OOF_CACHE.exists():
        raise SystemExit("Run `python -m eval.errors_ml` first. It creates the cross-validation scores this script reads.")

    dev = split_rows(load_rows(DATA_PATH), "dev")
    texts = [r["text"] for r in dev]
    y = np.array([r["label"] for r in dev])
    proba = np.load(OOF_CACHE)
    if len(proba) != len(texts):
        raise SystemExit("The cached scores do not match the dev set. Run `python -m eval.errors_ml --refresh`.")

    print(f"Model: {judge.model}")
    evaluate(judge, texts, y, proba, args.low, args.threshold, args.n, args.rpm, args.min_conf)


if __name__ == "__main__":
    main()