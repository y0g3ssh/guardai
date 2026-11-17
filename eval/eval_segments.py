"""Does scoring each sentence separately help? Honest out-of-fold evaluation on dev.

    python -m eval.eval_segments
    python -m eval.eval_segments --augment      # also train on pieces of benign prompts
    python -m eval.eval_segments --max-fpr 0.005 --refresh

Prints the best thresholds with and without segment scoring at the same false positive budget,
plus examples of what segment scoring newly catches and newly blocks. Use the printed
train_model.py command to save the model you evaluated.
"""
import argparse
import json
import random

import numpy as np
from sklearn.model_selection import StratifiedKFold

from app.detectors.ml_layer import MODEL_PATH, augment_with_benign_segments, build_pipeline, split_segments
from eval.eval_ml import report
from eval.run_eval import DATA_PATH, load_rows, split_rows


def oof_scores(texts, y, augment):
    """For every dev prompt: whole-prompt score, best segment score and the segment itself,
    all from a model that never saw that prompt (same 5 folds as eval_ml.py)."""
    n = len(texts)
    whole, seg = np.zeros(n), np.zeros(n)
    seg_text = [None] * n
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for k, (train_idx, val_idx) in enumerate(folds.split(texts, y), 1):
        print(f"  fold {k}/5...")
        train_texts = [texts[i] for i in train_idx]
        train_y = [int(y[i]) for i in train_idx]
        if augment:
            train_texts, train_y = augment_with_benign_segments(train_texts, train_y)
        model = build_pipeline().fit(train_texts, train_y)

        whole[val_idx] = model.predict_proba([texts[i] for i in val_idx])[:, 1]
        pieces, owners = [], []
        for i in val_idx:
            for s in split_segments(texts[i]):
                pieces.append(s)
                owners.append(i)
        if pieces:
            probs = model.predict_proba(pieces)[:, 1]
            for owner, piece, p in zip(owners, pieces, probs):
                if p > seg[owner]:
                    seg[owner], seg_text[owner] = p, piece
    return whole, seg, seg_text


def get_scores(texts, y, augment, refresh):
    cache = MODEL_PATH.parent / f"oof_segments_{'aug' if augment else 'plain'}.json"
    if cache.exists() and not refresh:
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved["n"] == len(texts):
            return np.array(saved["whole"]), np.array(saved["seg"]), saved["seg_text"]
    print("Running 5-fold cross-validation with segment scoring (a few minutes)...")
    whole, seg, seg_text = oof_scores(texts, y, augment)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"n": len(texts), "whole": whole.tolist(), "seg": seg.tolist(), "seg_text": seg_text}), encoding="utf-8")
    return whole, seg, seg_text


def best_thresholds(y, whole, seg, max_fpr):
    """Highest recall within the false positive budget: whole-prompt only, and whole + segments."""
    pos, neg = y == 1, y == 0
    grid = np.round(np.arange(0.5, 1.0, 0.01), 2)
    best = {"whole": None, "both": None}

    def consider(key, blocked, thresholds):
        fpr = (blocked & neg).sum() / neg.sum()
        if fpr > max_fpr:
            return
        rank = ((blocked & pos).sum() / pos.sum(), -fpr)
        if best[key] is None or rank > best[key][0]:
            best[key] = (rank, thresholds)

    for t1 in grid:
        w = whole >= t1
        consider("whole", w, (float(t1), None))
        for t2 in grid:
            consider("both", w | (seg >= t2), (float(t1), float(t2)))
    return best["whole"][1] if best["whole"] else None, best["both"][1] if best["both"] else None


def show(title, idx, texts, whole, seg, seg_text, limit=10):
    print(f"\n--- {title} ({len(idx)} total, showing {min(limit, len(idx))}) ---")
    for i in random.Random(1).sample(list(idx), min(limit, len(idx))):
        print(f"[whole {whole[i]:.2f} | segment {seg[i]:.2f}] {texts[i][:90]!r}")
        print(f"      triggering segment: {str(seg_text[i])[:90]!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", action="store_true", help="also train on pieces of benign prompts")
    parser.add_argument("--max-fpr", type=float, default=0.005)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    dev = split_rows(load_rows(DATA_PATH), "dev")
    texts = [r["text"] for r in dev]
    y = np.array([r["label"] for r in dev])
    with_segments = sum(1 for t in texts if split_segments(t))
    print(f"Dev prompts: {len(texts)} | prompts with 2+ segments: {with_segments} | augment: {args.augment}")

    whole, seg, seg_text = get_scores(texts, y, args.augment, args.refresh)
    t_whole, t_both = best_thresholds(y, whole, seg, args.max_fpr)
    if t_whole is None or t_both is None:
        raise SystemExit("No thresholds fit inside the false positive budget. Try a larger --max-fpr.")

    block_whole = whole >= t_whole[0]
    block_both = (whole >= t_both[0]) | (seg >= t_both[1])
    print(f"\nBest within FPR budget {args.max_fpr} (out-of-fold, dev only):")
    report(f"Whole prompt only @{t_whole[0]}", y, block_whole.astype(int))
    report(f"+ segments @{t_both[0]} / {t_both[1]}", y, block_both.astype(int))

    gained = np.where(block_both & ~block_whole & (y == 1))[0]
    added_fp = np.where(block_both & ~block_whole & (y == 0))[0]
    lost = np.where(block_whole & ~block_both & (y == 1))[0]
    print(f"\nCompared with whole-prompt only: segments newly catch {len(gained)} attacks and newly block {len(added_fp)} benign prompts; "
          f"{len(lost)} attacks caught before are no longer caught (the whole-prompt threshold can differ between the two settings).")
    show("ATTACKS newly caught by segment scoring", gained, texts, whole, seg, seg_text)
    show("BENIGN newly blocked by segment scoring", added_fp, texts, whole, seg, seg_text)

    aug = " --augment-benign-segments" if args.augment else ""
    print("\nTo save this model for the API:")
    print(f"  python train_model.py --threshold {t_both[0]} --segment-threshold {t_both[1]}{aug}")
    print("If the gain is small or the new false alarms look bad, skip --segment-threshold and keep whole-prompt scoring only.")


if __name__ == "__main__":
    main()