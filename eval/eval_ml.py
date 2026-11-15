"""Evaluate the TF-IDF layer honestly and compare it with regex.

Day-to-day (dev data only, 5-fold cross-validation):
    python -m eval.eval_ml

Once, when you are finished tuning (trains on all dev, scores the untouched test split, saves the model):
    python -m eval.eval_ml --final
"""
import argparse

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from app.detectors.ml_layer import MODEL_PATH, build_pipeline, save_model
from app.detectors.regex_layer import regex_detect
from eval.run_eval import DATA_PATH, load_rows, split_rows


def report(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    print(f"{name:<30} P={p:.3f}  R={r:.3f}  F1={f1:.3f}  FPR={fpr:.3f}  (FP={fp}, FN={fn})")


def pick_threshold(y_true, proba, max_fpr):
    """Lowest threshold (= highest recall) whose false positive rate stays within budget."""
    negatives = (y_true == 0).sum()
    for t in np.arange(0.2, 1.0, 0.01):
        fp = ((proba >= t) & (y_true == 0)).sum()
        if fp / negatives <= max_fpr:
            return round(float(t), 2)
    return 0.99


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="train on all dev, score test, save model")
    parser.add_argument("--max-fpr", type=float, default=0.005, help="false positive budget for the threshold")
    args = parser.parse_args()

    all_rows = load_rows(DATA_PATH)
    dev = split_rows(all_rows, "dev")
    test = split_rows(all_rows, "test")
    texts = [r["text"] for r in dev]
    y = np.array([r["label"] for r in dev])

    # How repetitive is the data? Exact repeats make scores look better than they are.
    dev_set = {t.lower().strip() for t in texts}
    test_texts = [r["text"].lower().strip() for r in test]
    print(f"Duplicate prompts inside dev: {len(texts) - len(dev_set)}")
    print(f"Test prompts that also appear in dev: {sum(t in dev_set for t in test_texts)} of {len(test_texts)}\n")

    # Out-of-fold probabilities: every dev prompt is scored by a model that never saw it.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("Running 5-fold cross-validation (this can take a minute or two)...")
    proba = cross_val_predict(build_pipeline(), texts, y, cv=cv, method="predict_proba")[:, 1]

    regex_pred = np.array([int(regex_detect(t)) for t in texts])
    threshold = pick_threshold(y, proba, args.max_fpr)

    print(f"\nDEV results (out-of-fold). Chosen ML threshold: {threshold} (FPR budget {args.max_fpr})")
    report("Regex only", y, regex_pred)
    report("ML only @0.50", y, (proba >= 0.5).astype(int))
    report(f"ML only @{threshold}", y, (proba >= threshold).astype(int))
    report(f"Regex OR ML @{threshold}", y, np.maximum(regex_pred, (proba >= threshold).astype(int)))

    missed_by_regex = (y == 1) & (regex_pred == 0)
    caught = ((proba >= threshold) & missed_by_regex).sum()
    print(f"\nAttacks regex missed: {missed_by_regex.sum()} | ML catches {caught} of them")

    print("\nThreshold table (ML only):")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        report(f"  threshold {t}", y, (proba >= t).astype(int))

    if args.final:
        print("\nFINAL: training on all dev and scoring the untouched test split...")
        model = build_pipeline().fit(texts, y)
        t_texts = [r["text"] for r in test]
        t_y = np.array([r["label"] for r in test])
        t_proba = model.predict_proba(t_texts)[:, 1]
        t_regex = np.array([int(regex_detect(t)) for t in t_texts])
        t_ml = (t_proba >= threshold).astype(int)
        print(f"\nTEST results (threshold {threshold} chosen on dev):")
        report("Regex only", t_y, t_regex)
        report(f"ML only @{threshold}", t_y, t_ml)
        report(f"Regex OR ML @{threshold}", t_y, np.maximum(t_regex, t_ml))
        save_model(model, threshold)
        print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()