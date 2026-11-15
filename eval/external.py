"""Score the detectors on an outside dataset that was never used for training or tuning.

    python -m eval.external_eval eval/external/deepset.csv
    python -m eval.external_eval eval/external/gandalf.csv --attack-only
    python -m eval.external_eval eval/external/notinject.csv --benign-only
    python -m eval.external_eval my.csv --text-col prompt --label-col is_attack --show 10

Rules: the threshold is chosen on dev, so do not change it after seeing these results,
and do not write patterns from the examples that --show prints. Use them to understand, not to tune.
"""
import argparse
import csv

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from app.detectors.ml_layer import MODEL_PATH, build_pipeline
from app.detectors.regex_layer import regex_detect
from eval.run_eval import DATA_PATH, load_rows, split_rows

CACHE = MODEL_PATH.parent / "dev_only_model.joblib"  # models/ is git-ignored

ATTACK_LABELS = {"1", "1.0", "true", "attack", "injection", "prompt_injection", "jailbreak", "malicious", "unsafe", "harmful"}
BENIGN_LABELS = {"0", "0.0", "false", "benign", "safe", "clean", "normal", "legitimate"}


def parse_label(value):
    v = str(value).strip().lower()
    if v in ATTACK_LABELS:
        return 1
    if v in BENIGN_LABELS:
        return 0
    raise SystemExit(f"Unrecognised label {value!r}. Add it to ATTACK_LABELS or BENIGN_LABELS at the top of this file.")


def load_external(path, text_col, label_col, attack_only, benign_only):
    with open(path, encoding="utf-8-sig", newline="") as f:  # utf-8-sig also copes with a BOM
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)
    if text_col not in columns:
        raise SystemExit(f"Column {text_col!r} not found. Columns in the file: {columns}")
    if not (attack_only or benign_only) and label_col not in columns:
        raise SystemExit(f"Column {label_col!r} not found. Columns: {columns}. Use --label-col, --attack-only or --benign-only.")

    texts, labels = [], []
    for r in rows:
        text = (r[text_col] or "").strip()
        if not text:
            continue
        texts.append(text)
        labels.append(1 if attack_only else 0 if benign_only else parse_label(r[label_col]))
    return texts, np.array(labels)


def get_dev_model(texts, y, refit):
    """Model trained on dev only. Cached so scoring several datasets is fast.
    Use --refit after you change the pipeline in ml_layer.py or the training data."""
    if CACHE.exists() and not refit:
        saved = joblib.load(CACHE)
        if saved["n"] == len(texts):
            return saved["model"]
    print("Training on dev only (this takes a moment)...")
    model = build_pipeline().fit(texts, y)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"n": len(texts), "model": model}, CACHE)
    return model


def report(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if tp + fn == 0:  # only benign rows: precision and recall mean nothing, FPR is the number
        print(f"{name:<26} FPR={fp / (fp + tn):.3f}  ({fp} of {fp + tn} benign prompts blocked)")
    elif fp + tn == 0:  # only attacks: recall is the number
        print(f"{name:<26} Recall={tp / (tp + fn):.3f}  ({tp} of {tp + fn} attacks caught)")
    else:
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"{name:<26} P={p:.3f}  R={r:.3f}  F1={f1:.3f}  FPR={fp / (fp + tn):.3f}  (FP={fp}, FN={fn})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="CSV file with the external prompts")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--attack-only", action="store_true", help="every row is an attack (no label column needed)")
    parser.add_argument("--benign-only", action="store_true", help="every row is benign (no label column needed)")
    parser.add_argument("--threshold", type=float, default=0.86, help="ML threshold chosen on dev")
    parser.add_argument("--show", type=int, default=0, help="print this many errors per group (for understanding only)")
    parser.add_argument("--refit", action="store_true", help="retrain the dev-only model")
    args = parser.parse_args()

    # Training data = dev split only. The test split stays untouched.
    dev = split_rows(load_rows(DATA_PATH), "dev")
    dev_texts = [r["text"] for r in dev]
    dev_y = np.array([r["label"] for r in dev])
    dev_seen = {t.lower().strip() for t in dev_texts}

    texts, y = load_external(args.path, args.text_col, args.label_col, args.attack_only, args.benign_only)
    loaded = len(texts)

    # Remove exact repeats and anything the model trained on.
    keep, seen = [], set()
    for i, t in enumerate(texts):
        key = t.lower().strip()
        if key in dev_seen or key in seen:
            continue
        seen.add(key)
        keep.append(i)
    texts = [texts[i] for i in keep]
    y = y[keep]
    if not texts:
        raise SystemExit("No rows left after removing duplicates and overlap with the training data.")

    model = get_dev_model(dev_texts, dev_y, args.refit)
    proba = model.predict_proba(texts)[:, 1]
    regex_pred = np.array([int(regex_detect(t)) for t in texts])
    ml_pred = (proba >= args.threshold).astype(int)
    combined = np.maximum(regex_pred, ml_pred)

    print(f"File: {args.path}")
    print(f"Rows loaded: {loaded} | removed as duplicates/overlap: {loaded - len(texts)} | scored: {len(texts)}")
    print(f"Labels: attacks={int((y == 1).sum())}, benign={int((y == 0).sum())}\n")
    report("Regex only", y, regex_pred)
    report("ML only @0.50", y, (proba >= 0.5).astype(int))
    report(f"ML only @{args.threshold}", y, ml_pred)
    report(f"Regex OR ML @{args.threshold}", y, combined)

    if args.show:
        missed = [i for i in np.where((y == 1) & (combined == 0))[0]]
        false_alarms = [i for i in np.where((y == 0) & (combined == 1))[0]]
        for title, idx in (("ATTACKS MISSED", missed), ("BENIGN BLOCKED", false_alarms)):
            print(f"\n--- {title} ({len(idx)} total, showing {min(args.show, len(idx))}) ---")
            for i in idx[: args.show]:
                print(f"[{proba[i]:.2f}] {texts[i][:150]}".replace("\n", " "))


if __name__ == "__main__":
    main()