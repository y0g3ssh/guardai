"""GuardAI eval harness.
"""
import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from app.detectors.regex_layer import regex_detect as detect

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

DATA_PATH = Path(__file__).parent / "train.csv"  # change if you renamed it


def load_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return [{"text": r["text"], "label": int(r["label"])} for r in csv.DictReader(f)]


def split_rows(rows, split, seed=42, dev_frac=0.8):
    """Fixed 80/20 dev/test split. Build detectors against 'dev'. Report final numbers on 'test'."""
    if split == "all":
        return rows
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * dev_frac)
    return shuffled[:cut] if split == "dev" else shuffled[cut:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    args = parser.parse_args()

    rows = split_rows(load_rows(DATA_PATH), args.split)
    y_true = [r["label"] for r in rows]
    y_pred = [int(detect(r["text"])) for r in rows]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print(f"Split: {args.split} | rows: {len(rows)} | labels: {dict(Counter(y_true))}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1:        {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"False positive rate: {fpr:.3f}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")


if __name__ == "__main__":
    main()