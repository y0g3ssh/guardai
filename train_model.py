"""Train the ML layer on the dev split and save it where the API expects it.

    python train_model.py
    python train_model.py --threshold 0.8

Trained on dev only, so the untouched test split stays valid for your final numbers.
"""
import argparse

from app.detectors.ml_layer import MODEL_PATH, build_pipeline, save_model
from eval.run_eval import DATA_PATH, load_rows, split_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.86, help="block when the ML score is at least this")
    args = parser.parse_args()

    dev = split_rows(load_rows(DATA_PATH), "dev")
    texts = [r["text"] for r in dev]
    labels = [r["label"] for r in dev]

    print(f"Training on {len(texts)} dev prompts...")
    model = build_pipeline().fit(texts, labels)
    save_model(model, args.threshold)
    print(f"Saved to {MODEL_PATH} (threshold {args.threshold})")


if __name__ == "__main__":
    main()