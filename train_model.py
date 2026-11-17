"""Train the ML layer on the dev split and save it where the API expects it.

    python train_model.py
    python train_model.py --threshold 0.86 --segment-threshold 0.95 --augment-benign-segments

Use the numbers that eval_segments.py recommends. Trained on dev only, so the untouched
test split stays valid for your final numbers.
"""
import argparse

from app.detectors.ml_layer import MODEL_PATH, augment_with_benign_segments, build_pipeline, save_model
from eval.run_eval import DATA_PATH, load_rows, split_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.86, help="block when the whole-prompt ML score is at least this")
    parser.add_argument("--segment-threshold", type=float, default=None,
                        help="block when any single segment scores at least this (omit to switch segment scoring off)")
    parser.add_argument("--augment-benign-segments", action="store_true",
                        help="also train on pieces of benign prompts (use the same setting you evaluated)")
    args = parser.parse_args()

    dev = split_rows(load_rows(DATA_PATH), "dev")
    texts = [r["text"] for r in dev]
    labels = [r["label"] for r in dev]
    if args.augment_benign_segments:
        texts, labels = augment_with_benign_segments(texts, labels)

    print(f"Training on {len(texts)} prompts/segments (from {len(dev)} dev prompts)...")
    model = build_pipeline().fit(texts, labels)
    save_model(model, args.threshold, segment_threshold=args.segment_threshold)
    print(f"Saved to {MODEL_PATH} (threshold {args.threshold}, segment threshold {args.segment_threshold})")


if __name__ == "__main__":
    main()
