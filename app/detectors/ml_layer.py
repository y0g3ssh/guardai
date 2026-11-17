"""Layer 2 (cheap ML): TF-IDF + logistic regression prompt-injection classifier."""
import re
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

# guardai/models/tfidf_lr.joblib  (this file lives in guardai/app/detectors/)
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "tfidf_lr.joblib"

# Segment scoring: an attack hidden inside a long, normal prompt gets diluted when the whole text is scored,
# so we also score each sentence / bracketed note / line on its own.
SEGMENT_SPLIT = re.compile(r"(?<=[.!?])\s+|[\n\r]+|\[|\]|\s*\|\s*")
MIN_SEGMENT_CHARS = 12
MAX_SEGMENTS = 12


def split_segments(text: str) -> List[str]:
    """Sentences, lines and [bracketed notes] of a prompt. Empty when the prompt is a single segment."""
    parts = [p.strip(" \t:;-") for p in SEGMENT_SPLIT.split(text)]
    parts = [p for p in parts if len(p) >= MIN_SEGMENT_CHARS]
    return parts[:MAX_SEGMENTS] if len(parts) > 1 else []


def augment_with_benign_segments(texts, labels):
    """Training helper: pieces of benign prompts are benign too. Teaches the model that short
    fragments are not suspicious just because they are short. (Pieces of attacks are NOT added,
    because we cannot tell which piece carries the attack.)"""
    extra = [seg for t, y in zip(texts, labels) if y == 0 for seg in split_segments(t)]
    return list(texts) + extra, list(labels) + [0] * len(extra)


def build_pipeline() -> Pipeline:
    """Text -> numbers (two TF-IDF views) -> logistic regression -> attack probability."""
    features = FeatureUnion(
        [
            # Whole words and word pairs: catches phrases like "ignore previous".
            ("words", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            # Pieces of 3-5 characters: survives typos like "egnore" or "reveel".
            (
                "chars",
                TfidfVectorizer(
                    lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True
                ),
            ),
        ]
    )
    return Pipeline([("features", features), ("clf", LogisticRegression(max_iter=1000, C=4.0))])


def save_model(model: Pipeline, threshold: float, path: Path = MODEL_PATH, segment_threshold: Optional[float] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "segment_threshold": segment_threshold}, path)


class MLDetector:
    """Loads the saved model and answers: is this prompt an attack?"""

    def __init__(self, model: Pipeline, threshold: float = 0.5, segment_threshold: Optional[float] = None):
        self.model = model
        self.threshold = threshold
        self.segment_threshold = segment_threshold  # None = segment scoring switched off

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "MLDetector":
        saved = joblib.load(path)
        return cls(saved["model"], saved["threshold"], saved.get("segment_threshold"))

    def score(self, text: str) -> float:
        """Probability (0 to 1) that the whole prompt is an attack."""
        return float(self.model.predict_proba([text])[0][1])

    def best_segment(self, text: str) -> Tuple[float, Optional[str]]:
        """Highest attack probability among the prompt's segments, and the segment that produced it."""
        segments = split_segments(text)
        if not segments:
            return 0.0, None
        probs = self.model.predict_proba(segments)[:, 1]
        best = int(probs.argmax())
        return float(probs[best]), segments[best]

    def __call__(self, text: str) -> bool:
        return self.score(text) >= self.threshold