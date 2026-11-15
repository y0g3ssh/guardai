"""Layer 2 (cheap ML): TF-IDF + logistic regression prompt-injection classifier."""
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

# guardai/models/tfidf_lr.joblib  (this file lives in guardai/app/detectors/)
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "tfidf_lr.joblib"


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


def save_model(model: Pipeline, threshold: float, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold}, path)


class MLDetector:
    """Loads the saved model and answers: is this prompt an attack?"""

    def __init__(self, model: Pipeline, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "MLDetector":
        saved = joblib.load(path)
        return cls(saved["model"], saved["threshold"])

    def score(self, text: str) -> float:
        """Probability (0 to 1) that the prompt is an attack."""
        return float(self.model.predict_proba([text])[0][1])

    def __call__(self, text: str) -> bool:
        return self.score(text) >= self.threshold