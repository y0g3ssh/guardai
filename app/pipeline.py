"""Runs the detector layers in order (cheapest first) and returns a single decision."""
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from app.detectors.ml_layer import MODEL_PATH, MLDetector
from app.detectors.regex_layer import triggered

log = logging.getLogger("guardai")


@dataclass
class Decision:
    action: str  # "allow" or "block"
    layer: Optional[str]  # which layer decided: "regex", "ml", or None when allowed
    ml_score: Optional[float] = None  # None when regex already blocked or no model is loaded
    rules: List[str] = field(default_factory=list)  # regex rules that fired
    latency_ms: float = 0.0


class GuardPipeline:
    def __init__(self, ml: Optional[MLDetector] = None):
        self.ml = ml

    @classmethod
    def load(cls) -> "GuardPipeline":
        try:
            ml = MLDetector.load()
        except FileNotFoundError:
            log.warning("No trained model at %s, running regex-only. Run: python train_model.py", MODEL_PATH)
            ml = None
        return cls(ml)

    def check(self, text: str) -> Decision:
        start = time.perf_counter()

        # Layer 1: regex. Instant, and it explains itself through the rule names.
        rules = triggered(text)
        if rules:
            return Decision("block", "regex", None, rules, self._ms(start))

        # Layer 2: TF-IDF classifier.
        score = None
        if self.ml is not None:
            score = self.ml.score(text)
            if score >= self.ml.threshold:
                return Decision("block", "ml", score, [], self._ms(start))

        # Layer 3 (Gemini) will go here, to double-check blocks and settle the uncertain band.
        return Decision("allow", None, score, [], self._ms(start))

    @staticmethod
    def _ms(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 2)