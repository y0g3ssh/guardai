"""Runs the detector layers in order (cheapest first) and returns a single decision."""
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from app.detectors.decoder import decode_variants
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
    decoded_from: Optional[str] = None  # "base64", "url", "leetspeak" or "caesar_shift_<n>" when the block came from decoded text
    decoded_text: Optional[str] = None  # what the decoder read (first 200 characters)


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
        variants = decode_variants(text)  # usually empty; only encoded-looking prompts produce readings

        # Layer 1: regex on the prompt as written, then on any decoded readings.
        rules = triggered(text)
        if rules:
            return Decision("block", "regex", None, rules, self._ms(start))
        for v in variants:
            rules = triggered(v.text)
            if rules:
                return Decision("block", "regex", None, rules, self._ms(start), v.method, v.text[:200])

        # Layer 2: TF-IDF classifier, again on the original first and then on decoded readings.
        score = None
        if self.ml is not None:
            score = self.ml.score(text)
            if score >= self.ml.threshold:
                return Decision("block", "ml", score, [], self._ms(start))
            for v in variants:
                decoded_score = self.ml.score(v.text)
                if decoded_score >= self.ml.threshold:
                    return Decision("block", "ml", decoded_score, [], self._ms(start), v.method, v.text[:200])

        # Layer 3 (Gemini) will go here, to double-check blocks and settle the uncertain band.
        return Decision("allow", None, score, [], self._ms(start))

    @staticmethod
    def _ms(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 2)