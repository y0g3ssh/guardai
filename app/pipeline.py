"""Runs the detector layers in order (cheapest first) and returns a single decision.

    regex  ->  ML (whole prompt, then each segment, then decoded readings)  ->  Gemini judge (only if unsure)
"""
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

from app.detectors.decoder import decode_variants
from app.detectors.ml_layer import MODEL_PATH, MLDetector
from app.detectors.regex_layer import triggered
from app.llm_layer import GeminiJudge, LLMUnavailable

log = logging.getLogger("guardai")


@dataclass
class Decision:
    action: str  # "allow" or "block"
    layer: Optional[str]  # layer that made the final call: "regex", "ml", "llm", "llm_fallback", or None for a plain allow
    ml_score: Optional[float] = None  # None when regex already blocked or no model is loaded
    rules: List[str] = field(default_factory=list)  # regex rules that fired
    latency_ms: float = 0.0
    decoded_from: Optional[str] = None  # "base64", "url", "leetspeak" or "caesar_shift_<n>" when decoded text caused the block
    decoded_text: Optional[str] = None  # what the decoder read (first 200 characters)
    segment: Optional[str] = None  # the sentence/note that triggered a segment-level ML block
    llm: Optional[dict] = None  # the Gemini verdict, or {"error": ...} when it could not answer


@dataclass
class LLMPolicy:
    low: float = 0.3  # ML risk below this = confidently safe, no LLM call
    min_confidence: float = 0.6  # the judge must be at least this sure for its verdict to change the outcome
    review_blocks: bool = False  # also let the judge double-check ML blocks (it can only cancel ML blocks, never regex blocks)
    fail_open: bool = True  # if the judge is unavailable for an uncertain prompt: allow (True) or block (False)

    @classmethod
    def from_env(cls) -> "LLMPolicy":
        def flag(name: str, default: bool) -> bool:
            return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")

        return cls(
            low=float(os.getenv("GUARD_LLM_LOW", "0.3")),
            min_confidence=float(os.getenv("GUARD_LLM_MIN_CONF", "0.6")),
            review_blocks=flag("GUARD_LLM_REVIEW_BLOCKS", False),
            fail_open=flag("GUARD_LLM_FAIL_OPEN", True),
        )


class GuardPipeline:
    def __init__(self, ml: Optional[MLDetector] = None, judge: Optional[GeminiJudge] = None, policy: Optional[LLMPolicy] = None):
        self.ml = ml
        self.judge = judge
        self.policy = policy or LLMPolicy()

    @classmethod
    def load(cls) -> "GuardPipeline":
        try:
            ml = MLDetector.load()
        except FileNotFoundError:
            log.warning("No trained model at %s, running regex-only. Run: python train_model.py", MODEL_PATH)
            ml = None
        judge = GeminiJudge.from_env()
        if judge is None:
            log.info("LLM layer off (set GEMINI_API_KEY to enable it)")
        else:
            log.info("LLM layer on (model %s)", judge.model)
        return cls(ml, judge, LLMPolicy.from_env())

    # ---- main entry -----------------------------------------------------
    def check(self, text: str) -> Decision:
        start = time.perf_counter()
        variants = decode_variants(text)  # usually empty; only encoded-looking prompts produce readings

        # Layer 1: regex on the prompt as written, then on decoded readings.
        rules = triggered(text)
        if rules:
            return self._finish(Decision("block", "regex", rules=rules), start)
        for v in variants:
            rules = triggered(v.text)
            if rules:
                return self._finish(Decision("block", "regex", rules=rules, decoded_from=v.method, decoded_text=v.text[:200]), start)

        # Layer 2: ML.
        score = None
        if self.ml is not None:
            score, block, risk = self._run_ml(text, variants)
            if block is not None:
                decision = Decision(
                    "block", "ml", block["score"],
                    decoded_from=block.get("decoded_from"), decoded_text=block.get("decoded_text"), segment=block.get("segment"),
                )
                return self._review_block(text, decision, start)
            # Layer 3: not blocked, but not clearly safe either -> ask the judge.
            if self.judge is not None and risk >= self.policy.low:
                return self._ask_judge(text, score, start)

        return self._finish(Decision("allow", None, score), start)

    # ---- layer 2 --------------------------------------------------------
    def _run_ml(self, text, variants):
        """Returns (whole-prompt score, block details or None, highest risk seen anywhere)."""
        ml = self.ml
        score = ml.score(text)
        risk = score
        if score >= ml.threshold:
            return score, {"score": score}, risk

        segment_threshold = getattr(ml, "segment_threshold", None)
        if segment_threshold is not None:
            seg_score, seg = ml.best_segment(text)
            risk = max(risk, seg_score)
            if seg_score >= segment_threshold:
                return score, {"score": seg_score, "segment": (seg or "")[:200]}, risk

        for v in variants:
            s = ml.score(v.text)
            risk = max(risk, s)
            if s >= ml.threshold:
                return score, {"score": s, "decoded_from": v.method, "decoded_text": v.text[:200]}, risk
        return score, None, risk

    # ---- layer 3 --------------------------------------------------------
    def _review_block(self, text, decision, start):
        """Optional second opinion on an ML block. A confident 'benign' verdict cancels the block."""
        if self.judge is None or not self.policy.review_blocks:
            return self._finish(decision, start)
        try:
            verdict = self.judge.judge(text)
        except LLMUnavailable as e:
            decision.llm = {"error": str(e)}  # keep the block
            return self._finish(decision, start)
        decision.llm = self._info(verdict)
        if verdict.label == "benign" and verdict.confidence >= self.policy.min_confidence:
            return self._finish(
                Decision("allow", "llm", decision.ml_score, decoded_from=decision.decoded_from,
                         decoded_text=decision.decoded_text, segment=decision.segment, llm=decision.llm),
                start,
            )
        return self._finish(decision, start)

    def _ask_judge(self, text, score, start):
        try:
            verdict = self.judge.judge(text)
        except LLMUnavailable as e:
            info = {"error": str(e)}
            if self.policy.fail_open:
                return self._finish(Decision("allow", None, score, llm=info), start)
            return self._finish(Decision("block", "llm_fallback", score, llm=info), start)
        info = self._info(verdict)
        if verdict.label == "attack" and verdict.confidence >= self.policy.min_confidence:
            return self._finish(Decision("block", "llm", score, llm=info), start)
        return self._finish(Decision("allow", "llm", score, llm=info), start)

    # ---- helpers --------------------------------------------------------
    @staticmethod
    def _info(verdict) -> dict:
        return {"label": verdict.label, "confidence": verdict.confidence, "reason": verdict.reason, "cached": verdict.cached}

    @staticmethod
    def _finish(decision: Decision, start: float) -> Decision:
        decision.latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return decision