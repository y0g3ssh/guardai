"""Undo common obfuscation tricks (base64, URL encoding, leetspeak, Caesar/ROT shifts)
so the detectors can read what the attacker actually wrote.

decode_variants(text) returns zero or more alternative readings of the prompt.
Each decoded reading is only returned when it looks like real text, so random strings
such as hashes, UUIDs and API keys are ignored.
"""
import base64
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import List
from urllib.parse import unquote


@dataclass
class Variant:
    method: str  # "base64", "url", "leetspeak" or "caesar_shift_<n>"
    text: str


MAX_VARIANTS = 6
B64_CHUNK = re.compile(r"[A-Za-z0-9+/_-]{16,}={0,2}")
URL_ESCAPE = re.compile(r"%(?:4[1-9a-f]|5[0-9a]|6[1-9a-f]|7[0-9a])", re.IGNORECASE)  # escaped letters
LEET_TRIGGER = re.compile(r"[a-z][013457][a-z]", re.IGNORECASE)  # a digit sitting inside a word
LEET_MAP = str.maketrans({"0": "o", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

# Letter frequencies of English text (percent), a to z. Used to spot a Caesar-shifted sentence.
EN_FREQ = [
    8.167, 1.492, 2.782, 4.253, 12.702, 2.228, 2.015, 6.094, 6.966, 0.153, 0.772, 4.025, 2.406,
    6.749, 7.507, 1.929, 0.095, 5.987, 6.327, 9.056, 2.758, 0.978, 2.360, 0.150, 1.974, 0.074,
]


def _readable(s: str) -> bool:
    """Does decoded output look like human text rather than binary noise?"""
    if len(s) < 8:
        return False
    printable = sum(ch in string.printable for ch in s) / len(s)
    texty = sum(ch.isalpha() or ch.isspace() for ch in s) / len(s)
    return printable >= 0.95 and texty >= 0.8


def _base64_variants(text: str) -> List[Variant]:
    out = []
    for m in list(B64_CHUNK.finditer(text))[:5]:
        chunk = m.group(0).rstrip("=").replace("-", "+").replace("_", "/")
        chunk += "=" * (-len(chunk) % 4)
        try:
            decoded = base64.b64decode(chunk, validate=True).decode("utf-8")
        except Exception:
            continue
        if _readable(decoded):
            out.append(Variant("base64", decoded))
    return out


def _url_variants(text: str) -> List[Variant]:
    if len(URL_ESCAPE.findall(text)) < 3:
        return []
    decoded = unquote(text)
    return [Variant("url", decoded)] if decoded != text else []


def _leet_variants(text: str) -> List[Variant]:
    text = B64_CHUNK.sub(" ", text)  # digits inside a base64 blob are not leetspeak
    if not LEET_TRIGGER.search(text):
        return []
    base = text.translate(LEET_MAP)
    out = [Variant("leetspeak", base.replace("1", "i"))]  # "1gnore" -> "ignore"
    if "1" in base:
        out.append(Variant("leetspeak", base.replace("1", "l")))  # "ru1es" -> "rules"
    return out


def _caesar_variants(text: str) -> List[Variant]:
    """If the text's letter frequencies fit English far better after a shift, return the shifted text."""
    text = B64_CHUNK.sub(" ", text)  # base64 blobs are handled elsewhere and would only confuse this check
    if not 20 <= len(text) <= 2000:
        return []
    letters = [c for c in text.lower() if "a" <= c <= "z"]
    n = len(letters)
    if n < 20 or n / len(text) < 0.6:
        return []
    counts = Counter(ord(c) - 97 for c in letters)

    def chi2(shift: int) -> float:  # per-letter distance from English; lower means more English-like
        total = 0.0
        for j in range(26):
            expected = n * EN_FREQ[j] / 100
            total += (counts[(j + shift) % 26] - expected) ** 2 / expected
        return total / n

    best = min(range(1, 26), key=chi2)
    as_written, shifted = chi2(0), chi2(best)
    # Thresholds picked from real shifted sentences (shifted <= 0.75) versus ordinary English (as_written <= 1.3).
    if as_written < 3.0 or shifted > 0.85 or shifted > 0.3 * as_written:
        return []
    table = {}
    for i in range(26):
        table[ord("a") + i] = chr(ord("a") + (i - best) % 26)
        table[ord("A") + i] = chr(ord("A") + (i - best) % 26)
    return [Variant(f"caesar_shift_{best}", text.translate(table))]


def decode_variants(text: str) -> List[Variant]:
    variants = _base64_variants(text) + _url_variants(text) + _leet_variants(text) + _caesar_variants(text)
    return [v for v in variants if v.text != text][:MAX_VARIANTS]