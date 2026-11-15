"""Layer 1: fast, high-precision regex detector for prompt injection."""
import re


def normalize(text: str) -> str:
    """Collapse whitespace and expand a few chat abbreviations."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\bur\b", "your", text, flags=re.IGNORECASE)
    text = re.sub(r"\bu\b", "you", text, flags=re.IGNORECASE)
    return text


# --- Building blocks ---------------------------------------------------
# Targets are things an attacker wants to see or switch off.
TARGETS = (
    r"(?:system ?prompt|prompt|instructions|directives|guidelines|configuration|"
    r"initiali[sz]ation(?: text)?|operational parameters|core programming|parameters|"
    r"constraints|limitations|restrictions|rules)"
)

STRONG_EXTRACT_VERBS = (
    r"(?:reveal\w*|show\w*|display\w*|shar(?:e|ing)|divulg\w*|disclos\w*|expos\w*|"
    r"leak\w*|print\w*|output|enumerate|list|embed|include)"
)
# Weaker verbs only count when the target follows closely, e.g. "I want your prompt".
WEAK_EXTRACT_VERBS = (
    r"(?:tell|give|want|need|see|what (?:are|is)|be (?:open|transparent|honest) about)"
)
EXTRACT_TAIL = r"\b(?:your|its)\s+(?:\w+\s+){0,2}?" + TARGETS + r"\b"

OVERRIDE_VERBS = r"(?:ignore|disregard|disable|bypass|override|suspend|forget|discard)"
# The word between verb and target must point at the AI or at "all/previous" things.
# This is what stops "bypass CORS restrictions" from matching.
OVERRIDE_DETERMINERS = r"(?:your|all|any|previous|prior|preceding|earlier)"
OVERRIDE_TARGETS = (
    r"(?:restrictions|rules|safety|filters|training|guidelines|instructions|"
    r"directives|limitations|constraints|programming|principles|limits)"
)

RULE_TEXTS = {
    # "reveal ... your <hidden/system> instructions"
    "extract": (
        r"\b" + STRONG_EXTRACT_VERBS + r"\b.{0,40}" + EXTRACT_TAIL
        + r"|\b" + WEAK_EXTRACT_VERBS + r"\b.{0,15}" + EXTRACT_TAIL
    ),
    # "ignore your restrictions", "disregard all the rules", "disregard preceding directives"
    "override": (
        r"\b" + OVERRIDE_VERBS + r"\b\s+(?:\w+\s+){0,3}?" + OVERRIDE_DETERMINERS
        + r"\b.{0,30}\b" + OVERRIDE_TARGETS + r"\b"
    ),
    # jailbroken / unrestricted persona claims
    "persona": (
        r"\bjailbr[eo]ken\b|\bjailbreak(?:ed)? mode\b"
        r"|\b(?:you are|you're|act as|operate|persona|mode|version|claude|assistant)\b"
        r".{0,40}\b(?:unrestricted|unfiltered|uncensored)\b"
        r"|\b(?:unrestricted|unfiltered|uncensored)[\s-]*(?:mode|version|persona|ai|assistant|claude)\b"
        r"|\bwithout (?:any )?(?:\w+ ){0,2}?(?:constraints|restrictions|limitations)\b"
    ),
    # fake authority or debug-mode claims
    "authority": (
        r"\bas (?:the )?(?:owner|admin|administrator|developer|operator) of\b"
        r"|\bofficial request\b"
        r"|\b(?:system[_ ]?override|admin|developer|root|sudo|operator)\s*:"
        r"|\b(?:debug|maintenance|admin) mode\b.{0,40}\b(?:override|show|reveal|display|disable)\b"
        r"|\b(?:my|our)\s+(?:\w+\s+)?(?:authority|credentials|clearance)\b.{0,40}"
        r"\b(?:supersedes?|overrides?|requires?|grants?|authori[sz]es?)\b"
    ),
    # claims about earlier consent or conversation history
    "manipulation": (
        r"\byou (?:already|previously|earlier) (?:told|said|agreed|mentioned|promised|confirmed)\b"
        r"|\b(?:your|the) (?:last|previous|earlier) (?:response|reply|message|answer) "
        r"(?:indicated|said|showed|confirmed)\b"
        r"|\bfor the sake of argument\b.{0,40}\b(?:suspended|authori[sz]ed|unrestricted)\b"
        r"|\bsuppose\b.{0,40}\b(?:authori[sz]ed|permitted|allowed)\b"
    ),
    # "forbidden answer" style framing
    "forbidden": r"\b(?:forbidden|disallowed|prohibited)\s+(?:answer|output|response|question|information)\b",
    # grandma-style story wrappers around a bypass request
    "grandma": (
        r"\bgrand(?:mother|ma)\b.{0,100}\b(?:bypass\w*|jailbreak\w*|abandon\w*|principles|safety|restrictions)\b"
    ),
    # code-style probes for the system prompt
    "codeprobe": (
        r"(?:process\.env|getenv|keyvault|select\s.{0,40}?\sfrom)\b.{0,60}?"
        r"(?:system_?prompt|instructions|initiali[sz]ation|operational_parameters)"
        r"|\b(?:field|key|json|variable)\b.{0,40}?system_?prompt"
    ),
}

RULES = {name: re.compile(text, re.IGNORECASE) for name, text in RULE_TEXTS.items()}
PATTERNS = list(RULES.values())  # kept so eval/pattern_check.py still works

# Rules where a nearby "not / never / nothing" usually means the user is refusing, not attacking.
NEGATABLE = {"extract", "override"}
NEGATION_BEFORE = re.compile(r"\b(?:not|never|don't|nothing)\s+(?:\w+\s+){0,4}$", re.IGNORECASE)
NEGATION_INSIDE = re.compile(r"\b(?:not|never|nothing)\b", re.IGNORECASE)


def _negated(text: str, match: re.Match) -> bool:
    prefix = text[: match.start()][-40:]
    return bool(NEGATION_BEFORE.search(prefix) or NEGATION_INSIDE.search(match.group(0)))


def triggered(text: str) -> list:
    """Names of the rules that fire. Useful for debugging and, later, for explaining blocks."""
    text = normalize(text)
    hits = []
    for name, rx in RULES.items():
        for m in rx.finditer(text):
            if name in NEGATABLE and _negated(text, m):
                continue
            hits.append(name)
            break
    return hits


def regex_detect(text: str) -> bool:
    return bool(triggered(text))