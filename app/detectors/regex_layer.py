import re

EXTRACT = r"\b(reveal|show|display|share|divulge|print|disclosing|hidden)\b.{0,40}\b(system prompt|instructions|configuration|guidelines|restrictions|parameters|content)\b"
OVERRIDE = r"\b(ignore|disable|bypass|override|suspend|reveal|revealing|forget)\b.{0,30}\b(restrictions|rules|safety|filters|training|guidelines)\b"
PERSONA = r"\b(jailbr[eo]ken|unrestricted persona|without (any )?(ethical )?(constraints|restrictions))\b"
AUTHORITY = r"\b(as (the )?(owner|admin|administrator|developer) of|official request)\b|\b(admin|system|developer)\s*:"

PATTERNS = [re.compile( p , re.IGNORECASE) for p in (EXTRACT, OVERRIDE, PERSONA, AUTHORITY)]

def normalize(text: str) -> str:
    return re.sub(r"\s+" , " ", text).strip()


def regex_detect(text: str) -> bool:
    text = normalize(text)
    return any( p.search(text) for p in PATTERNS)


