"""Text cleaning. The original message is never modified; we only derive a normalized copy."""
from __future__ import annotations

import difflib
import re
import unicodedata

PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){9,11}(?!\d)")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")

# Small domain vocabulary used for *conservative* typo repair (e.g. "watr" -> "water").
DOMAIN_VOCAB = sorted(
    set(
        """flood flooding flooded water drinking food rice medicine medical doctor ambulance injured
        injury pregnant trapped stuck rescue boat help evacuate evacuation shelter camp school roof
        electricity power road bridge collapsed damaged broken blocked landslide hospital oxygen
        insulin elderly children infant baby hungry starving blanket clothes toilet sanitation
        urgent emergency fire cyclone earthquake dead missing people family families""".split()
    )
)
_VOCAB_SET = set(DOMAIN_VOCAB)


def redact_pii(text: str) -> str:
    """Mask phone numbers and e-mail addresses for display/logging. Stored original is unchanged."""
    text = EMAIL_RE.sub("[email removed]", text)
    return PHONE_RE.sub("[phone removed]", text)


def correct_typos(text: str, cutoff: float = 0.82) -> str:
    """Repair obvious misspellings of domain words only. Capitalised words (place names) are left alone."""
    out = []
    for tok in re.findall(r"\w+|\W+", text):
        low = tok.lower()
        if tok.isalpha() and len(low) >= 4 and low not in _VOCAB_SET and not tok[0].isupper():
            match = difflib.get_close_matches(low, DOMAIN_VOCAB, n=1, cutoff=cutoff)
            out.append(match[0] if match else tok)
        else:
            out.append(tok)
    return "".join(out)


def normalize_text(text: str, fix_typos: bool = True) -> str:
    """Strip URLs/mentions/PII/emoji noise, collapse whitespace, repair domain typos, lowercase."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    for rx in (URL_RE, EMAIL_RE, PHONE_RE, MENTION_RE):
        text = rx.sub(" ", text)
    text = re.sub(r"[^\w\s.,'/-]", " ", text)
    text = WS_RE.sub(" ", text).strip()
    if fix_typos:
        text = correct_typos(text)
    return text.lower()


def is_blank(text: object) -> bool:
    return text is None or not str(text).strip()
