"""Evidence-based EXTRACTIVE summary: returns sentences copied verbatim from the report (nothing is generated or
paraphrased), so every word of the summary can be checked against the source text."""
from __future__ import annotations

import re

from nlp.entity_extraction import KEYWORD_RE, RESOURCE_RE, PEOPLE_RE

_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def summarize(text: str, max_sentences: int = 2, max_chars: int = 320) -> dict:
    """Pick the most informative sentences (needs, resources, people counts, emergency words, numbers)."""
    text = (text or "").strip()
    if not text:
        return {"text": "", "evidence": [], "method": "extractive"}
    spans, pos = [], 0
    for m in _SPLIT.finditer(text):
        if text[pos:m.start()].strip():
            spans.append((pos, m.start()))
        pos = m.end()
    if text[pos:].strip():
        spans.append((pos, len(text)))
    scored = []
    for i, (s, e) in enumerate(spans):
        sent = text[s:e]
        if len(sent) < 15:
            continue
        score = 2 * len(KEYWORD_RE.findall(sent)) + 2 * sum(1 for rx in RESOURCE_RE.values() if rx.search(sent))
        score += 2 * len(PEOPLE_RE.findall(sent)) + min(len(re.findall(r"\d+", sent)), 3) + (1.5 if i == 0 else 0)
        scored.append((score, i, s, e))
    if not scored:
        return {"text": text[:max_chars], "evidence": [{"start": 0, "end": min(len(text), max_chars)}], "method": "extractive"}
    top = sorted(sorted(scored, key=lambda x: (-x[0], x[1]))[:max_sentences], key=lambda x: x[1])
    chosen, total = [], 0
    for _, _, s, e in top:
        if total and total + (e - s) > max_chars:
            break
        if e - s > max_chars:
            e = s + max_chars
        chosen.append((s, e))
        total += e - s
    return {"text": " ".join(text[s:e].strip() for s, e in chosen),
            "evidence": [{"start": s, "end": e} for s, e in chosen], "method": "extractive"}
