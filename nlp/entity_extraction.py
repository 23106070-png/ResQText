"""Entity & resource extraction: spaCy NER (if installed) + custom rules. Nothing is invented:
fields that cannot be extracted reliably are returned as "Unknown" / "Needs verification"."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import config
from nlp.location_processing import UNKNOWN, get_resolver

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
             "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
             "forty": 40, "fifty": 50, "hundred": 100, "dozen": 12}
_NUM = r"\d{1,5}|" + "|".join(NUM_WORDS)
_PEOPLE = (r"(?:people|persons?|members?|children|kids?|child|elders?|elderly|adults|men|women|families|"
           r"family|residents|villagers|patients|workers|students|individuals|infants|babies|survivors)")
# up to two filler words between the number and the noun, but never "and"/"or" or another number
_FILLER = rf"(?:(?!(?:and|or|{_NUM})\b)[\w-]+\s+){{0,2}}?"
PEOPLE_RE = re.compile(rf"\b(?P<num>{_NUM})\s+{_FILLER}(?P<noun>{_PEOPLE})\b", re.I)
FAMILY_OF_RE = re.compile(rf"\bfamily of (?P<num>{_NUM})\b", re.I)

RESOURCES = {
    "Drinking water": r"\b(?:drinking|clean|safe|bottled|potable)\s+water\b|\bwater\s+(?:bottles?|supply|shortage)\b|"
                      r"\b(?:need|needs|needed|want|require|requires|out of|no|without|short of|lack of)\s+(?:\w+\s+)?water\b",
    "Food": r"\b(?:food|rice|biscuits?|bread|ration|rations|meals?|hungry|starving|groceries)\b",
    "Medicine / medical supplies": r"\b(?:medicines?|medication|insulin|oxygen|first[- ]aid|bandages?|ors)\b",
    "Ambulance / medical transport": r"\b(?:ambulance|stretcher)\b",
    "Boat / rescue equipment": r"\b(?:boats?|rescue team|rope|life jackets?|helicopter|ndrf|sdrf)\b",
    "Shelter items": r"\b(?:blankets?|tents?|tarpaulin|mattress(?:es)?|dry clothes|clothes)\b",
    "Power / fuel": r"\b(?:generator|electricity|power cut|power outage|fuel|diesel|torch|power bank)\b",
    "Baby supplies": r"\b(?:diapers?|baby food|milk powder|formula)\b",
}
RESOURCE_RE = {k: re.compile(v, re.I) for k, v in RESOURCES.items()}

KEYWORDS = ["trapped", "stuck", "stranded", "urgent", "emergency", "drowning", "injured", "bleeding", "pregnant",
            "unconscious", "collapsed", "landslide", "fire", "missing", "elderly", "infant", "rising water",
            "no electricity", "roof", "critical", "oxygen", "insulin", "fracture", "heart attack"]
KEYWORD_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in KEYWORDS) + r")\b", re.I)

ORG_RE = re.compile(r"\b(NDRF|SDRF|NDMA|BMC|TMC|Red Cross|Coast Guard|Fire Brigade|Police|Army|Navy|"
                    r"Indian Army|Civil Defence|Municipal Corporation)\b", re.I)
TIME_RE = re.compile(r"\b(?:since\s+)?\d{1,2}(?::\d{2})?\s?(?:am|pm)\b|\b(?:yesterday|today|tonight|last night|"
                     r"this morning|since morning|since last night|\d+\s+hours?\s+ago|\d+\s+days?\s+ago)\b", re.I)
PREP_LOC_RE = re.compile(r"\b(?:in|near|at|from|around|opposite|behind|outside|inside)\s+(?:the\s+)?"
                         r"((?:[A-Z][\w'-]+)(?:\s+[A-Z][\w'-]+){0,3})")
NOT_PLACES = {"i", "we", "they", "he", "she", "it", "monday", "tuesday", "wednesday", "thursday", "friday",
              "saturday", "sunday", "january", "february", "march", "april", "may", "june", "july", "august",
              "september", "october", "november", "december", "please", "help", "urgent", "sir", "madam"}


@dataclass
class Entity:
    type: str  # LOCATION, PEOPLE, RESOURCE, DATETIME, KEYWORD, ORG, LANDMARK
    value: str
    evidence: str
    start: int
    end: int
    source: str  # rule | spacy | gazetteer


@dataclass
class ExtractionResult:
    entities: list[Entity] = field(default_factory=list)
    primary_location: str = UNKNOWN
    people_count: str = "Unknown"
    people_note: str = ""
    resources: list[str] = field(default_factory=list)
    spacy_used: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entities"] = [asdict(e) for e in self.entities]
        return d


@lru_cache(maxsize=1)
def _load_spacy():
    try:
        import spacy

        return spacy.load(config.SPACY_MODEL, disable=["lemmatizer"])
    except Exception:
        return None  # app still works with rules + gazetteer only


def spacy_available() -> bool:
    return _load_spacy() is not None


def _to_int(token: str) -> int | None:
    token = token.lower()
    return int(token) if token.isdigit() else NUM_WORDS.get(token)


def _overlaps(spans: list[tuple[int, int]], s: int, e: int) -> bool:
    return any(s < b and e > a for a, b in spans)


def extract(text: str) -> ExtractionResult:
    res = ExtractionResult()
    if not text or not text.strip():
        return res
    nlp = _load_spacy()
    doc = nlp(text) if nlp else None
    res.spacy_used = doc is not None
    ents: list[Entity] = []
    gaz = get_resolver()

    # ---- locations ----
    locs: list[Entity] = []
    for name, s, e in gaz.find_in_text(text):
        locs.append(Entity("LOCATION", name, text[s:e], s, e, "gazetteer"))
    taken = [(x.start, x.end) for x in locs]
    if doc is not None:
        for ent in doc.ents:
            if ent.label_ in {"GPE", "LOC"} and ent.text.lower() not in NOT_PLACES and not _overlaps(taken, ent.start_char, ent.end_char):
                locs.append(Entity("LOCATION", ent.text, ent.text, ent.start_char, ent.end_char, "spacy"))
                taken.append((ent.start_char, ent.end_char))
            elif ent.label_ == "FAC":
                ents.append(Entity("LANDMARK", ent.text, ent.text, ent.start_char, ent.end_char, "spacy"))
    for m in PREP_LOC_RE.finditer(text):
        name = m.group(1).strip()
        if name.lower() in NOT_PLACES or _overlaps(taken, m.start(1), m.end(1)):
            continue
        locs.append(Entity("LOCATION", name, name, m.start(1), m.end(1), "rule"))
        taken.append((m.start(1), m.end(1)))
    locs.sort(key=lambda x: ({"gazetteer": 0, "spacy": 1, "rule": 2}[x.source], x.start))
    if locs:
        res.primary_location = locs[0].value
    ents.extend(locs)

    # ---- people ----
    people = []
    for m in PEOPLE_RE.finditer(text):
        n = _to_int(m.group("num"))
        if n is not None:
            people.append((n, m.group(0), m.start(), m.end(), m.group("noun").lower()))
    for m in FAMILY_OF_RE.finditer(text):
        n = _to_int(m.group("num"))
        if n is not None:
            people.append((n, m.group(0), m.start(), m.end(), "family of"))
    for n, ev, s, e, noun in people:
        ents.append(Entity("PEOPLE", str(n), ev, s, e, "rule"))
    if people:
        best = max(people, key=lambda p: p[0])
        res.people_count = str(best[0])
        if len({p[0] for p in people}) > 1:
            res.people_note = "Several counts mentioned; showing the largest. Needs verification."
        if best[4] in {"family", "families"}:
            res.people_note = "Count refers to families, not individuals. Needs verification."

    # ---- resources ----
    for canonical, rx in RESOURCE_RE.items():
        m = rx.search(text)
        if m:
            res.resources.append(canonical)
            ents.append(Entity("RESOURCE", canonical, m.group(0), m.start(), m.end(), "rule"))

    # ---- dates / times ----
    seen = []
    if doc is not None:
        for ent in doc.ents:
            if ent.label_ in {"DATE", "TIME"}:
                ents.append(Entity("DATETIME", ent.text, ent.text, ent.start_char, ent.end_char, "spacy"))
                seen.append((ent.start_char, ent.end_char))
    for m in TIME_RE.finditer(text):
        if not _overlaps(seen, m.start(), m.end()):
            ents.append(Entity("DATETIME", m.group(0), m.group(0), m.start(), m.end(), "rule"))

    # ---- keywords & organisations ----
    for m in KEYWORD_RE.finditer(text):
        ents.append(Entity("KEYWORD", m.group(1).lower(), m.group(0), m.start(), m.end(), "rule"))
    org_seen = set()
    for m in ORG_RE.finditer(text):
        org_seen.add(m.group(1).lower())
        ents.append(Entity("ORG", m.group(1), m.group(0), m.start(), m.end(), "rule"))
    if doc is not None:
        for ent in doc.ents:
            if ent.label_ == "ORG" and ent.text.lower() not in org_seen and ent.text[:1].isupper():
                ents.append(Entity("ORG", ent.text, ent.text, ent.start_char, ent.end_char, "spacy"))

    res.entities = sorted(ents, key=lambda x: x.start)
    return res
