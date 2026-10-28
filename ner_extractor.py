"""
=============================================================================
 ner_extractor.py  –  Narrative Timeline Extractor
=============================================================================
 PURPOSE
 -------
 Step 2 of the pipeline:
   • Extract DATE / TIME expressions from each sentence
   • Extract named entities: persons, organisations, locations
   • Normalise extracted dates into a sortable datetime object
   • Assign a confidence score to each extraction

 APPROACH
 --------
 Rule-based NER using carefully crafted regular expressions.
 This is the standard hybrid approach used in many production systems:
   - No large model download required
   - Fast, transparent, and debuggable
   - Handles the most common date/time patterns in English text

 When spaCy / transformers ARE available (e.g. on a server with internet),
 the drop-in replacement using spaCy is shown at the bottom of this file.
=============================================================================
"""

import re
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATE / TIME PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

MONTH_MAP = {
    "january":1, "jan":1, "february":2, "feb":2, "march":3, "mar":3,
    "april":4,  "apr":4, "may":5,      "june":6, "jun":6,
    "july":7,   "jul":7, "august":8,  "aug":8,  "september":9, "sep":9, "sept":9,
    "october":10,"oct":10,"november":11,"nov":11,"december":12,"dec":12,
}

# Build month name regex group
_MONTH_NAMES = "|".join(
    sorted(MONTH_MAP.keys(), key=len, reverse=True)  # longest first to avoid partial matches
)

# All the patterns we can match (ordered from most-specific to least)
DATE_PATTERNS = [
    # Full ISO  2022-06-15
    (r"\b(\d{4})-(\d{2})-(\d{2})\b",                                   "ISO"),
    # DD/MM/YYYY or MM/DD/YYYY
    (r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b",                      "SLASH"),
    # Month DD, YYYY   e.g. "January 5, 2022"
    (rf"\b({_MONTH_NAMES})\.?\s+(\d{{1,2}})[,\s]+(\d{{4}})\b",         "MDY", re.IGNORECASE),
    # DD Month YYYY    e.g. "5 January 2022"
    (rf"\b(\d{{1,2}})\s+({_MONTH_NAMES})\.?\s+(\d{{4}})\b",            "DMY", re.IGNORECASE),
    # Month YYYY       e.g. "January 2022"
    (rf"\b({_MONTH_NAMES})\.?\s+(\d{{4}})\b",                          "MY",  re.IGNORECASE),
    # "in 2022" / "by 2022" / "since 2020"
    (r"\b(?:in|by|since|during|after|before|from|until|of)\s+(\d{4})\b", "YEAR_PREP", re.IGNORECASE),
    # Bare 4-digit year as last resort (inside larger phrase)
    (r"\b((?:19|20)\d{2})\b",                                            "YEAR"),
    # Relative: "last year", "this month", "next week"
    (r"\b(last|this|next)\s+(year|month|week|quarter)\b",               "REL",  re.IGNORECASE),
    # Quarters: "Q1 2022", "first quarter of 2021"
    (r"\bQ([1-4])\s+(\d{4})\b",                                         "QUARTER"),
    # Early/mid/late + period
    (r"\b(early|mid|late)\s+(\d{4}|(?:" + _MONTH_NAMES + r"))\b",      "PERIOD", re.IGNORECASE),
]

# Pre-compile all patterns
_COMPILED_PATTERNS = []
for item in DATE_PATTERNS:
    pattern_str, tag = item[0], item[1]
    flags = item[2] if len(item) > 2 else 0
    _COMPILED_PATTERNS.append((re.compile(pattern_str, flags), tag))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_dates(sentence: str) -> list[dict]:
    """
    Find all date/time expressions in a sentence.

    Returns
    -------
    List of dicts, each with:
      'text'       – matched text as it appears in the sentence
      'normalized' – datetime object if parseable, else None
      'confidence' – float 0–1 (higher = more specific date)
      'tag'        – internal pattern tag for debugging
    """
    found  = []
    seen_spans = []   # track char positions to avoid overlapping matches

    for pattern, tag in _COMPILED_PATTERNS:
        for m in pattern.finditer(sentence):
            start, end = m.start(), m.end()

            # Skip if this span overlaps a previously matched span
            overlap = any(
                not (end <= s or start >= e)
                for s, e in seen_spans
            )
            if overlap:
                continue

            text = m.group(0).strip()
            normalized, confidence = _normalize_date(m, tag, text)

            found.append({
                "text"      : text,
                "normalized": normalized,
                "confidence": confidence,
                "tag"       : tag,
                "span"      : (start, end),
            })
            seen_spans.append((start, end))

    # Sort by position in sentence
    found.sort(key=lambda x: x["span"][0])
    return found


def _normalize_date(match, tag: str, text: str) -> tuple[Optional[datetime], float]:
    """
    Parse a regex match into a Python datetime and a confidence score.

    Confidence scale:
      1.0 = exact date (year + month + day)
      0.8 = year + month
      0.6 = year only (from preposition)
      0.5 = bare year
      0.3 = relative / vague
    """
    try:
        g = match.groups()

        if tag == "ISO":
            year, month, day = int(g[0]), int(g[1]), int(g[2])
            return datetime(year, month, day), 1.0

        elif tag == "SLASH":
            # Heuristic: if first number > 12, it must be DD/MM/YYYY
            a, b, year = int(g[0]), int(g[1]), int(g[2])
            if a > 12:
                day, month = a, b
            else:
                month, day = a, b
            return datetime(year, month, min(day, 28)), 1.0

        elif tag == "MDY":
            month_str, day, year = g[0].lower(), int(g[1]), int(g[2])
            month = MONTH_MAP.get(month_str.rstrip("."), 1)
            return datetime(year, month, min(int(day), 28)), 1.0

        elif tag == "DMY":
            day, month_str, year = int(g[0]), g[1].lower(), int(g[2])
            month = MONTH_MAP.get(month_str.rstrip("."), 1)
            return datetime(year, month, min(day, 28)), 1.0

        elif tag == "MY":
            month_str, year = g[0].lower(), int(g[1])
            month = MONTH_MAP.get(month_str.rstrip("."), 1)
            return datetime(year, month, 1), 0.8

        elif tag == "YEAR_PREP":
            year = int(g[0])
            return datetime(year, 1, 1), 0.6

        elif tag == "YEAR":
            year = int(g[0])
            return datetime(year, 1, 1), 0.5

        elif tag == "QUARTER":
            q, year = int(g[0]), int(g[1])
            month_start = (q - 1) * 3 + 1
            return datetime(year, month_start, 1), 0.75

        elif tag == "PERIOD":
            # "early 2022" → Jan 2022; "mid" → Jun; "late" → Oct
            qualifier = g[0].lower()
            period    = g[1].lower()
            if period.isdigit():
                year  = int(period)
                month = 1 if qualifier == "early" else 6 if qualifier == "mid" else 10
                return datetime(year, month, 1), 0.55
            else:
                return None, 0.3

        elif tag == "REL":
            # Relative expressions – no absolute datetime
            return None, 0.3

    except (ValueError, IndexError, TypeError):
        pass

    return None, 0.2


# ─────────────────────────────────────────────────────────────────────────────
# 3.  NAMED ENTITY EXTRACTION  (persons, orgs, locations)
# ─────────────────────────────────────────────────────────────────────────────

# Simple gazetteer-style keyword lists for common org / location indicators
ORG_SUFFIXES    = r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*(?:\s+(?:Inc|Corp|Ltd|LLC|Co|Group|Institute|University|College|School|Foundation|Agency|Ministry|Department|Bureau|Association|Federation|Union|Organization|Organisation|Committee|Commission|Council|Authority|Bank|Fund|Trust|Labs|Technologies|Solutions|Services|Systems|International|Global)\.?)\b"
PERSON_TITLES   = r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|President|CEO|CTO|CFO|Director|Minister|Senator|Governor|General|Captain|Commander|Chief)\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\b"
LOCATION_WORDS  = r"\b(?:in|at|from|to|near|across|throughout)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"

_ORG_RE    = re.compile(ORG_SUFFIXES)
_PERSON_RE = re.compile(PERSON_TITLES)
_LOC_RE    = re.compile(LOCATION_WORDS)

def extract_entities(sentence: str) -> dict:
    """
    Extract named entities from a sentence.

    Returns
    -------
    dict with keys 'persons', 'orgs', 'locations' — each a list of strings
    """
    persons   = [m.group(0).strip() for m in _PERSON_RE.finditer(sentence)]
    orgs      = [m.group(0).strip() for m in _ORG_RE.finditer(sentence)]
    locations = [m.group(1).strip() for m in _LOC_RE.finditer(sentence)
                 if len(m.group(1)) > 2]   # filter single-char noise

    # Deduplicate
    return {
        "persons"  : list(dict.fromkeys(persons)),
        "orgs"     : list(dict.fromkeys(orgs)),
        "locations": list(dict.fromkeys(locations)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FULL NER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_ner(sentences: list[str]) -> list[dict]:
    """
    Run full NER on every sentence.

    Parameters
    ----------
    sentences : list of sentence strings from the preprocessor

    Returns
    -------
    List of dicts, one per sentence:
      'sentence'  – original sentence text
      'dates'     – list of date dicts from extract_dates()
      'entities'  – dict from extract_entities()
      'has_date'  – bool convenience flag
    """
    results = []
    for sent in sentences:
        dates    = extract_dates(sent)
        entities = extract_entities(sent)
        results.append({
            "sentence": sent,
            "dates"   : dates,
            "entities": entities,
            "has_date": len(dates) > 0,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SPACY DROP-IN  (uncomment when spaCy is installed)
# ─────────────────────────────────────────────────────────────────────────────
"""
import spacy
_nlp = spacy.load("en_core_web_sm")

def run_ner_spacy(sentences):
    results = []
    for sent in sentences:
        doc   = _nlp(sent)
        dates = [{"text": ent.text, "normalized": None, "confidence": 0.85, "tag": "SPACY"}
                 for ent in doc.ents if ent.label_ in ("DATE", "TIME")]
        entities = {
            "persons"  : [e.text for e in doc.ents if e.label_ == "PERSON"],
            "orgs"     : [e.text for e in doc.ents if e.label_ == "ORG"],
            "locations": [e.text for e in doc.ents if e.label_ in ("GPE","LOC")],
        }
        results.append({"sentence": sent, "dates": dates, "entities": entities, "has_date": bool(dates)})
    return results
"""


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_sentences = [
        "On January 5, 2022, Dr. Smith founded the startup in San Francisco.",
        "By March 2022, the company launched its flagship product.",
        "In Q3 2022, they raised $10M in funding.",
        "The CEO announced global expansion in December 2022.",
        "Early 2023 saw the company enter the Asian market.",
    ]
    ner_results = run_ner(test_sentences)
    for r in ner_results:
        print(f"\nSentence: {r['sentence'][:60]}...")
        print(f"  Dates   : {[d['text'] for d in r['dates']]}")
        print(f"  Entities: {r['entities']}")
