"""
=============================================================================
 event_extractor.py  –  Narrative Timeline Extractor
=============================================================================
 PURPOSE
 -------
 Step 3 of the pipeline:
   • Determine which sentences are "events" (not background / meta text)
   • Score each sentence's event-importance
   • Extract a short description (headline) for each event
   • Sort events chronologically

 APPROACH : Rule-based + ML Hybrid
 ----------------------------------
 1. Rule-based signals  (fast, interpretable):
    - Presence of action verbs (launched, announced, …)
    - Presence of a date/time expression
    - Sentence position (early sentences often set context)
    - Named-entity density

 2. TF-IDF keyword weight  (lightweight ML):
    - Build a TF-IDF matrix over all sentences
    - High-TF-IDF sentences contain distinctive terms → likely events

 3. Combined score = weighted sum of all signals
=============================================================================
"""

import re
import math
from collections import Counter
from datetime import datetime
from typing import Optional

from preprocessor import ACTION_VERBS, STOP_WORDS, remove_stopwords, tokenise


# ─────────────────────────────────────────────────────────────────────────────
# 1.  TF-IDF SCORER  (no external library)
# ─────────────────────────────────────────────────────────────────────────────

def _build_tfidf(token_lists: list[list[str]]) -> list[dict[str, float]]:
    """
    Build TF-IDF scores for each document (sentence) in the corpus.

    Returns
    -------
    List of dicts: token → tfidf score, one dict per sentence
    """
    N = len(token_lists)
    if N == 0:
        return []

    # Document frequency: how many sentences contain each token
    df: dict[str, int] = Counter()
    for tokens in token_lists:
        for t in set(tokens):
            df[t] += 1

    # IDF: log(N / df)
    idf: dict[str, float] = {t: math.log(N / (df[t] + 1)) + 1 for t in df}

    # TF-IDF per sentence
    tfidf_scores = []
    for tokens in token_lists:
        tf: dict[str, int] = Counter(tokens)
        total = len(tokens) if tokens else 1
        scores = {t: (tf[t] / total) * idf[t] for t in tf}
        tfidf_scores.append(scores)

    return tfidf_scores


def _sentence_tfidf_score(tfidf: dict[str, float]) -> float:
    """Mean TF-IDF score of a sentence (= content richness indicator)."""
    if not tfidf:
        return 0.0
    return sum(tfidf.values()) / len(tfidf)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  RULE-BASED SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

def _action_verb_score(tokens: list[str]) -> float:
    """
    Fraction of tokens that are action verbs.
    Sentences with many action verbs are almost certainly events.
    """
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in ACTION_VERBS)
    return min(hits / max(len(tokens), 1) * 10, 1.0)   # scale & cap at 1


def _entity_density(entities: dict) -> float:
    """
    Score based on number of named entities found.
    Events typically mention who / where.
    """
    total = (len(entities.get("persons", [])) +
             len(entities.get("orgs", [])) +
             len(entities.get("locations", [])))
    return min(total * 0.25, 1.0)


_CAUSAL_WORDS = re.compile(
    r"\b(because|therefore|consequently|as a result|due to|led to|"
    r"caused|resulting in|following|after|subsequently)\b",
    re.IGNORECASE,
)

def _causal_signal(sentence: str) -> float:
    """Causal / sequential language often accompanies events."""
    matches = _CAUSAL_WORDS.findall(sentence)
    return min(len(matches) * 0.3, 0.9)


_NUMERIC_RE = re.compile(r"\$[\d,.]+|\b\d+[\d,]*\s*(?:million|billion|thousand|%|percent)?\b")

def _numeric_signal(sentence: str) -> float:
    """
    Sentences with specific numbers (prices, percentages) are usually events.
    """
    matches = _NUMERIC_RE.findall(sentence)
    return min(len(matches) * 0.2, 0.8)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  COMPOSITE SCORING
# ─────────────────────────────────────────────────────────────────────────────

# Weights for each signal  (must sum to 1.0)
WEIGHTS = {
    "has_date"    : 0.30,   # date/time present → almost certainly an event
    "action_verb" : 0.25,   # contains action verb
    "tfidf"       : 0.20,   # content richness
    "entities"    : 0.12,   # named entity density
    "causal"      : 0.07,   # causal language
    "numeric"     : 0.06,   # specific numbers
}

def score_sentence(
    sentence    : str,
    tokens      : list[str],
    entities    : dict,
    has_date    : bool,
    tfidf_dict  : dict[str, float],
) -> float:
    """
    Compute a 0–1 event importance score for one sentence.
    Higher = more likely to be a timeline-worthy event.
    """
    signals = {
        "has_date"    : 1.0 if has_date else 0.0,
        "action_verb" : _action_verb_score(tokens),
        "tfidf"       : min(_sentence_tfidf_score(tfidf_dict) * 3, 1.0),
        "entities"    : _entity_density(entities),
        "causal"      : _causal_signal(sentence),
        "numeric"     : _numeric_signal(sentence),
    }
    score = sum(WEIGHTS[k] * signals[k] for k in WEIGHTS)
    return round(score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DESCRIPTION GENERATION  (extractive headline)
# ─────────────────────────────────────────────────────────────────────────────

def generate_description(sentence: str, max_words: int = 15) -> str:
    """
    Generate a short headline-style description from a sentence.

    Strategy
    --------
    1. Find the first action verb in the sentence.
    2. Take the substring from just before the verb to end of sentence.
    3. Trim to `max_words` words.
    4. Capitalise and clean up.

    Falls back to simply taking the first `max_words` words.
    """
    sentence = sentence.strip()

    # Strategy 1: find action verb and build phrase around it
    words = sentence.split()
    for i, word in enumerate(words):
        clean = re.sub(r"[^a-z]", "", word.lower())
        if clean in ACTION_VERBS:
            # Take a window of words starting 2 before the verb
            start = max(0, i - 2)
            snippet = " ".join(words[start: start + max_words])
            snippet = snippet.rstrip(".,;:")
            return snippet[0].upper() + snippet[1:] if snippet else sentence[:80]

    # Fallback: first max_words words
    snippet = " ".join(words[:max_words]).rstrip(".,;:")
    return snippet[0].upper() + snippet[1:] if snippet else sentence[:80]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TEMPORAL SORTING
# ─────────────────────────────────────────────────────────────────────────────

def _sort_key(event: dict):
    """
    Sorting key for events:
      - Events with a normalised datetime sort by that datetime
      - Events without a date sort to the end (preserving document order)
    """
    dt = event.get("datetime")
    if dt and isinstance(dt, datetime):
        return (0, dt, event["doc_order"])
    # No date → maintain document order but placed after dated events
    return (1, datetime(9999, 12, 31), event["doc_order"])


def sort_events(events: list[dict]) -> list[dict]:
    """
    Sort events chronologically.

    Events with explicit dates are sorted by date.
    Events without dates are placed after all dated events,
    in the order they appeared in the original text.

    Parameters
    ----------
    events : list of event dicts (from extract_events)

    Returns
    -------
    Sorted list of event dicts, with 'event_number' field added
    """
    sorted_evts = sorted(events, key=_sort_key)
    for i, evt in enumerate(sorted_evts, 1):
        evt["event_number"] = i
    return sorted_evts


# ─────────────────────────────────────────────────────────────────────────────
# 6.  HIGHLIGHT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def highlight_events_in_text(original_text: str, events: list[dict]) -> str:
    """
    Return the original text with event sentences wrapped in
    <mark class="event-N"> … </mark> tags for HTML display.

    Parameters
    ----------
    original_text : the cleaned input text
    events        : sorted list of event dicts

    Returns
    -------
    HTML string with event sentences highlighted
    """
    result = original_text
    # Sort events by sentence length descending to avoid nested replacements
    sorted_by_len = sorted(events, key=lambda e: len(e["sentence"]), reverse=True)
    for evt in sorted_by_len:
        sent = re.escape(evt["sentence"])
        n    = evt.get("event_number", 0)
        replacement = f'<mark class="event-{n}" data-event="{n}">{evt["sentence"]}</mark>'
        result = re.sub(sent, replacement, result, count=1)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7.  FULL EXTRACTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

# Minimum score threshold to be included as an event
EVENT_THRESHOLD = 0.18

def extract_events(ner_results: list[dict], token_lists: list[list[str]]) -> list[dict]:
    """
    Main event extraction function.

    Parameters
    ----------
    ner_results  : output from ner_extractor.run_ner()
    token_lists  : per-sentence token lists from preprocessor

    Returns
    -------
    Sorted list of event dicts, each containing:
      'event_number' – 1-based index after sorting
      'sentence'     – original sentence
      'description'  – extractive headline
      'date_text'    – date string as found in text (or "Unknown")
      'datetime'     – Python datetime object (or None)
      'date_conf'    – confidence of date extraction (0–1)
      'event_score'  – overall event importance score (0–1)
      'entities'     – extracted named entities dict
      'doc_order'    – position in original document
    """
    # Build TF-IDF across all sentences
    tfidf_all = _build_tfidf(token_lists)

    events = []
    for i, (ner, tokens, tfidf) in enumerate(
        zip(ner_results, token_lists, tfidf_all)
    ):
        sentence = ner["sentence"]
        entities = ner["entities"]
        dates    = ner["dates"]
        has_date = ner["has_date"]

        # Compute composite event score
        evt_score = score_sentence(sentence, tokens, entities, has_date, tfidf)

        # Skip sentences below threshold (likely background / transition text)
        if evt_score < EVENT_THRESHOLD:
            continue

        # Best date for this sentence (highest confidence)
        best_date = max(dates, key=lambda d: d["confidence"]) if dates else None

        events.append({
            "sentence"   : sentence,
            "description": generate_description(sentence),
            "date_text"  : best_date["text"]        if best_date else "Unknown",
            "datetime"   : best_date["normalized"]   if best_date else None,
            "date_conf"  : best_date["confidence"]   if best_date else 0.0,
            "event_score": evt_score,
            "entities"   : entities,
            "doc_order"  : i,
            "all_dates"  : dates,
        })

    return sort_events(events)


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from preprocessor import preprocess
    from ner_extractor import run_ner

    sample = """
    In January 2020, a new virus was first reported in Wuhan, China.
    By March 11, 2020, the World Health Organization declared a global pandemic.
    On March 15, 2020, many countries imposed nationwide lockdowns.
    Scientists began developing vaccines in early 2020.
    In December 2020, the first COVID-19 vaccines received emergency authorization.
    By mid-2021, vaccination campaigns were underway worldwide.
    The global economy began recovering in late 2021 and early 2022.
    """

    processed = preprocess(sample)
    ner_out   = run_ner(processed["sentences"])
    events    = extract_events(ner_out, processed["tokens"])

    print(f"\nExtracted {len(events)} events:\n")
    for e in events:
        print(f"  Event {e['event_number']:2d} | {e['date_text']:20s} | Score={e['event_score']:.2f} | {e['description']}")
