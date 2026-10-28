"""
=============================================================================
 preprocessor.py  –  Narrative Timeline Extractor
=============================================================================
 PURPOSE
 -------
 Step 1 of the pipeline:
   • Clean raw input text (fix encoding, whitespace, etc.)
   • Segment into individual sentences
   • Tokenise each sentence into words
   • Provide helper utilities used by other modules

 WHY THIS MATTERS
 ----------------
 All downstream NLP (NER, event extraction) works at the sentence level.
 Good segmentation = better event detection.
=============================================================================
"""

import re
import unicodedata


# ─────────────────────────────────────────────────────────────────────────────
# 1.  TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise and clean raw input text.

    Steps
    -----
    1. Unicode normalisation (handles curly quotes, em-dashes, etc.)
    2. Replace HTML entities (&amp; → &, etc.)
    3. Collapse multiple whitespace / newlines into single spaces
    4. Strip leading / trailing whitespace

    Parameters
    ----------
    text : raw input string from the user

    Returns
    -------
    Cleaned string ready for segmentation
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Normalise unicode (NFKC handles ligatures, fancy quotes, etc.)
    text = unicodedata.normalize("NFKC", text)

    # 2. Replace common HTML entities
    html_entities = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#39;": "'", "&nbsp;": " ",
    }
    for entity, char in html_entities.items():
        text = text.replace(entity, char)

    # 3. Normalise dashes and quotes to ASCII equivalents
    text = text.replace("\u2013", "-").replace("\u2014", " - ")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # 4. Collapse multiple whitespace / newlines
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SENTENCE SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

# Abbreviations that end with a period but are NOT sentence boundaries
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc",
    "inc", "ltd", "corp", "co", "est", "dept", "approx", "govt",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "st", "ave", "blvd", "rd",
}

def _is_abbreviation(word: str) -> bool:
    """Return True if `word` (before a period) is a known abbreviation."""
    return word.lower().rstrip(".") in _ABBREVIATIONS


def segment_sentences(text: str) -> list[str]:
    """
    Split text into a list of sentences using rule-based segmentation.

    Algorithm
    ---------
    • Split on sentence-ending punctuation (.  !  ?) followed by whitespace
      and an uppercase letter or digit.
    • Protect abbreviations (Dr., Jan., etc.) from false splits.
    • Filter out empty / very short fragments.

    Parameters
    ----------
    text : cleaned input string

    Returns
    -------
    List of sentence strings, in document order
    """
    if not text:
        return []

    # Protect known abbreviations by replacing their periods temporarily
    protected = text
    for abbr in _ABBREVIATIONS:
        # e.g.  "Dr." → "Dr<DOT>"
        pattern = re.compile(r"\b" + re.escape(abbr) + r"\.", re.IGNORECASE)
        protected = pattern.sub(abbr.capitalize() + "<DOT>", protected)

    # Core split: punctuation followed by space + uppercase / digit
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])", protected)

    sentences = []
    for part in parts:
        # Restore protected periods
        part = part.replace("<DOT>", ".")
        part = part.strip()
        # Discard fragments shorter than 15 characters (noise)
        if len(part) >= 15:
            sentences.append(part)

    return sentences


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TOKENISATION
# ─────────────────────────────────────────────────────────────────────────────

def tokenise(sentence: str) -> list[str]:
    """
    Split a sentence into individual word tokens.

    • Lowercases
    • Keeps alphanumeric tokens and apostrophe-contractions
    • Removes pure punctuation tokens

    Returns
    -------
    List of lowercase word strings
    """
    tokens = re.findall(r"\b[\w']+\b", sentence.lower())
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SIMPLE POS HELPERS  (no external library needed)
# ─────────────────────────────────────────────────────────────────────────────

# Common English stop-words
STOP_WORDS = {
    "the", "a", "an", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "up",
    "about", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again", "then",
    "once", "and", "but", "or", "nor", "not", "so", "yet", "both",
    "either", "neither", "each", "every", "all", "any", "few", "more",
    "most", "other", "some", "such", "no", "than", "too", "very",
    "just", "because", "as", "until", "while", "although", "though",
    "if", "unless", "since", "where", "when", "how", "what", "which",
    "who", "whom", "whose", "that", "this", "these", "those",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "theirs",
}

# Action verbs that strongly signal events
ACTION_VERBS = {
    "launched", "announced", "released", "published", "revealed", "unveiled",
    "introduced", "created", "founded", "established", "started", "began",
    "opened", "closed", "ended", "finished", "completed", "achieved",
    "acquired", "merged", "partnered", "signed", "agreed", "approved",
    "rejected", "cancelled", "suspended", "resigned", "appointed", "hired",
    "fired", "promoted", "elected", "won", "lost", "defeated",
    "raised", "invested", "funded", "raised", "sold", "bought", "purchased",
    "expanded", "grew", "increased", "decreased", "fell", "rose", "dropped",
    "arrested", "charged", "sentenced", "convicted", "acquitted", "released",
    "attacked", "killed", "died", "survived", "injured", "saved",
    "discovered", "found", "identified", "confirmed", "denied", "claimed",
    "reported", "stated", "said", "declared", "warned", "predicted",
    "banned", "blocked", "lifted", "imposed", "removed", "changed",
    "built", "developed", "deployed", "shipped", "tested", "demonstrated",
    "entered", "joined", "left", "departed", "arrived", "returned",
    "passed", "failed", "graduated", "enrolled",
}


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove common stop-words from a token list."""
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PREPROCESSING PIPELINE  (called by other modules)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(raw_text: str) -> dict:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    raw_text : user-supplied text

    Returns
    -------
    dict with keys:
      'cleaned'   – cleaned text string
      'sentences' – list of sentence strings
      'tokens'    – list of per-sentence token lists (stop-words removed)
    """
    cleaned   = clean_text(raw_text)
    sentences = segment_sentences(cleaned)
    tokens    = [remove_stopwords(tokenise(s)) for s in sentences]

    return {
        "cleaned"  : cleaned,
        "sentences": sentences,
        "tokens"   : tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    On January 5, 2022, Dr. Smith founded the startup. By March 2022,
    the company launched its flagship product. In June, they raised $10M
    in Series A funding. The CEO announced global expansion in Dec. 2022.
    """
    result = preprocess(sample)
    print(f"Sentences found: {len(result['sentences'])}")
    for i, s in enumerate(result['sentences'], 1):
        print(f"  [{i}] {s}")
