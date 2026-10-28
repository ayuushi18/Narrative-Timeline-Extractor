"""
=============================================================================
 timeline_engine.py  –  Narrative Timeline Extractor
=============================================================================
 PURPOSE
 -------
 Master pipeline module that orchestrates all steps:
   1. Preprocess text
   2. Run NER
   3. Extract and score events
   4. Sort chronologically
   5. Return structured output ready for the Streamlit dashboard

 This is the ONLY module that app.py needs to import.
=============================================================================
"""

import csv
import io
import json
from datetime import datetime
from typing import Optional

from preprocessor  import preprocess
from ner_extractor import run_ner
from event_extractor import extract_events, highlight_events_in_text


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE TEXTS  (for the dashboard demo)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_TEXTS = {
    "🦠 COVID-19 Pandemic": """
In December 2019, a cluster of pneumonia cases was first reported in Wuhan, China.
By January 2020, Chinese authorities confirmed a novel coronavirus was the cause.
On January 30, 2020, the World Health Organization declared a Public Health Emergency of International Concern.
In February 2020, the disease was officially named COVID-19.
On March 11, 2020, the WHO declared COVID-19 a global pandemic.
Many countries imposed nationwide lockdowns in March and April 2020 to slow the spread.
Scientists around the world began developing vaccines at an unprecedented pace in early 2020.
In May 2020, the United States launched Operation Warp Speed to accelerate vaccine development.
By November 2020, Pfizer and BioNTech announced their vaccine showed 90% efficacy.
In December 2020, the first COVID-19 vaccines received emergency use authorization.
Vaccination campaigns rolled out across Europe and North America starting in January 2021.
By mid-2021, over 2 billion vaccine doses had been administered globally.
New variants including Delta and Omicron emerged in 2021 and spread rapidly.
In late 2021, many countries began lifting restrictions as vaccination rates rose.
The global economy began its recovery in 2022, though supply chain disruptions persisted.
    """,

    "🚀 SpaceX Milestones": """
Elon Musk founded SpaceX in May 2002 with the goal of reducing space transportation costs.
In September 2008, Falcon 1 became the first privately developed liquid-fueled rocket to reach orbit.
SpaceX launched the Dragon spacecraft for the first time in December 2010.
In May 2012, Dragon became the first commercial spacecraft to dock with the International Space Station.
The company successfully landed its first orbital rocket booster in December 2015.
In February 2018, the Falcon Heavy rocket launched carrying a Tesla Roadster as payload.
SpaceX flew its first crewed mission, Demo-2, to the ISS on May 30, 2020.
In 2021, SpaceX's Inspiration4 mission sent the first all-civilian crew to orbit in September.
The Starship prototype conducted its first successful high-altitude test in May 2021.
By 2022, SpaceX had launched over 30 missions, setting a new annual record.
    """,

    "📱 Apple Inc. History": """
Steve Jobs and Steve Wozniak founded Apple Computer Company on April 1, 1976.
Apple launched the Apple II in June 1977, which became one of the first successful mass-produced personal computers.
In January 1984, Apple introduced the Macintosh computer with a revolutionary graphical interface.
Steve Jobs was ousted from Apple in 1985 following a power struggle with the board of directors.
Apple acquired NeXT in December 1996, bringing Steve Jobs back to the company.
Jobs returned as CEO in September 1997 and immediately began restructuring the company.
Apple launched the iMac in August 1998, reviving the company's fortunes with a bold new design.
The iTunes Store launched in April 2003, transforming the music industry.
On January 9, 2007, Steve Jobs introduced the first iPhone, revolutionizing the smartphone market.
The App Store opened in July 2008 with 500 apps available at launch.
Apple introduced the iPad in January 2010, creating the modern tablet market.
Steve Jobs passed away on October 5, 2011, after a long battle with pancreatic cancer.
Tim Cook took over as CEO and led Apple to become the world's first trillion-dollar company in August 2018.
Apple launched the M1 chip in November 2020, transitioning Macs away from Intel processors.
    """,
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def extract_timeline(raw_text: str) -> dict:
    """
    Full end-to-end pipeline.

    Parameters
    ----------
    raw_text : user-supplied text string

    Returns
    -------
    dict with keys:
      'events'         – sorted list of event dicts
      'total_events'   – count of extracted events
      'total_sentences'– total sentences in input
      'has_dates'      – number of events with dates
      'highlighted_html' – input text with events highlighted
      'stats'          – processing statistics dict
      'error'          – error message string (or None)
    """
    if not raw_text or not raw_text.strip():
        return {"error": "Please provide some text to analyse.", "events": []}

    if len(raw_text.strip()) < 50:
        return {"error": "Text is too short. Please provide at least a few sentences.", "events": []}

    try:
        # ── Step 1: Preprocess ────────────────────────────────────────────────
        processed = preprocess(raw_text)
        sentences = processed["sentences"]
        tokens    = processed["tokens"]

        if len(sentences) < 2:
            return {"error": "Could not segment text into sentences. Please check the input.", "events": []}

        # ── Step 2: NER ───────────────────────────────────────────────────────
        ner_results = run_ner(sentences)

        # ── Step 3: Event extraction ──────────────────────────────────────────
        events = extract_events(ner_results, tokens)

        if not events:
            return {
                "error"  : None,
                "events" : [],
                "warning": "No events were detected. Try a text with more action verbs or date references.",
                "total_sentences": len(sentences),
            }

        # ── Step 4: Generate highlighted HTML ─────────────────────────────────
        highlighted = highlight_events_in_text(processed["cleaned"], events)

        # ── Step 5: Statistics ────────────────────────────────────────────────
        scores      = [e["event_score"] for e in events]
        dated_count = sum(1 for e in events if e["datetime"] is not None)

        stats = {
            "total_sentences" : len(sentences),
            "events_extracted": len(events),
            "events_with_dates": dated_count,
            "avg_confidence"  : round(sum(scores) / len(scores), 3),
            "max_confidence"  : round(max(scores), 3),
            "min_confidence"  : round(min(scores), 3),
        }

        return {
            "error"            : None,
            "events"           : events,
            "total_events"     : len(events),
            "total_sentences"  : len(sentences),
            "has_dates"        : dated_count,
            "highlighted_html" : highlighted,
            "stats"            : stats,
        }

    except Exception as exc:
        return {"error": f"Processing error: {str(exc)}", "events": []}


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def events_to_csv(events: list[dict]) -> str:
    """Convert event list to a CSV string for download."""
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["event_number", "date_text", "description",
                    "event_score", "date_conf", "sentence"],
        extrasaction="ignore",
    )
    writer.writeheader()
    for e in events:
        writer.writerow({
            "event_number": e.get("event_number", ""),
            "date_text"   : e.get("date_text", "Unknown"),
            "description" : e.get("description", ""),
            "event_score" : round(e.get("event_score", 0), 4),
            "date_conf"   : round(e.get("date_conf", 0), 4),
            "sentence"    : e.get("sentence", ""),
        })
    return output.getvalue()


def events_to_json(events: list[dict]) -> str:
    """Convert event list to a pretty-printed JSON string for download."""
    clean = []
    for e in events:
        clean.append({
            "event_number": e.get("event_number"),
            "date_text"   : e.get("date_text", "Unknown"),
            "description" : e.get("description"),
            "event_score" : round(e.get("event_score", 0), 4),
            "date_confidence": round(e.get("date_conf", 0), 4),
            "sentence"    : e.get("sentence"),
            "entities"    : e.get("entities", {}),
        })
    return json.dumps(clean, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for name, text in SAMPLE_TEXTS.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        result = extract_timeline(text)
        if result.get("error"):
            print(f"  ERROR: {result['error']}")
            continue
        stats = result["stats"]
        print(f"  Sentences: {stats['total_sentences']}  "
              f"Events: {stats['events_extracted']}  "
              f"Dated: {stats['events_with_dates']}")
        print()
        for e in result["events"]:
            print(f"  Event {e['event_number']:2d} | {e['date_text']:25s} | "
                  f"Score={e['event_score']:.2f} | {e['description'][:50]}")
