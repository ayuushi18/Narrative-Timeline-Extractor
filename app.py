"""
=============================================================================
 app.py  –  Narrative Timeline Extractor  |  Streamlit Dashboard
=============================================================================
 Run with:   streamlit run app.py
 Requires:   pip install streamlit pandas matplotlib
=============================================================================
"""

import io
import math
import textwrap
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import streamlit as st

from timeline_engine import extract_timeline, events_to_csv, events_to_json, SAMPLE_TEXTS


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Narrative Timeline Extractor",
    page_icon  = "🕰️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — editorial serif aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Source+Sans+3:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

  /* ── Root variables ── */
  :root {
    --cream:   #FAF7F2;
    --warm:    #F0EAE0;
    --sand:    #D6C9B4;
    --walnut:  #7A6652;
    --espresso:#2C1F14;
    --ink:     #1A1208;
    --rust:    #C05A2A;
    --teal:    #2A7A6E;
    --gold:    #C49A2A;
    --blue:    #2A4A7A;
  }

  /* ── Global ── */
  html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background: var(--cream);
  }

  /* ── Hide default streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Headings ── */
  h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
  }

  /* ── Main header ── */
  .page-header {
    background: var(--espresso);
    color: #FAF7F2;
    padding: 2.4rem 2.8rem 2rem;
    border-radius: 6px;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
  }
  .page-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(192,90,42,0.2);
  }
  .page-header h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.2rem; font-weight: 700;
    margin-bottom: .3rem; color: #FAF7F2;
  }
  .page-header p {
    font-size: .95rem; color: rgba(250,247,242,0.75);
    font-weight: 300; margin:0;
  }

  /* ── Metric cards ── */
  .metric-row { display:flex; gap:14px; margin-bottom:1.4rem; flex-wrap:wrap; }
  .metric-card {
    background: white; border:1px solid var(--sand);
    border-radius:8px; padding:16px 20px; flex:1; min-width:120px;
    border-top: 3px solid var(--rust);
  }
  .metric-label { font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; color:var(--walnut); margin-bottom:4px; }
  .metric-value { font-family:'Playfair Display',serif; font-size:2rem; color:var(--espresso); font-weight:700; line-height:1; }
  .metric-sub   { font-size:.72rem; color:var(--walnut); margin-top:3px; }

  /* ── Event timeline items ── */
  .timeline-item {
    display: flex; gap: 18px; margin-bottom: 1.2rem;
    animation: slideIn .4s ease both;
  }
  @keyframes slideIn {
    from { opacity:0; transform:translateX(-12px); }
    to   { opacity:1; transform:translateX(0); }
  }
  .timeline-num {
    min-width: 40px; height:40px; border-radius:50%;
    background: var(--espresso); color: #FAF7F2;
    font-family: 'Playfair Display', serif;
    font-size: 1rem; font-weight:700;
    display:flex; align-items:center; justify-content:center;
    flex-shrink:0; margin-top:2px;
  }
  .timeline-body {
    flex:1;
    border-left: 2px solid var(--sand); padding-left:14px;
  }
  .timeline-date {
    font-family: 'JetBrains Mono', monospace;
    font-size:.72rem; color:var(--rust); font-weight:500;
    text-transform:uppercase; letter-spacing:.08em; margin-bottom:4px;
  }
  .timeline-desc {
    font-size:1rem; color:var(--espresso); font-weight:500;
    margin-bottom:4px; line-height:1.4;
  }
  .timeline-sent {
    font-size:.84rem; color:var(--walnut); font-weight:300;
    line-height:1.55;
  }

  /* ── Score badge ── */
  .score-badge {
    display:inline-block;
    padding:2px 8px; border-radius:999px;
    font-family:'JetBrains Mono',monospace; font-size:.65rem;
    font-weight:500; margin-left:8px; vertical-align:middle;
  }
  .score-high { background:#D4EDDA; color:#155724; }
  .score-mid  { background:#FFF3CD; color:#856404; }
  .score-low  { background:#F8D7DA; color:#721C24; }

  /* ── Entity chips ── */
  .entity-chips { display:flex; flex-wrap:wrap; gap:5px; margin-top:6px; }
  .chip {
    font-size:.67rem; padding:2px 7px; border-radius:4px; font-weight:500;
  }
  .chip-person { background:#EEF2FF; color:#3730A3; border:1px solid #C7D2FE; }
  .chip-org    { background:#F0FDF4; color:#166534; border:1px solid #BBF7D0; }
  .chip-loc    { background:#FFF7ED; color:#9A3412; border:1px solid #FED7AA; }

  /* ── Highlight text ── */
  mark {
    background: rgba(192,90,42,.15); border-radius:3px;
    padding:1px 3px; color:var(--espresso);
  }

  /* ── Table ── */
  .event-table th { background:var(--espresso) !important; color:#FAF7F2 !important; }
  .event-table tr:nth-child(even) { background:var(--warm); }

  /* ── Section divider ── */
  .section-rule {
    border:none; border-top:1px solid var(--sand);
    margin:1.8rem 0 1.4rem;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: var(--espresso) !important;
  }
  [data-testid="stSidebar"] * { color: #FAF7F2 !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label { color: rgba(250,247,242,.75) !important; }

  /* ── Buttons ── */
  .stButton > button {
    background: var(--rust) !important; color: #FAF7F2 !important;
    border: none !important; border-radius:5px !important;
    font-weight:600 !important; letter-spacing:.04em !important;
    padding:.55rem 1.8rem !important; font-size:.9rem !important;
    transition: background .2s !important;
  }
  .stButton > button:hover { background: #A04822 !important; }

  /* ── Download buttons ── */
  .stDownloadButton > button {
    background: transparent !important; color: var(--walnut) !important;
    border: 1px solid var(--sand) !important; border-radius:5px !important;
    font-size:.82rem !important;
  }
  .stDownloadButton > button:hover {
    border-color: var(--walnut) !important; color:var(--espresso) !important;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def score_class(score: float) -> str:
    if score >= 0.55:
        return "score-high"
    elif score >= 0.35:
        return "score-mid"
    return "score-low"


def render_timeline_item(event: dict, delay: float = 0.0) -> str:
    n     = event["event_number"]
    date  = event["date_text"]
    desc  = event["description"]
    sent  = event["sentence"]
    score = event["event_score"]
    ents  = event.get("entities", {})

    sc  = score_class(score)
    badge = f'<span class="score-badge {sc}">score {score:.2f}</span>'

    # Entity chips
    chips = ""
    for p in ents.get("persons", []):
        chips += f'<span class="chip chip-person">👤 {p}</span>'
    for o in ents.get("orgs", []):
        chips += f'<span class="chip chip-org">🏢 {o[:30]}</span>'
    for l in ents.get("locations", []):
        chips += f'<span class="chip chip-loc">📍 {l}</span>'
    chips_html = f'<div class="entity-chips">{chips}</div>' if chips else ""

    return f"""
    <div class="timeline-item" style="animation-delay:{delay:.2f}s">
      <div class="timeline-num">{n}</div>
      <div class="timeline-body">
        <div class="timeline-date">🗓 {date}</div>
        <div class="timeline-desc">{desc}{badge}</div>
        <div class="timeline-sent">{sent}</div>
        {chips_html}
      </div>
    </div>
    """


def build_timeline_chart(events: list[dict]) -> plt.Figure:
    """
    Create a horizontal Gantt-style timeline visualisation.
    Each event is a labelled dot on a horizontal time axis.
    """
    dated = [e for e in events if e.get("datetime")]
    undated = [e for e in events if not e.get("datetime")]

    fig, ax = plt.subplots(figsize=(12, max(3, len(events) * 0.55 + 1.5)))
    fig.patch.set_facecolor("#FAF7F2")
    ax.set_facecolor("#FAF7F2")

    colors = ["#C05A2A","#2A7A6E","#2A4A7A","#C49A2A","#7A6652","#6B2A7A"]

    if dated:
        # Sort by datetime for x-axis
        dated_sorted = sorted(dated, key=lambda e: e["datetime"])
        dt_vals = [e["datetime"] for e in dated_sorted]
        dt_nums = matplotlib.dates.date2num(dt_vals)

        min_x = dt_nums[0]  - (dt_nums[-1] - dt_nums[0]) * 0.05
        max_x = dt_nums[-1] + (dt_nums[-1] - dt_nums[0]) * 0.05
        if min_x == max_x:
            min_x -= 30; max_x += 30

        ax.set_xlim(min_x, max_x)

        # Baseline
        ax.axhline(0.5, color="#D6C9B4", linewidth=1.5, zorder=1)

        for i, (e, xval) in enumerate(zip(dated_sorted, dt_nums)):
            color = colors[i % len(colors)]
            y_offset = 0.5 + (0.3 if i % 2 == 0 else -0.3)

            # Connector line
            ax.plot([xval, xval], [0.5, y_offset], color=color, lw=1, alpha=.5, zorder=2)
            # Dot
            ax.scatter([xval], [y_offset], s=90, color=color, zorder=4, edgecolors="white", linewidth=1.5)

            # Label
            label = f"E{e['event_number']}: " + textwrap.shorten(e["description"], width=28, placeholder="…")
            va = "bottom" if y_offset > 0.5 else "top"
            ax.annotate(
                label, (xval, y_offset),
                xytext=(0, 6 if va == "bottom" else -6),
                textcoords="offset points",
                ha="center", va=va, fontsize=6.5,
                color="#2C1F14",
                fontfamily="serif",
            )

        ax.xaxis_date()
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
        plt.xticks(rotation=30, ha="right", fontsize=8)
    else:
        # No dates: just plot event numbers in sequence
        ax.axhline(0.5, color="#D6C9B4", linewidth=1.5, zorder=1)
        n = len(events)
        for i, e in enumerate(events):
            x = i / max(n - 1, 1)
            ax.scatter([x], [0.5], s=90, color=colors[i % len(colors)],
                       zorder=4, edgecolors="white", linewidth=1.5)
            label = f"E{e['event_number']}"
            ax.annotate(label, (x, 0.5), xytext=(0, 10), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color="#2C1F14")
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("Document order (no dates extracted)", fontsize=9)

    # Undated note
    if undated:
        ax.text(
            0.99, 0.02,
            f"+ {len(undated)} event(s) without dates not shown on axis",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="#7A6652", style="italic",
        )

    ax.set_yticks([])
    ax.set_ylim(0, 1)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#D6C9B4")

    ax.set_title("Chronological Event Timeline", fontfamily="serif",
                 fontsize=12, color="#2C1F14", pad=10)
    plt.tight_layout()
    return fig


def build_score_chart(events: list[dict]) -> plt.Figure:
    """Horizontal bar chart of event confidence scores."""
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(events) * 0.42 + 0.8)))
    fig.patch.set_facecolor("#FAF7F2")
    ax.set_facecolor("#FAF7F2")

    labels = [f"E{e['event_number']}: " + textwrap.shorten(e["description"], 30, placeholder="…") for e in events]
    scores = [e["event_score"] for e in events]
    colors_list = ["#C05A2A" if s >= 0.55 else "#C49A2A" if s >= 0.35 else "#D6C9B4" for s in scores]

    bars = ax.barh(labels[::-1], scores[::-1], color=colors_list[::-1],
                   edgecolor="white", linewidth=.5, height=.65)
    ax.bar_label(bars, labels=[f"{s:.2f}" for s in scores[::-1]],
                 padding=4, fontsize=7.5, color="#2C1F14",
                 fontfamily="monospace")

    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Event confidence score", fontsize=9)
    ax.axvline(0.55, color="#C05A2A", linewidth=1, linestyle="--", alpha=.5)
    ax.axvline(0.35, color="#C49A2A", linewidth=1, linestyle="--", alpha=.5)
    ax.tick_params(axis="y", labelsize=7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    legend_elements = [
        mpatches.Patch(color="#C05A2A", label="High (≥0.55)"),
        mpatches.Patch(color="#C49A2A", label="Medium (≥0.35)"),
        mpatches.Patch(color="#D6C9B4", label="Low (<0.35)"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")
    ax.set_title("Event Confidence Scores", fontfamily="serif", fontsize=11, color="#2C1F14")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🕰️ Timeline Extractor")
    st.markdown("---")

    st.markdown("**Try a Sample**")
    sample_key = st.selectbox(
        "Load example text",
        ["— none —"] + list(SAMPLE_TEXTS.keys()),
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Settings**")
    min_score = st.slider("Min event score", 0.0, 0.8, 0.18, 0.01,
                          help="Events below this score are filtered out.")
    show_sent = st.checkbox("Show full sentences", value=True)
    show_ents = st.checkbox("Show named entities", value=True)

    st.markdown("---")
    st.markdown("**Architecture**")
    st.markdown("""
```
Input Text
    ↓
preprocessor.py
  • Clean & segment
  • Tokenise
    ↓
ner_extractor.py
  • Date / time NER
  • Person / Org / Loc
    ↓
event_extractor.py
  • TF-IDF scoring
  • Action verb signals
  • Composite ranking
    ↓
timeline_engine.py
  • Temporal sort
  • Highlight HTML
    ↓
Streamlit Dashboard
```
    """)
    st.markdown("---")
    st.caption("Final-Year AI/ML Project\nNarrative NLP · Python 3.10+")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="page-header">
  <h1>🕰️ Narrative Timeline Extractor</h1>
  <p>Paste any news article, story, or case study and extract a structured chronological timeline of events using NLP.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────────────────────────────────────

col_in, col_tip = st.columns([3, 1])

with col_in:
    # Populate from sample selector
    default_text = ""
    if sample_key != "— none —":
        default_text = SAMPLE_TEXTS[sample_key].strip()

    input_text = st.text_area(
        "✍️ Input Text",
        value=default_text,
        height=230,
        placeholder=(
            "Paste your news article, story, or case study here…\n\n"
            "Example:\n"
            "On January 2022, the company launched its product. "
            "By March 2022, sales had increased by 40%.\n"
            "In June 2022, they expanded to global markets."
        ),
        label_visibility="visible",
    )

with col_tip:
    st.markdown("#### 💡 Tips")
    st.markdown("""
- Works best with **news articles**, biographies, or case studies
- Include **dates** for better timeline ordering
- Minimum **3–4 sentences** recommended
- Action verbs like *launched*, *announced*, *expanded* help detection
    """)

btn_col, clear_col, _ = st.columns([1, 1, 4])
with btn_col:
    run = st.button("🔍 Extract Timeline", use_container_width=True)
with clear_col:
    if st.button("🗑 Clear", use_container_width=True):
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING & OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

if run and input_text.strip():
    with st.spinner("Analysing text and extracting timeline…"):
        result = extract_timeline(input_text)

    # Filter by sidebar minimum score
    if result.get("events"):
        result["events"] = [e for e in result["events"] if e["event_score"] >= min_score]
        for i, e in enumerate(result["events"], 1):
            e["event_number"] = i

    # ── Error handling ─────────────────────────────────────────────────────────
    if result.get("error"):
        st.error(result["error"])
        st.stop()

    if result.get("warning"):
        st.warning(result["warning"])

    events = result.get("events", [])
    if not events:
        st.info("No events passed the confidence threshold. Try lowering the Min Event Score in the sidebar.")
        st.stop()

    stats = result.get("stats", {})

    # ── KPI row ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-label">Sentences</div>
        <div class="metric-value">{stats.get('total_sentences', '—')}</div>
        <div class="metric-sub">in input text</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Events Found</div>
        <div class="metric-value">{len(events)}</div>
        <div class="metric-sub">above threshold</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">With Dates</div>
        <div class="metric-value">{sum(1 for e in events if e.get('datetime'))}</div>
        <div class="metric-sub">temporal anchors</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Avg Score</div>
        <div class="metric-value">{sum(e['event_score'] for e in events)/len(events):.2f}</div>
        <div class="metric-sub">confidence</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ── TABS ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗓 Timeline View", "📋 Table View", "📊 Visualisations", "🔍 Highlighted Text"
    ])

    # ════════════════════════════════════════════
    # TAB 1 — TIMELINE
    # ════════════════════════════════════════════
    with tab1:
        st.markdown("### Extracted Events — Chronological Order")
        st.caption(f"Sorted by detected date (undated events appear last in document order)")

        timeline_html = ""
        for i, event in enumerate(events):
            timeline_html += render_timeline_item(event, delay=i * 0.05)

        st.markdown(timeline_html, unsafe_allow_html=True)

        # Download buttons
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        dl1, dl2, _ = st.columns([1, 1, 3])
        with dl1:
            st.download_button(
                "⬇ Download CSV",
                data=events_to_csv(events),
                file_name="timeline.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "⬇ Download JSON",
                data=events_to_json(events),
                file_name="timeline.json",
                mime="application/json",
                use_container_width=True,
            )

    # ════════════════════════════════════════════
    # TAB 2 — TABLE
    # ════════════════════════════════════════════
    with tab2:
        st.markdown("### Event Table")

        rows = []
        for e in events:
            persons   = ", ".join(e["entities"].get("persons", []))
            orgs      = ", ".join(e["entities"].get("orgs", [])[:2])
            locations = ", ".join(e["entities"].get("locations", [])[:2])
            rows.append({
                "#"              : e["event_number"],
                "Date"           : e["date_text"],
                "Description"    : e["description"],
                "Score"          : round(e["event_score"], 3),
                "Date Confidence": round(e["date_conf"], 3),
                "Persons"        : persons,
                "Orgs"           : orgs,
                "Locations"      : locations,
            })
        df = pd.DataFrame(rows)

        st.dataframe(
            df,
            use_container_width=True,
            height=400,
            column_config={
                "#"              : st.column_config.NumberColumn(width="small"),
                "Score"          : st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=1, format="%.3f"),
                "Date Confidence": st.column_config.ProgressColumn(
                    "Date Conf", min_value=0, max_value=1, format="%.3f"),
            },
            hide_index=True,
        )

        st.download_button(
            "⬇ Download Table as CSV",
            data=df.to_csv(index=False),
            file_name="timeline_table.csv",
            mime="text/csv",
        )

    # ════════════════════════════════════════════
    # TAB 3 — VISUALISATIONS
    # ════════════════════════════════════════════
    with tab3:
        st.markdown("### Visual Timeline")
        fig_timeline = build_timeline_chart(events)
        st.pyplot(fig_timeline, use_container_width=True)
        plt.close(fig_timeline)

        st.markdown("### Event Confidence Scores")
        fig_scores = build_score_chart(events)
        st.pyplot(fig_scores, use_container_width=True)
        plt.close(fig_scores)

        # Date distribution bar chart
        dated = [e for e in events if e.get("datetime")]
        if dated:
            st.markdown("### Date Distribution")
            years = [e["datetime"].year for e in dated]
            year_counts = pd.Series(years).value_counts().sort_index()
            fig_yr, ax_yr = plt.subplots(figsize=(7, 2.5))
            fig_yr.patch.set_facecolor("#FAF7F2")
            ax_yr.set_facecolor("#FAF7F2")
            ax_yr.bar(year_counts.index.astype(str), year_counts.values,
                      color="#C05A2A", alpha=.8, edgecolor="white")
            ax_yr.set_xlabel("Year", fontsize=9)
            ax_yr.set_ylabel("Events", fontsize=9)
            ax_yr.set_title("Events per Year", fontfamily="serif", fontsize=11, color="#2C1F14")
            for spine in ["top","right"]:
                ax_yr.spines[spine].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_yr, use_container_width=True)
            plt.close(fig_yr)

    # ════════════════════════════════════════════
    # TAB 4 — HIGHLIGHTED TEXT
    # ════════════════════════════════════════════
    with tab4:
        st.markdown("### Original Text with Events Highlighted")
        st.caption("Event sentences are highlighted — hover or search for context.")

        highlighted = result.get("highlighted_html", input_text)
        st.markdown(
            f"""
            <div style="background:white; border:1px solid #D6C9B4; border-radius:8px;
                        padding:20px 24px; font-family:'Source Sans 3',sans-serif;
                        font-size:.92rem; line-height:1.8; color:#2C1F14;
                        max-height:500px; overflow-y:auto;">
              {highlighted}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Legend
        st.markdown("""
        <div style="margin-top:10px; font-size:.8rem; color:#7A6652;">
          <mark style="background:rgba(192,90,42,.15);padding:2px 4px;border-radius:3px">Highlighted sentences</mark>
          were identified as significant events by the NLP pipeline.
        </div>
        """, unsafe_allow_html=True)

elif run and not input_text.strip():
    st.warning("Please enter some text before clicking Extract Timeline.")
