"""The portal's design system: indigo brand, tinted canvas, white rounded cards, soft status pills.

Theme colours and the font live in .streamlit/config.toml; keep the constants below in sync with it.
Cards: any `st.container(key="card_<name>")` is a white card; `card_brand_<name>` is a filled gradient card.
Cards placed side by side in st.columns stretch to the same height.
"""
import html
import re

import streamlit as st

INK, MUTED, FAINT = "#14142b", "#64677d", "#9a9cb0"
CANVAS, CARD, LINE = "#f4f5fb", "#ffffff", "#e4e6f0"
BRAND, BRAND_DARK, BRAND_SOFT, VIOLET = "#4f46e5", "#3730a3", "#eef0ff", "#7c3aed"
GRADIENT = "linear-gradient(135deg, #4f46e5 0%, #6d4aea 50%, #8b5cf6 100%)"

VERDICT = {
    "SUPPORTS": {"word": "Looks true", "why": "The sources back this up.", "tone": "green"},
    "REFUTES": {"word": "Looks false", "why": "The sources say otherwise.", "tone": "red"},
    "NOT ENOUGH INFO": {"word": "Can't tell", "why": "Nothing we found settles it.", "tone": "amber"},
}
TONE = {  # (text, soft background, solid bar)
    "green": ("#15803d", "#e6f6ec", "#22c55e"),
    "red": ("#c2261d", "#fdeceb", "#ef4444"),
    "amber": ("#a15c07", "#fef3d7", "#f59e0b"),
    "brand": (BRAND, BRAND_SOFT, BRAND),
    "violet": (VIOLET, "#f3edff", "#8b5cf6"),
    "gray": ("#4b4e63", "#f0f1f6", FAINT),
}
RELATION = {"entailment": ("Backs it up", "green"), "contradiction": ("Contradicts it", "red"),
            "neutral": ("Doesn't settle it", "gray")}

e = html.escape

CSS = f"""
/* ---------- page ---------- */
[data-testid="stMainBlockContainer"] {{ max-width: 1360px; padding: 1.6rem 1.75rem 3.5rem; }}
[data-testid="stMain"] [data-testid="stVerticalBlock"] {{ gap: 1rem; }}
[data-testid="stDecoration"] {{ display: none; }}
[data-testid="stHeader"] {{ background: transparent; }}
h1 {{ font-size: 2rem !important; line-height: 1.15 !important; letter-spacing: -0.03em; padding: 0 !important; }}
h2 {{ font-size: 1.3rem !important; letter-spacing: -0.02em; padding: .6rem 0 0 !important; }}

/* ---------- equal-height cards in a row ---------- */
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [class*="st-key-card_"]) {{
    align-items: stretch !important; }}
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] {{ height: 100%; }}
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:last-child:has(> [class*="st-key-card_"]) {{
    flex: 1 1 auto; }}
[data-testid="stLayoutWrapper"] > [class*="st-key-card_"] {{ height: 100%; }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{ border-right: 1px solid {LINE}; width: 240px !important; min-width: 240px !important; }}
[data-testid="stSidebarHeader"] {{ padding: 1.1rem 1rem .5rem; }}
[data-testid="stSidebarNavItems"] {{ padding: .3rem .75rem; }}
[data-testid="stSidebarNavLink"] {{ border-radius: 10px !important; padding: 9px 12px !important; margin: 2px 0; }}
[data-testid="stSidebarNavLink"]:hover {{ background: {CANVAS} !important; }}
[data-testid="stSidebarNavLink"] span {{ font-size: .92rem; font-weight: 500; color: {MUTED}; }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: {BRAND_SOFT} !important; }}
[data-testid="stSidebarNavLink"][aria-current="page"] span {{ color: {BRAND}; font-weight: 600; }}
[data-testid="stSidebarUserContent"] {{ padding: .5rem .75rem 1rem; }}

/* ---------- cards ---------- */
div[class*="st-key-card_"] {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 18px; padding: 24px 26px;
    box-shadow: 0 1px 2px rgba(20,20,43,.04); }}
div[class*="st-key-card_brand"] {{ background: radial-gradient(90% 70% at 100% 100%, rgba(56,189,248,.45) 0%, rgba(56,189,248,0) 60%), {GRADIENT}; border: none; color: #fff;
    box-shadow: 0 10px 30px rgba(79,70,229,.25); }}
div[class*="st-key-card_brand"] .label {{ color: rgba(255,255,255,.85); }}
div[class*="st-key-card_brand"] .big, div[class*="st-key-card_brand"] .stMarkdown {{ color: #fff; }}
div[class*="st-key-card_"] [data-testid="stExpander"] details {{ border-radius: 12px; }}
div[class*="st-key-card_hits"] button {{ justify-content: flex-start; text-align: left; }}
div[class*="st-key-card_hits"] button > div {{ justify-content: flex-start; }}
div[class*="st-key-card_hits"] button p {{ text-align: left; }}

div.st-key-hitlist {{ gap: .25rem !important; }}

/* ---------- page header ---------- */
.eyebrow {{ display: inline-flex; align-items: center; gap: 6px; font-size: .8rem; font-weight: 600; color: {BRAND};
    background: {BRAND_SOFT}; padding: 4px 11px; border-radius: 999px; }}
.stMarkdown p.lede {{ font-size: .98rem !important; line-height: 1.55; color: {MUTED}; max-width: 70ch; margin: .35rem 0 .6rem; }}

/* ---------- text helpers ---------- */
.label {{ font-size: .9rem; font-weight: 600; color: {INK}; letter-spacing: -0.005em; margin-bottom: .45rem; }}
.muted {{ color: {MUTED}; }}
.small {{ font-size: .87rem; }}
.big {{ font-size: 1.25rem; line-height: 1.4; font-weight: 600; letter-spacing: -0.02em; color: {INK}; margin: .1rem 0 .9rem; }}
.claim {{ font-size: 1.25rem; line-height: 1.4; font-weight: 600; letter-spacing: -0.02em; color: {INK}; margin: .6rem 0 .25rem; }}

/* ---------- pills ---------- */
.pill {{ display: inline-flex; align-items: center; padding: 5px 12px; border-radius: 999px;
    font-size: .8rem; font-weight: 600; line-height: 1.3; white-space: nowrap; }}

.ico {{ font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; font-size: 20px; line-height: 1;
    letter-spacing: normal; text-transform: none; display: inline-block; white-space: nowrap; direction: ltr;
    font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased; vertical-align: middle; }}

/* ---------- evidence rows ---------- */
.ev {{ display: flex; gap: 14px; padding: 14px 16px; border: 1px solid {LINE}; border-radius: 14px; background: {CARD};
    margin-bottom: 10px; }}
.ev .badge {{ flex: 0 0 32px; height: 32px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: .9rem; }}
.ev .txt {{ font-size: .92rem; line-height: 1.5; color: {INK}; }}
.ev .meta {{ margin-top: 6px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: .79rem; color: {MUTED}; }}
.ev .meta .pill {{ font-size: .74rem; padding: 3px 10px; }}

/* ---------- bars ---------- */
.bar {{ height: 8px; border-radius: 999px; background: #eceef6; overflow: hidden; }}
.bar > div {{ height: 100%; border-radius: 999px; }}
.bar-row {{ display: flex; justify-content: space-between; font-size: .82rem; color: {MUTED}; margin: 13px 0 6px; }}
.bar-row b {{ color: {INK}; font-weight: 600; }}

/* ---------- stats ---------- */
.stats {{ display: flex; flex-wrap: wrap; row-gap: 14px; }}
.stats .s {{ flex: 1 1 120px; padding: 2px 18px; border-left: 1px solid {LINE}; }}
.stats .s:first-child {{ border-left: none; padding-left: 0; }}
.stats .n {{ font-size: 1.9rem; font-weight: 700; letter-spacing: -0.04em; line-height: 1.15; }}
.stats .l {{ font-size: .83rem; color: {MUTED}; line-height: 1.35; margin-top: 2px; max-width: 24ch; }}
.stat-tile {{ display: flex; gap: 12px; align-items: center; }}
.stat-tile .badge {{ flex: 0 0 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center; }}
.stat-tile .n {{ font-size: 1.5rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1.1; }}
.stat-tile .l {{ font-size: .8rem; color: {MUTED}; }}

/* ---------- summary (inside brand card) ---------- */
.sum-row {{ display: flex; justify-content: space-between; align-items: center; padding: 13px 16px; margin-top: 10px;
    background: rgba(255,255,255,.14); border-radius: 12px; }}
.sum-row .w {{ display: flex; align-items: center; gap: 8px; font-weight: 600; font-size: .92rem; color: #fff; }}
.sum-row .w .ico {{ font-size: 19px; }}
.sum-row .n {{ font-size: 1.3rem; font-weight: 700; color: #fff; }}

/* ---------- annotated input ---------- */
.annot {{ font-size: 1.02rem; line-height: 2.05; color: {INK}; }}
.annot .skip {{ color: {FAINT}; }}
.annot mark {{ padding: 3px 6px; border-radius: 8px; color: {INK}; box-decoration-break: clone; -webkit-box-decoration-break: clone; }}
.annot mark sup {{ font-size: .64rem; font-weight: 700; margin-left: 4px; padding: 1px 5px; border-radius: 6px;
    background: {INK}; color: #fff; vertical-align: 2px; }}

/* ---------- how-it-works card ---------- */
.how {{ color: #fff; }}
.how .h {{ font-size: 1.05rem; font-weight: 700; letter-spacing: -0.01em; }}
.how .sub {{ font-size: .84rem; color: rgba(255,255,255,.75); margin: 2px 0 14px; }}
.how .st {{ display: flex; gap: 12px; align-items: center; background: rgba(255,255,255,.95); color: {INK};
    border-radius: 14px; padding: 14px 16px; margin-top: 10px; box-shadow: 0 8px 22px rgba(20,20,60,.15); }}
.how .st .badge {{ flex: 0 0 34px; height: 34px; border-radius: 10px; display: flex; align-items: center; justify-content: center; }}
.how .st .t {{ font-size: .88rem; font-weight: 600; }}
.how .st .s {{ font-size: .76rem; color: {MUTED}; margin-top: 1px; }}

/* ---------- walkthrough steps ---------- */
.step-n {{ width: 30px; height: 30px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center;
    font-size: .85rem; font-weight: 700; }}
.step-title {{ font-size: 1rem; font-weight: 600; margin: 12px 0 2px; letter-spacing: -0.01em; }}
.step-body {{ font-size: .84rem; color: {MUTED}; line-height: 1.45; min-height: 3.7em; }}
.step-metric {{ font-size: 1.6rem; font-weight: 700; letter-spacing: -0.03em; margin-top: 10px; }}
.step-metric small {{ font-size: .84rem; color: {MUTED}; font-weight: 500; letter-spacing: 0; }}

/* ---------- lists / article / empty ---------- */
.rowlist .r {{ display: flex; gap: 12px; align-items: flex-start; padding: 12px 0; border-top: 1px solid #eff0f6; }}
.rowlist .r:first-child {{ border-top: none; }}
.rowlist .r .main {{ flex: 1; font-size: .92rem; line-height: 1.45; }}
.rowlist .r .side {{ flex: 0 0 auto; }}
.article {{ font-size: .97rem; line-height: 1.75; color: {INK}; }}
.article .ln {{ font-size: .66rem; color: {FAINT}; margin-right: 4px; vertical-align: 2px; font-weight: 600; }}
.empty {{ display: flex; gap: 14px; align-items: flex-start; }}
.empty .badge {{ flex: 0 0 40px; height: 40px; border-radius: 12px; background: {BRAND_SOFT}; display: flex; align-items: center;
    justify-content: center; color: {BRAND}; }}

/* ---------- sidebar info card ---------- */
.side-card {{ border-radius: 14px; padding: 16px 18px; background: {BRAND_SOFT}; font-size: .8rem; color: {MUTED}; line-height: 1.5; }}
.side-card b {{ color: {INK}; }}
"""


def inject_css():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def html_block(markup):
    st.markdown(markup, unsafe_allow_html=True)


# ---------------------------------------------------------------- building blocks

def card(name):
    return st.container(key=f"card_{name}")


def icon(name, color=None, size=20):
    style = f"font-size:{size}px;" + (f"color:{color};" if color else "")
    return f"<span class='ico' style='{style}'>{name}</span>"


def pill(text, tone="gray", dot=None):
    """Soft coloured label. `dot` is accepted for old call sites and ignored: pills have no dot."""
    fg, bg, _ = TONE[tone]
    return f"<span class='pill' style='color:{fg};background:{bg}'>{e(text)}</span>"


def verdict_pill(label):
    v = VERDICT[label]
    return pill(v["word"], v["tone"])


def bar(value, tone="brand", label=None, right=None):
    _, _, solid = TONE[tone]
    head = f"<div class='bar-row'><span>{label}</span><b>{right}</b></div>" if label else ""
    return f"{head}<div class='bar'><div style='width:{max(0, min(1, value)) * 100:.1f}%;background:{solid}'></div></div>"


def stats(items):
    """items: [(number, label)] or [(number, label, tone)] in a row with thin dividers."""
    cells = ""
    for it in items:
        n, l = it[0], it[1]
        color = TONE[it[2]][0] if len(it) > 2 else INK
        cells += f"<div class='s'><div class='n' style='color:{color}'>{e(str(n))}</div><div class='l'>{e(l)}</div></div>"
    html_block(f"<div class='stats'>{cells}</div>")


def stat_tile(number, text, ico, tone="brand"):
    fg, bg, _ = TONE[tone]
    return (f"<div class='stat-tile'><div class='badge' style='background:{bg};color:{fg}'>{icon(ico, size=22)}</div>"
            f"<div><div class='n'>{e(str(number))}</div><div class='l'>{e(text)}</div></div></div>")


def label(text):
    html_block(f"<div class='label'>{e(text)}</div>")


def page_header(eyebrow, title, lede_text):
    html_block(f"<span class='eyebrow'>{e(eyebrow)}</span>")
    st.title(title, anchor=False)
    html_block(f"<p class='lede'>{lede_text}</p>")


def empty_state(title, body, command=None, ico="hourglass_empty"):
    html_block(f"<div class='empty'><div class='badge'>{icon(ico)}</div><div><div style='font-weight:600'>{e(title)}</div>"
               f"<div class='muted small' style='margin-top:2px'>{body}</div></div></div>")
    if command:
        st.code(command, language="bash")


# ---------------------------------------------------------------- language helpers

def pct(x, digits=0):
    return f"{x * 100:.{digits}f}%"


def sureness(p):
    return "Very sure" if p >= 0.9 else "Fairly sure" if p >= 0.7 else "Not very sure"


def tidy(text):
    """FEVER text is tokenized ("New Zealand 's , ( 1861 )"); make it read like normal prose."""
    from src.task2_evidence_retrieval.data import FeverText
    t = FeverText.clean(text or "")
    t = re.sub(r"\s+([,.;:!?%)\]}]|'s\b|'re\b|n't\b|'ve\b|'d\b|'ll\b)", r"\1", t)
    t = re.sub(r"([(\[{$])\s+", r"\1", t)
    t = re.sub(r'"\s+([^"]*?)\s+"', r'"\1"', t)
    return t.replace(" -- ", " – ")


def title_of(title):
    from src.task2_evidence_retrieval.data import FeverText
    return FeverText.clean(title or "")


# ---------------------------------------------------------------- composite pieces

def evidence_item(ev, gold=False):
    nli = ev.get("nli")
    tone, rel = "gray", ""
    if nli:
        word, tone = RELATION[max(nli, key=nli.get)]
        rel = pill(word, tone)
    fg, bg, _ = TONE[tone]
    star = pill("The answer", "brand", dot=False) if gold else ""
    return (f"<div class='ev'><div class='badge' style='background:{bg};color:{fg}'>W</div><div>"
            f"<div class='txt'>{e(tidy(ev.get('text', '')))}</div>"
            f"<div class='meta'>{rel}{star}<span>Wikipedia · {e(title_of(ev.get('title')))}</span></div></div></div>")


def evidence_list(evidence, gold_keys=None):
    gold_keys = gold_keys or set()
    html_block("".join(evidence_item(ev, (ev.get("page_id"), ev.get("line_id")) in gold_keys) for ev in evidence))


def decisive_order(label, evidence):
    if label == "SUPPORTS":
        return sorted(evidence, key=lambda ev: -ev["nli"]["entailment"])
    if label == "REFUTES":
        return sorted(evidence, key=lambda ev: -ev["nli"]["contradiction"])
    return list(evidence)


def annotated_text(sentences, threshold, verdict_by_index=None):
    verdict_by_index = verdict_by_index or {}
    parts, n = [], 0
    for s in sentences:
        tip = f"Sounds factual: {pct(s['p_claim'])}"
        if s["p_claim"] >= threshold:
            n += 1
            tone = VERDICT.get(verdict_by_index.get(s["index"]), {}).get("tone", "brand")
            parts.append(f"<mark title='{tip}' style='background:{TONE[tone][1]}'>{e(s['sentence'])}<sup>{n}</sup></mark>")
        else:
            parts.append(f"<span class='skip' title='{tip}'>{e(s['sentence'])}</span>")
    html_block(f"<div class='annot'>{' '.join(parts)}</div>")


def how_it_works():
    steps = [("format_quote", "brand", "1 · Spot the claims", "Keeps factual sentences, skips small talk"),
             ("travel_explore", "violet", "2 · Find evidence", "Searches 31,000 Wikipedia pages"),
             ("gavel", "green", "3 · Decide", "True, false, or can't tell, with sources")]
    rows = "".join(f"<div class='st'><div class='badge' style='background:{TONE[t][1]};color:{TONE[t][0]}'>"
                   f"{icon(i, size=19)}</div><div><div class='t'>{a}</div><div class='s'>{b}</div></div></div>"
                   for i, t, a, b in steps)
    html_block(f"<div class='how'><div class='h'>How it works</div>"
               f"<div class='sub'>Three models, one after another.</div>{rows}</div>")
    # render inside ui.card("brand_how") so it gets the gradient and matches its neighbour's height


def summary_rows(counts):
    icons = {"SUPPORTS": ("check_circle", "#86efac"), "REFUTES": ("cancel", "#fca5a5"), "NOT ENOUGH INFO": ("help", "#fde68a")}
    return "".join(f"<div class='sum-row'><div class='w'>{icon(*icons[l])}{VERDICT[l]['word']}</div>"
                   f"<div class='n'>{n}</div></div>" for l, n in counts.items())


def evidence_table(evidence, gold_keys=None):
    """Developer view with every score."""
    import pandas as pd
    rows = []
    for i, ev in enumerate(evidence, 1):
        nli = ev.get("nli") or {}
        row = {"rank": i, "page": title_of(ev.get("title")), "sentence": tidy(ev.get("text", ""))}
        if gold_keys is not None:
            row["answer"] = "✓" if (ev.get("page_id"), ev.get("line_id")) in gold_keys else ""
        if nli:
            row.update({"agrees": nli["entailment"], "neutral": nli["neutral"], "disagrees": nli["contradiction"]})
        row.update({"rerank": ev.get("rerank_score"), "meaning": ev.get("retrieval_score"), "keyword": ev.get("doc_score")})
        rows.append(row)
    if not rows:
        return
    prog = lambda n: st.column_config.ProgressColumn(n, min_value=0.0, max_value=1.0, format="%.2f")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "agrees": prog("agrees"), "neutral": prog("neutral"), "disagrees": prog("disagrees"),
        "sentence": st.column_config.TextColumn(width="large"),
        "rerank": st.column_config.NumberColumn(format="%.2f"),
        "meaning": st.column_config.NumberColumn(format="%.2f"),
        "keyword": st.column_config.NumberColumn(format="%.1f")})


def chart_theme(chart):
    return (chart.configure(background="transparent", font="Inter")
            .configure_view(strokeWidth=0)
            .configure_axisX(grid=False)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#eef0f6", domainColor=LINE, tickColor=LINE,
                            labelFontSize=12, titleFontSize=12, titleFontWeight=500))
