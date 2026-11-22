import streamlit as st

import ui
from common import load_detector, load_verifier, run_evidence

# Every claim here has its answer page in the library, so the demo is right for the right reason.
EXAMPLES = {
    "A short speech": "Good evening everyone, and thank you for coming tonight. Japan is landlocked. "
                      "I honestly think pizza is the best food in the world. The American Civil War began in 1861.",
    "A false claim": "Mogadishu is a book.",
    "A TV fact": "Shadowhunters premiered in 2016.",
}


def use_example():
    choice = st.session_state.get("example")
    if choice:
        st.session_state["text"] = EXAMPLES[choice]


ui.page_header("Fact-check", "Is it true? Let's check.",
               "Paste a speech, a paragraph or one sentence. We find the statements worth checking, look them up "
               "in Wikipedia, and show you what the sources say.")

# ================================================================ input + how it works
left, right = st.columns([2, 1], gap="medium")
with left, ui.card("input"):
    ui.label("Your text")
    text = st.text_area("Your text", key="text", height=150, label_visibility="collapsed",
                        placeholder="e.g. Thank you all for coming. Indonesia has more than 17,000 islands.")
    st.pills("Try an example", list(EXAMPLES), key="example", on_change=use_example)
    b1, b2, _ = st.columns([1, 1, 2.2])
    go = b1.button("Check it", type="primary", icon=":material/arrow_forward:", disabled=not text.strip(),
                   width="stretch")
    with b2.popover("Settings", icon=":material/tune:", width="stretch"):
        whole = st.toggle("Treat it as one statement", key="whole",
                          help="Skip the step that picks out factual sentences.")
        claim_t = st.slider("How factual must a sentence sound to be checked?", 0.05, 0.95, 0.60, 0.05,
                            key="claim_t", disabled=whole, help="0.60 was tuned on held-out data.")
        nli_t = st.slider("How sure before saying true or false?", 0.05, 0.95, 0.50, 0.05, key="nli_t",
                          help="Below this, the answer is “Can't tell”.")
        top_k = st.slider("Evidence sentences per statement", 1, 10, 5, key="top_k")
with right, ui.card("brand_how"):
    ui.how_it_works()

# ================================================================ run
if go:
    slot = st.empty()
    with slot.status("Reading your text…") as status:
        sentences = None
        if whole:
            claims = [{"index": 0, "sentence": text.strip(), "p_claim": None}]
        else:
            try:
                detector = load_detector()
            except FileNotFoundError:
                status.update(label="The statement spotter isn't installed", state="error")
                ui.empty_state("Step 1 model is missing.", "Unzip <code>task1_roberta-base_label.zip</code> into the "
                               "project root, or turn on “Treat it as one statement” in Settings.", ico="error")
                st.stop()
            sentences = detector.detect(text, threshold=claim_t, return_all=True)
            claims = [s for s in sentences if s["is_claim"]]
        verifier = load_verifier() if claims else None
        results = []
        for i, c in enumerate(claims, 1):
            status.update(label=f"Looking up statement {i} of {len(claims)}…")
            v = verifier.verify(c["sentence"], run_evidence(c["sentence"], top_k=top_k), threshold=nli_t)
            results.append({"index": c["index"], "claim": c["sentence"], "p_claim": c["p_claim"], **v})
    slot.empty()
    st.session_state["check"] = {"sentences": sentences, "results": results, "claim_t": claim_t}

res = st.session_state.get("check")
if not res:
    st.stop()
sentences, results = res["sentences"], res["results"]

# ================================================================ results
st.header("Results", anchor=False)
counts = {lab: sum(r["label"] == lab for r in results) for lab in ui.VERDICT}

s_col, r_col = st.columns([1, 2], gap="medium") if sentences is not None else (st.container(), None)
with s_col, ui.card("brand_summary"):
    ui.label("Summary")
    n = len(results)
    ui.html_block(f"<div class='big' style='color:#fff;margin-bottom:4px'>{n} statement{'s' if n != 1 else ''} "
                  f"checked</div>{ui.summary_rows(counts)}")

if r_col is not None:
    with r_col, ui.card("read"):
        ui.label("What we read")
        ui.annotated_text(sentences, res["claim_t"], {r["index"]: r["label"] for r in results})
        skipped = len(sentences) - len(results)
        ui.html_block(f"<div class='small muted' style='margin-top:8px'>Highlighted = factual statements we checked. "
                      + (f"The {skipped} grey sentence{'s' if skipped != 1 else ''} "
                         f"{'are' if skipped != 1 else 'is'} opinion or small talk, so we skipped "
                         f"{'them' if skipped != 1 else 'it'}." if skipped else "Every sentence was factual.")
                      + "</div>")

if not results:
    with ui.card("none"):
        ui.empty_state("Nothing to check here.", "None of these sentences sounds like a factual statement. If you "
                       "think one does, lower “How factual must a sentence sound” in Settings.", ico="search_off")
    st.stop()

for i, r in enumerate(results, 1):
    v = ui.VERDICT[r["label"]]
    with ui.card(f"claim_{i}"):
        a, b = st.columns([1, 1.35], gap="large")
        with a:
            ui.html_block(f"<div style='display:flex;gap:8px;align-items:center'>{ui.pill(f'#{i}', 'gray', dot=False)}"
                          f"{ui.verdict_pill(r['label'])}</div>"
                          f"<div class='claim'>{ui.e(r['claim'])}</div>"
                          f"<div class='muted'>{v['why']}</div>"
                          + ui.bar(r["confidence"], v["tone"], "Confidence",
                                   f"{ui.sureness(r['confidence'])} · {ui.pct(r['confidence'])}")
                          + (ui.bar(r["p_claim"], "brand", "Sounds factual", ui.pct(r["p_claim"]))
                             if r["p_claim"] is not None else ""))
        with b:
            ui.label("What the sources say")
            ordered = ui.decisive_order(r["label"], r["evidence"])
            if not ordered:
                ui.empty_state("No sources found", "There's no page about this in the library.", ico="search_off")
            else:
                ui.evidence_list(ordered[:2])
        if r["evidence"]:
            with st.expander(f"All {len(r['evidence'])} sentences we read, with scores"):
                ui.evidence_list(ordered[2:])
                ui.evidence_table(r["evidence"])
