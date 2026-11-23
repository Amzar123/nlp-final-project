import random
import streamlit as st
import ui
from common import CLAIMS_PATH, corpus_index, load_claims, load_verifier, rel, run_evidence, sentence_text
from src.task2_evidence_retrieval.data import FeverText

ui.page_header("Step by step", "Follow a claim through the system",
               "Pick a claim where people already know the right answer. Run it and see what each of the four "
               "steps did, and exactly where things went right or wrong.")

claims = load_claims()
if not claims:
    with ui.card("missing"):
        ui.empty_state("No test claims yet.", f"<code>{rel(CLAIMS_PATH)}</code> is created when you build the library.",
                       "python -m src.task2_evidence_retrieval.build_corpus")
    st.stop()

by_id = {c["id"]: c for c in claims}
ids = list(by_id)
if st.session_state.get("claim_id") not in by_id:
    st.session_state["claim_id"] = ids[0]

def shuffle():
    st.session_state["claim_id"] = random.choice(ids)

def page_title(pid):
    page = corpus_index().get(pid)
    return page["title"] if page else FeverText.title_from_page_id(pid)

# pick + the claim
pick, show = st.columns([1, 1.7], gap="medium")
with pick, ui.card("pick"):
    ui.label("Choose a claim")
    st.selectbox("Claim", ids, key="claim_id", label_visibility="collapsed", format_func=lambda i: by_id[i]["claim"])
    c1, c2 = st.columns(2)
    c1.button("Random", icon=":material/shuffle:", on_click=shuffle, width="stretch")
    run = c2.button("Run it", type="primary", icon=":material/play_arrow:", width="stretch")
    ui.html_block("<div class='small muted' style='margin-top:10px'>Tip: type in the box to search.</div>")

claim = by_id[st.session_state["claim_id"]]
gold_sets = claim["evidence_sets"]
gold_keys = {tuple(s) for ev in gold_sets for s in ev}
gold_pages = {p for p, _ in gold_keys}

with show, ui.card("claim"):
    ui.html_block(f"<div style='display:flex;gap:8px;align-items:center'>{ui.pill('Test claim', 'gray', dot=False)}"
                  f"<span class='small muted'>Right answer</span>{ui.verdict_pill(claim['label'])}</div>"
                  f"<div class='claim' style='font-size:1.45rem'>{ui.e(claim['claim'])}</div>")
    ui.label("The sentence that proves it")
    items = []
    for page_id, line_id in gold_sets[0]:
        txt = sentence_text(page_id, line_id)
        items.append(ui.evidence_item({"text": txt, "title": page_title(page_id)}) if txt else
                     f"<div class='small' style='color:{ui.TONE['amber'][0]}'>This sentence ({ui.e(page_id)}, line "
                     f"{line_id}) isn't in the library, so the system can't find it.</div>")
    ui.html_block("".join(items))


# run
def full_set_in(keys):
    keys = set(keys)
    return any(all(tuple(s) in keys for s in ev) for ev in gold_sets)

def gold_rank(evidence):
    return next((i for i, ev in enumerate(evidence, 1) if (ev["page_id"], ev["line_id"]) in gold_keys), None)

if run:
    with st.spinner("Running all four steps…"):
        final, stages = run_evidence(claim["claim"], top_k=5, keep_stages=True)
        verdict = load_verifier().verify(claim["claim"], final)
    st.session_state["walk"] = {"id": claim["id"], "stages": stages, "verdict": verdict}

res = st.session_state.get("walk")
if not res or res["id"] != claim["id"]:
    st.stop()
stages, verdict = res["stages"], res["verdict"]
kept = verdict["evidence"]
doc_ids = [d["page_id"] for d in stages["docs"]]
bi, rr = stages["bi_encoder"], stages["reranked"]
rank_bi, rank_rr = gold_rank(bi), gold_rank(rr)
page_ok = gold_pages <= set(doc_ids)
in_final = full_set_in([(ev["page_id"], ev["line_id"]) for ev in kept])
right = verdict["label"] == claim["label"]
n_sent = sum(len(corpus_index()[p]["sentences"]) for p in doc_ids if p in corpus_index())

# outcome banner
said = ui.VERDICT[verdict["label"]]
if right:
    headline, why, tone = "Got it right", "Correct verdict, based on the right evidence." if in_final else \
        "Correct verdict, though not all of the answer sentences were found.", "green"
elif not in_final:
    headline, why, tone = "Got it wrong", "The answer sentence never reached the last step, so the verdict was " \
                                          "based on the wrong evidence.", "red"
else:
    headline, why, tone = "Got it wrong", "The right evidence was found, but the reading model misjudged it.", "red"
with ui.card("outcome"):
    ui.html_block(f"<div style='display:flex;gap:14px;align-items:center'>"
                  f"<div class='step-n' style='width:44px;height:44px;background:{ui.TONE[tone][1]};color:{ui.TONE[tone][0]}'>"
                  f"{ui.icon('check' if right else 'close', size=24)}</div>"
                  f"<div><div class='big' style='margin:0'>{headline}</div>"
                  f"<div class='muted small'>The system said {ui.verdict_pill(verdict['label'])} "
                  f"({ui.sureness(verdict['confidence']).lower()}). {why}</div></div></div>")

# four step cards
steps = [
    ("Keyword search", f"Find the {len(doc_ids)} pages that share the most words with the claim.",
     page_ok, "Found the page" if page_ok else "Missed the page",
     f"{len(gold_pages & set(doc_ids))}<small> of {len(gold_pages)} answer page{'s' if len(gold_pages) != 1 else ''}</small>",
     len(gold_pages & set(doc_ids)) / max(len(gold_pages), 1)),
    ("Meaning match", f"Rank their {n_sent} sentences by how close their meaning is to the claim.",
     rank_bi is not None, "In the shortlist" if rank_bi else "Not shortlisted",
     f"#{rank_bi}<small> of {len(bi)}</small>" if rank_bi else "—",
     1 - (rank_bi - 1) / max(len(bi), 1) if rank_bi else 0),
    ("Careful re-read", "A slower model re-reads them and keeps the best 5.",
     bool(rank_rr and rank_rr <= len(kept)), "In the top 5" if rank_rr and rank_rr <= len(kept) else "Dropped",
     f"#{rank_rr}<small> of {len(rr)}</small>" if rank_rr else "—",
     1 - (rank_rr - 1) / max(len(rr), 1) if rank_rr else 0),
    ("Decide", "A reading model checks if each sentence backs up or contradicts the claim.",
     right, "Right answer" if right else "Wrong answer", ui.e(said["word"]), verdict["confidence"]),
]
cols = st.columns(4, gap="small")
for n, (col, (title, body, ok, status, metric, frac)) in enumerate(zip(cols, steps), 1):
    tone = "green" if ok else "red"
    with col, ui.card(f"step_{n}"):
        ui.html_block(f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                      f"<span class='step-n' style='background:{ui.BRAND};color:#fff'>{n}</span>{ui.pill(status, tone)}</div>"
                      f"<div class='step-title'>{title}</div><div class='step-body'>{body}</div>"
                      f"<div class='step-metric'>{metric}</div><div style='margin-top:10px'>{ui.bar(frac, tone)}</div>")

# details
with ui.card("details"):
    ui.label("What each step saw")
    t1, t2, t3, t4 = st.tabs(["1 · Pages", "2 · Shortlist", "3 · Top 5 after re-read", "4 · Evidence used"])
    with t1:
        rows = "".join(f"<div class='r'><div class='side muted small' style='width:22px'>{i}</div>"
                       f"<div class='main'>{ui.e(ui.title_of(d['title']))}</div>"
                       f"<div class='side'>{ui.pill('Has the answer', 'brand', dot=False) if d['page_id'] in gold_pages else ''}</div></div>"
                       for i, d in enumerate(stages["docs"], 1))
        ui.html_block(f"<div class='rowlist'>{rows}</div>")
        missing = gold_pages - set(doc_ids)
        if missing:
            ui.html_block(f"<div class='small' style='color:{ui.TONE['red'][0]};margin-top:8px'>Missed: "
                          + ", ".join(ui.e(page_title(p)) for p in missing) + "</div>")
    with t2:
        ui.evidence_table(bi, gold_keys)
    with t3:
        ui.evidence_table(rr, gold_keys)
    with t4:
        ui.evidence_list(ui.decisive_order(verdict["label"], kept), gold_keys)
