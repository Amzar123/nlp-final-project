from collections import Counter

import streamlit as st

import ui
from common import CORPUS_PATH, corpus_index, load_corpus, load_retriever, rel

ui.page_header("Library", "What the system knows",
               "Facts are only checked against these pages, not all of Wikipedia. If a topic isn't here, "
               "the answer will be <i>Can't tell</i>. Search to see whether yours is covered.")

if not CORPUS_PATH.exists():
    with ui.card("missing"):
        ui.empty_state("The library hasn't been built yet.", f"<code>{rel(CORPUS_PATH)}</code> is missing. Build it with:",
                       "python -m src.task2_evidence_retrieval.build_corpus")
    st.stop()

corpus = load_corpus()
index = corpus_index()
src = Counter(p.get("source") for p in corpus)

# ================================================================ stat tiles
tiles = [(f"{len(corpus):,}", "Wikipedia pages in total", "library_books", "brand"),
         (f"{src.get('gold', 0):,}", "hold answers to test claims", "verified", "green"),
         (f"{src.get('hard_negative', 0):,}", "look-alikes, to make search harder", "content_copy", "amber"),
         (f"{src.get('random', 0):,}", "random background pages", "shuffle", "violet")]
for col, (n, text, ico, tone) in zip(st.columns(4, gap="small"), tiles):
    with col, ui.card(f"tile_{ico}"):
        ui.html_block(ui.stat_tile(n, text, ico, tone))


def open_page(pid):
    st.session_state["lib_page"] = pid


# ================================================================ search (left) + page (right)
left, right = st.columns([1, 2.1], gap="medium")
with left, ui.card("hits"):
    ui.label("Search a topic")
    q = st.text_input("Search a topic", placeholder="e.g. Japan, Mogadishu", label_visibility="collapsed",
                      icon=":material/search:")
    how = st.segmented_control("Match", ["Title", "Keywords"], default="Title", label_visibility="collapsed",
                               help="Keywords uses the same search as step 1 of the claim checker.")
    hits = []
    if q.strip():
        if how == "Keywords":
            retriever = load_retriever()
            hits = [retriever.pages[i]["id"] for i, s in retriever.retrieve_documents(q, k=12) if s > 0]
        else:
            ql = q.lower()
            matches = [p for p in corpus if ql in p["title"].lower()]
            matches.sort(key=lambda p: (p["title"].lower() != ql, not p["title"].lower().startswith(ql),
                                        len(p["title"])))
            hits = [p["id"] for p in matches[:12]]
        if hits and st.session_state.get("lib_page") not in hits:
            st.session_state["lib_page"] = hits[0]
    elif "lib_page" not in st.session_state:
        st.session_state["lib_page"] = next(p["id"] for p in corpus if p.get("source") == "gold")

    if q.strip() and not hits:
        ui.empty_state(f"Nothing about “{q}”", "Not in the library, so claims about it will come back as "
                       "<i>Can't tell</i>.", ico="search_off")
    else:
        ui.label(f"{len(hits)} result{'s' if len(hits) != 1 else ''}" if hits else "Start here")
        with st.container(key="hitlist"):
            for pid in hits or [st.session_state["lib_page"]]:
                active = pid == st.session_state["lib_page"]
                st.button(ui.title_of(index[pid]["title"]), key=f"open_{pid}",
                          type="secondary" if active else "tertiary", on_click=open_page, args=(pid,), width="stretch")

with right, ui.card("page"):
    page = index[st.session_state["lib_page"]]
    kind = {"gold": ("Holds a test answer", "green"), "hard_negative": ("Look-alike page", "amber"),
            "random": ("Background page", "violet")}.get(page.get("source"), ("Page", "gray"))
    n = len(page["sentences"])
    n_lines = f"{n} sentence{'s' if n != 1 else ''}"
    ui.html_block(f"<div style='display:flex;gap:8px'>{ui.pill(kind[0], kind[1])}"
                  f"{ui.pill(n_lines, 'gray', dot=False)}</div>"
                  f"<div class='big' style='font-size:1.5rem;margin:12px 0 8px'>{ui.e(ui.title_of(page['title']))}</div>")
    body = " ".join(f"<span class='ln'>{lid}</span>{ui.e(ui.tidy(t))}" for lid, t in page["sentences"] if t)
    ui.html_block(f"<div class='article'>{body}</div>")
