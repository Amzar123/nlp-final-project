import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score

import ui
from common import (CLAIMS_PATH, CORPUS_PATH, TASK1_DIR, TASK1_MODEL_DIR, TASK2_METRICS_PATH, TASK3_METRICS,
                    TEST_RESULTS_PATH, load_detector, read_json, rel)

SERIES = ["#2a78d6", "#eb6834"]  # validated categorical slots 1-2 (dataviz reference palette)

ui.page_header("Accuracy", "How well does it work?",
               "The honest numbers for each step, in plain words. Anything not measured yet shows the "
               "command to run.")

section = st.segmented_control("Section", ["Spotting statements", "Evidence & verdicts", "Test cases", "Setup"],
                               default="Spotting statements", label_visibility="collapsed")


def line_chart(df, x, y, series, x_title, y_format="%", rule_at=None, height=250):
    """Two-series line chart: 2px lines, legend above, nearest-point hover tooltip."""
    order = list(dict.fromkeys(df[series]))
    color = alt.Color(f"{series}:N", scale=alt.Scale(domain=order, range=SERIES[:len(order)]), legend=None)
    if y_format == "%":
        yscale = alt.Scale(domain=[max(0.0, float(np.floor(df[y].min() * 10) / 10)), 1.0])
        axis = alt.Axis(format="%", tickCount=5)
    else:
        yscale, axis = alt.Scale(zero=False), alt.Axis(format=".2f", tickCount=5)
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:Q", title=x_title, scale=alt.Scale(domain=[float(df[x].min()), float(df[x].max())], nice=False)),
        y=alt.Y(f"{y}:Q", title=None, scale=yscale, axis=axis), color=color)
    hover = alt.selection_point(fields=[x], nearest=True, on="pointerover", empty=False)
    layers = [base.mark_line(strokeWidth=2, interpolate="monotone"),
              base.mark_circle(size=70, stroke="#fff", strokeWidth=2).encode(
                  opacity=alt.condition(hover, alt.value(1), alt.value(0)),
                  tooltip=[alt.Tooltip(f"{series}:N", title=" "), alt.Tooltip(f"{x}:Q", title=x_title, format=".2f"),
                           alt.Tooltip(f"{y}:Q", title="value", format=".1%" if y_format == "%" else ".3f")]
              ).add_params(hover)]
    if rule_at is not None:
        layers.insert(0, alt.Chart(pd.DataFrame({x: [rule_at]})).mark_rule(color=ui.FAINT, strokeDash=[4, 3])
                      .encode(x=f"{x}:Q"))
    legend = "".join(f"<span style='display:inline-flex;align-items:center;gap:6px;margin-right:16px'>"
                     f"<span style='width:14px;height:3px;border-radius:2px;background:{c}'></span>{ui.e(s)}</span>"
                     for s, c in zip(order, SERIES))
    ui.html_block(f"<div class='small' style='margin:4px 0 10px'>{legend}</div>")
    st.altair_chart(ui.chart_theme(alt.layer(*layers).properties(height=height)), width="stretch")


def quote_rows(texts_scores):
    rows = "".join(f"<div class='r'><div class='main'>{ui.e(t)}</div>"
                   f"<div class='side'>{ui.pill(f'score {s:.2f}' if s >= 0.01 else f'score {s:.3f}', 'gray', dot=False)}</div></div>"
                   for t, s in texts_scores)
    ui.html_block(f"<div class='rowlist'>{rows}</div>")


# =================================================================== step 1
def spotting():
    m = read_json(TASK1_DIR / "metrics.json")
    if m is None:
        with ui.card("t1_missing"):
            ui.empty_state("Not trained yet.", "Train the statement spotter first:",
                           "python -m src.task1_claim_detection.train")
        return
    preds_path = TASK1_DIR / "test_predictions.csv"
    df = pd.read_csv(preds_path) if preds_path.exists() else None
    test = m["test@tuned"]

    n = f"{len(df):,} " if df is not None else ""
    tiles = [(ui.pct(test["accuracy"]), f"labelled correctly, on {n}unseen debate sentences", "task_alt", "brand"),
             (ui.pct(test["recall"]), "of real factual statements get caught", "radar", "green"),
             (ui.pct(test["precision"]), "of what it flags really is factual", "target", "violet"),
             (ui.pct(m["val@tuned"]["accuracy"]), "on the harder validation set", "school", "amber")]
    for col, (num, text, ico, tone) in zip(st.columns(4, gap="small"), tiles):
        with col, ui.card(f"t1_tile_{ico}"):
            ui.html_block(ui.stat_tile(num, text, ico, tone))

    if df is not None:
        y, p = df["label"].to_numpy(), df["p_claim"].to_numpy()
        with ui.card("t1_cutoff"):
            ui.label("Try a different cut-off")
            thr = st.slider("A sentence counts as factual when its score is at least", 0.05, 0.95,
                            float(m["tuned_threshold"]), 0.05)
            yhat = (p >= thr).astype(int)
            a, b = st.columns([1.7, 1], gap="large")
            with a:
                rows = []
                for g in np.round(np.arange(0.05, 0.951, 0.05), 2):
                    yh = (p >= g).astype(int)
                    rows += [{"cut-off": g, "measure": "Real statements caught", "value": recall_score(y, yh)},
                             {"cut-off": g, "measure": "Flags that are right",
                              "value": precision_score(y, yh, zero_division=0)}]
                line_chart(pd.DataFrame(rows), "cut-off", "value", "measure", "cut-off", rule_at=thr, height=230)
            with b:
                rec, prec, acc = recall_score(y, yhat), precision_score(y, yhat, zero_division=0), accuracy_score(y, yhat)
                ui.html_block(f"<div class='big' style='margin-bottom:0'>At {thr:.2f}</div>"
                              + ui.bar(rec, "brand", "Real statements caught", ui.pct(rec))
                              + ui.bar(prec, "violet", "Flags that are right", ui.pct(prec))
                              + ui.bar(acc, "green", "Overall accuracy", ui.pct(acc, 1))
                              + f"<div class='small muted' style='margin-top:16px'>We use {m['tuned_threshold']:.2f}, "
                                "picked on validation data. Lower = catches more, but flags more chit-chat.</div>")

    gap = read_json(TASK1_DIR / "domain_gap_paper_dev.json")
    cols = st.columns([1.5, 1], gap="medium") if (df is not None and gap) else [st.container(), st.container()]
    if df is not None:
        with cols[0], ui.card("t1_errors"):
            ui.label("Where it goes wrong")
            kind = st.segmented_control("Mistakes", ["Flagged, but not factual", "Factual, but missed"],
                                        default="Flagged, but not factual", label_visibility="collapsed")
            missed = kind == "Factual, but missed"
            err = (df[(yhat == 0) & (y == 1)].sort_values("p_claim") if missed else
                   df[(yhat == 1) & (y == 0)].sort_values("p_claim", ascending=False))
            ui.html_block(f"<div class='small muted' style='margin:4px 0 6px'>{len(err)} sentences at this cut-off. "
                          + ("These are factual, but the model wasn't convinced." if missed else
                             "These sound factual, but annotators marked them as opinion or small talk.") + "</div>")
            quote_rows(zip(err["text"][:5], err["p_claim"][:5]))
            if len(err) > 5:
                with st.expander(f"Show all {len(err)}"):
                    st.dataframe(err[["text", "p_claim"]].rename(columns={"text": "sentence", "p_claim": "score"}),
                                 hide_index=True, width="stretch")
    if gap:
        with cols[1], ui.card("t1_gap"):
            ui.label("Does it work on FEVER claims too?")
            ui.html_block(f"<div class='big' style='margin-bottom:4px'>{ui.pct(gap['pass_rate@tuned']['all'], 1)} of "
                          f"{gap['n_claims']:,} FEVER claims pass step 1</div>"
                          + ui.bar(gap["pass_rate@tuned"]["all"], "brand")
                          + "<div class='muted small' style='margin:10px 0 6px'>All of them are factual, so ideally "
                            "100% would. The ones it drops are short and oddly worded:</div>")
            low = pd.DataFrame(gap.get("lowest_scored_claims", [])).drop_duplicates("claim").head(5)
            quote_rows(zip(low["claim"], low["p_claim"]))

    with ui.card("t1_more"):
        with st.expander("Training details"):
            log = pd.DataFrame(m["log_history"])
            tr = log.dropna(subset=["loss"])[["step", "loss"]].assign(measure="Training data")
            ev = log.dropna(subset=["eval_loss"])[["step", "eval_loss"]].rename(columns={"eval_loss": "loss"}) \
                .assign(measure="Validation data")
            st.caption("Loss, lower is better. Validation loss creeps up in the last epoch while accuracy holds: "
                       "the model got a bit overconfident, not worse.")
            line_chart(pd.concat([tr, ev]), "step", "loss", "measure", "training step", y_format=".2f", height=220)
            st.json({**m["args"], "best_step": m["best_step"], "threshold": m["tuned_threshold"]}, expanded=False)
        with st.expander("Score your own sentences"):
            txt = st.text_area("One sentence per line", "Good evening everyone.\nIndonesia has more than 17,000 islands.")
            if st.button("Score", type="primary"):
                det = load_detector()
                lines = [s.strip() for s in txt.splitlines() if s.strip()]
                rows = "".join(f"<div class='r'><div class='main'>{ui.e(s)}</div><div class='side'>"
                               f"{ui.pill(('Factual ' if sc >= det.threshold else 'Not factual ') + ui.pct(sc), 'green' if sc >= det.threshold else 'gray')}"
                               f"</div></div>" for s, sc in zip(lines, det.score(lines)))
                ui.html_block(f"<div class='rowlist'>{rows}</div>")


# =================================================================== steps 2 + 3
def evidence_and_verdicts():
    m2 = read_json(TASK2_METRICS_PATH)
    with ui.card("t2"):
        ui.label("Step 2 · Finding the evidence")
        if m2 is None:
            ui.empty_state("Not measured yet", "Checks how often search finds the page and the exact sentence with the "
                           "answer, for the 500 test claims. Takes a few minutes.",
                           "python -m src.evaluate_retrieval            # add --limit 50 for a quick run")
        else:
            b, a = m2.get("before/hit@1", 0), m2.get("after/hit@1", 0)
            ui.html_block(f"<div class='big'>For {ui.pct(m2.get('doc_recall@5', 0))} of claims, the page with the "
                          f"answer is in the top 5 search results.</div>")
            ui.stats([(ui.pct(m2.get("candidate_recall", 0)), "answer makes the 30-sentence shortlist"),
                      (ui.pct(m2.get("after/evidence_recall@5", 0)), "full answer is in the final 5"),
                      (ui.pct(a), f"answer ranked first after re-reading (was {ui.pct(b)})")])

    runs = {k: read_json(p) for k, p in TASK3_METRICS.items()}
    m3 = runs.get("retrieved")
    with ui.card("t3"):
        ui.label("Step 3 · Deciding the verdict")
        if m3 is None:
            ui.empty_state("Not measured yet", "Runs the full system on the test claims (plus some with no answer in "
                           "Wikipedia) and scores every verdict.",
                           "python -m src.evaluate                     # add --limit 100 for a quick run\n"
                           "python -m src.evaluate --oracle-evidence   # same, but given the right evidence")
            return
        f = m3["fixed_threshold"]
        ui.html_block(f"<div class='big'>Right verdict on {ui.pct(f['accuracy'])} of {f['n']} claims.</div>")
        items = [(ui.pct(f["accuracy"]), "right verdict"),
                 (ui.pct(f["fever_score"]), "right verdict and right evidence")]
        oracle = runs.get("oracle")
        if oracle:
            items.append((ui.pct(oracle["fixed_threshold"]["accuracy"]), "right verdict when given the correct evidence"))
        ui.stats(items)

    names = {"SUPPORTS": "true", "REFUTES": "false", "NOT ENOUGH INFO": "can't tell"}
    cm = f["confusion_matrix"]
    a, b = st.columns([1, 1.3], gap="medium")
    with a, ui.card("t3_cm"):
        ui.label("What it said vs. the truth")
        st.dataframe(pd.DataFrame(cm["matrix"], index=[f"Really {names[l]}" for l in cm["labels"]],
                                  columns=[f"Said {names[l]}" for l in cm["labels"]]), width="stretch")
        st.caption("The diagonal is correct answers.")
    with b, ui.card("t3_sweep"):
        ui.label("How strict should it be?")
        sw = pd.DataFrame(m3["sweep_tune_half"])
        long = pd.concat([sw.assign(measure="Right verdict", value=sw["accuracy"]),
                          sw.assign(measure="Right verdict + evidence", value=sw["fever_score"])])
        line_chart(long, "threshold", "value", "measure", "sureness needed to say true/false",
                   rule_at=m3["tuned_threshold"], height=220)


# =================================================================== test cases
def test_cases():
    res = read_json(TEST_RESULTS_PATH)
    if res is None:
        with ui.card("tc_missing"):
            ui.empty_state("No results yet", "Write cases in <code>tests/test_cases.json</code>, then run:",
                           "python -m src.run_test_cases")
        return
    cd, ver = res.get("claim_detection", []), res.get("verification", [])
    with ui.card("tc_head"):
        ui.label("Hand-written test cases")
        ui.stats([(f"{sum(r['passed'] for r in cd)}/{len(cd)}", "factual-or-not checks passed"),
                  (f"{sum(r['passed'] for r in ver)}/{len(ver)}", "fact-checks passed")])
        st.caption("From tests/test_cases.json. Good for catching regressions, too few to measure accuracy.")

    a, b = st.columns(2, gap="medium")
    with a, ui.card("tc_ver"):
        ui.label("Fact-checks")
        rows = []
        for r in sorted(ver, key=lambda r: r["passed"]):
            side = (ui.verdict_pill(r["got"]) if r["passed"] else
                    f"{ui.verdict_pill(r['got'])}<div class='small muted' style='text-align:right;margin-top:4px'>"
                    f"should be {ui.VERDICT[r['expected']]['word'].lower()}</div>")
            mark = ui.icon("check_circle" if r["passed"] else "cancel",
                           ui.TONE["green" if r["passed"] else "red"][2], 20)
            rows.append(f"<div class='r'><div class='side'>{mark}</div><div class='main'>{ui.e(r['claim'])}</div>"
                        f"<div class='side'>{side}</div></div>")
        ui.html_block(f"<div class='rowlist'>{''.join(rows)}</div>")
    with b, ui.card("tc_cd"):
        ui.label("Factual or not?")
        rows = []
        for r in sorted(cd, key=lambda r: r["passed"]):
            mark = ui.icon("check_circle" if r["passed"] else "cancel",
                           ui.TONE["green" if r["passed"] else "red"][2], 20)
            want = "Factual" if r["expected"] == "claim" else "Not factual"
            rows.append(f"<div class='r'><div class='side'>{mark}</div><div class='main'>{ui.e(r['text'])}"
                        f"<div class='small muted'>{ui.e(r.get('category', ''))}</div></div>"
                        f"<div class='side'>{ui.pill(want, 'gray', dot=False)}</div></div>")
        ui.html_block(f"<div class='rowlist'>{''.join(rows)}</div>")


# =================================================================== setup
def setup():
    with ui.card("setup"):
        top_l, top_r = st.columns([3, 1], vertical_alignment="center")
        with top_l:
            ui.label("Files this portal reads")
        if top_r.button("Re-read files", icon=":material/refresh:", width="stretch"):
            read_json.clear()
        items = [
            ("Statement spotter (step 1 model)", TASK1_MODEL_DIR / "config.json", "python -m src.task1_claim_detection.train"),
            ("Wikipedia library", CORPUS_PATH, "python -m src.task2_evidence_retrieval.build_corpus"),
            ("Test claims", CLAIMS_PATH, "python -m src.task2_evidence_retrieval.build_corpus"),
            ("Test case results", TEST_RESULTS_PATH, "python -m src.run_test_cases"),
            ("Search measurements", TASK2_METRICS_PATH, "python -m src.evaluate_retrieval"),
            ("Verdict measurements", TASK3_METRICS["retrieved"], "python -m src.evaluate"),
            ("Verdict measurements, correct evidence given", TASK3_METRICS["oracle"],
             "python -m src.evaluate --oracle-evidence"),
        ]
        for name, path, cmd in items:
            ok = path.exists()
            ui.html_block(f"<div class='rowlist'><div class='r'><div class='main'><b style='font-weight:600'>{name}</b>"
                          f"<div class='small muted'>{rel(path)}</div></div><div class='side'>"
                          f"{ui.pill('Ready', 'green') if ok else ui.pill('Missing', 'amber')}</div></div></div>")
            if not ok:
                st.code(cmd, language="bash")
        st.caption("The NLI and search models download from Hugging Face the first time they're used.")


{"Spotting statements": spotting, "Evidence & verdicts": evidence_and_verdicts,
 "Test cases": test_cases, "Setup": setup}.get(section or "Spotting statements")()
