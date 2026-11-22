# Fact-Checking Pipeline

raw text -> **Task 1** claim detection ([src/task1_claim_detection](src/task1_claim_detection/README.md))
-> **Task 2** evidence retrieval & reranking ([src/task2_evidence_retrieval](src/task2_evidence_retrieval/README.md))
-> **Task 3** NLI verification ([src/task3_verification](src/task3_verification/README.md)).

New to the code? Start with the overview in [docs/PANDUAN_KODE.md](docs/PANDUAN_KODE.md) (Bahasa Indonesia):
data flow, what each file does, setup order and the main classes.

```bash
python -m src.pipeline "Good evening everyone. The Eiffel Tower is located in Berlin."
```

## Web UI (dev portal)

```bash
streamlit run app/portal.py
```

| Page | What it's for |
|---|---|
| Check | Paste text, get each factual statement marked *Looks true / Looks false / Can't tell* with the Wikipedia sentences behind it. Thresholds live under Settings; raw scores under each card. |
| Step by step | Run a FEVER test claim (answer known) and see each step, BM25 → bi-encoder → cross-encoder → NLI, and where the answer sentence was lost. |
| Library | Search the Wikipedia subset to check whether a topic is covered at all. |
| Accuracy | Task 1 metrics with a cut-off slider and its mistakes, Task 2/3 metrics (once generated), hand-picked test results, and a setup checklist. |

Layout: `app/portal.py` (navigation), `app/views/` (one file per page), `app/ui.py` (visual language),
`app/common.py` (cached data + model loading), `app/static/` (logo), theme in `.streamlit/config.toml`.
Models load on first use and stay cached for the server session.

## Project layout

```
app/                          # Streamlit dev portal (portal.py + views/, see "Web UI" above)
src/
  paths.py                    # shared locations: data/, models/, outputs/
  pipeline.py                 # end-to-end Task 1 -> 2 -> 3
  data_prep/                  # dataset download + Task 1 preprocessing
  task1_claim_detection/      # ClaimBuster classifier (train / infer / baseline)
  task2_evidence_retrieval/   # BM25 -> bi-encoder -> cross-encoder reranking
  task3_verification/         # NLI verdict + evaluation
notebooks/                    # Colab training notebook + bundle script
data/  models/  outputs/      # generated, git-ignored
```

Run everything from the project root as a module: `python -m src.<package>.<module>`.
