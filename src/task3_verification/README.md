# Task 3: Claim Verification (NLI)

Takes a claim and the top-k evidence sentences from Task 2, and decides
**SUPPORTS / REFUTES / NOT ENOUGH INFO** with a natural language inference (NLI) model.

```
claim     "The Eiffel Tower is located in Berlin."
evidence  "Eiffel Tower: The Eiffel Tower is a wrought iron lattice tower ... in Paris , France ."
   -> NLI(premise = evidence, hypothesis = claim) -> contradiction 0.97 -> REFUTES
```

## Method

- Model: [`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli),
  used zero-shot (no extra training). Any 3-way NLI checkpoint works via `--model`; the label
  order is read from the model config.
- Premise = `"<page title>: <sentence>"` (PTB tokens like `-LRB-` are cleaned), hypothesis = claim.
- Every top-k evidence sentence is scored on its own, plus **one extra pair with all top-k sentences
  concatenated**, for claims that need two sentences together (multi-hop). Turn off with `--no-concat`.
- Aggregation:
  - `support = max P(entailment)`, `refute = max P(contradiction)` over all pairs
  - `max(support, refute) < threshold` → NOT ENOUGH INFO, otherwise the larger of the two wins
  - default threshold 0.5; `src/evaluate.py` tunes it on half of the eval claims

Note: the NLI model was trained on FEVER-NLI, which is built from the FEVER **train** split. Our
eval claims come from `shared_task_dev`, so there is no claim overlap, but the model has seen the
FEVER style.

## How to run

All commands run from the project root. Task 2's corpus must be built first
(`python -m src.task2_evidence_retrieval.build_corpus`, see [task2_evidence_retrieval/README.md](../task2_evidence_retrieval/README.md)).

```bash
# one claim, end-to-end (retrieval + NLI)
python -m src.pipeline --no-claim-detection "The Eiffel Tower is located in Berlin."

# evaluation: 500 SUPPORTS/REFUTES claims (claims_sample.jsonl) + 250 NEI claims from shared_task_dev
python -m src.evaluate --limit 50          # quick test
python -m src.evaluate                     # retrieved evidence (real pipeline)
python -m src.evaluate --oracle-evidence   # gold evidence (upper bound for NLI)
```

Output in `outputs/task3/`: `metrics_{retrieved,oracle}.json` and `predictions_{retrieved,oracle}.jsonl`
(per-claim NLI probabilities, so other aggregation rules can be tried without re-running the model).

### Metrics

- **Label accuracy** and **macro F1** over the 3 classes, plus per-class P/R/F1 and confusion matrix
- **FEVER score**: label is correct AND (for SUPPORTS/REFUTES) a complete gold evidence set is in the top-k evidence
- Reported twice: on all claims at `--threshold`, and on the test half with the threshold tuned on the other half

## Using it in code

```python
from src.task3_verification.nli import NLIVerifier
verifier = NLIVerifier()                     # threshold=0.5
result = verifier.verify(claim, evidence)    # evidence = EvidencePipeline.run(claim) output, or plain strings
# {"label": "REFUTES", "confidence": 0.97, "scores": {...}, "evidence": [{..., "nli": {...}, "nli_label": ...}]}
```

## Full pipeline (Task 1 → 2 → 3)

[`src/pipeline.py`](../pipeline.py) chains claim detection, evidence retrieval and NLI:

```bash
python -m src.pipeline "Good evening everyone. The Eiffel Tower is located in Berlin. I love pizza."
python -m src.pipeline --no-claim-detection "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."
python -m src.pipeline --json "..."
```

```python
from src.pipeline import FactCheckPipeline
pipe = FactCheckPipeline.load()      # needs models/task1_roberta-base_label/model unless use_claim_detection=False
pipe.check(raw_text)
# [{"claim", "check_worthy_confidence", "verdict", "verdict_confidence", "scores", "evidence": [...]}, ...]
```

## Results

TODO: fill in after running `python -m src.evaluate` (retrieved and oracle evidence).

## Known limitations

- **NEI on the subset corpus is easier than in reality**: NEI claims have no gold pages, and the subset
  corpus mostly contains pages related to the S/R sample, so retrieval usually returns unrelated text
  and NLI says neutral. On full Wikipedia, NEI claims would get closely related (but insufficient) evidence.
- Errors from Task 1 (dropped claims) and Task 2 (missing evidence → NEI) carry over; compare
  `retrieved` vs `oracle` to see how much of the error comes from retrieval.
- Max-aggregation lets one strong sentence decide the verdict; a wrong-but-confident sentence from an
  unrelated page (e.g. a disambiguation page) can flip it.
