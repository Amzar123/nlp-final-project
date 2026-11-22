# Task 1: Claim Detection

Splits raw text into sentences and keeps the ones that are **factual claims**, so that only
those go on to evidence retrieval (Task 2) and verification (Task 3).

```
"Hello everyone! I think pizza is great. Barack Obama was born in Hawaii in 1961."
   -> "Hello everyone!"                           not a claim (greeting)
   -> "I think pizza is great."                   not a claim (opinion)
   -> "Barack Obama was born in Hawaii in 1961."  claim -> sent to Task 2
```

## Data: ClaimBuster

[ClaimBuster](https://zenodo.org/records/3609356) has about 23.5k sentences from US presidential
debates (1960–2016), each labelled `1` = CFS (check-worthy factual), `0` = UFS (unimportant factual)
or `-1` = NFS (non-factual). FEVER isn't used for training because it only contains claims, so it
has no negative examples.

| Column | Mapping | Used for |
|---|---|---|
| `label` | CFS + UFS = 1, NFS = 0 | main model ("can this be checked against Wikipedia?") |
| `label_checkworthy` | CFS = 1, UFS + NFS = 0 | comparison run (ClaimBuster's original framing) |

Splits: `crowdsourced.csv` → train (20,151) / val (2,240), stratified on the verdict.
`groundtruth.csv` (expert labels) → test (1,032).
Sentences whose text also appears in the test set (15) are removed from train/val. Duplicate texts
inside crowdsourced are collapsed, and dropped completely when their labels disagree.

## How to run

All commands run from the project root.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 1. download ClaimBuster (+ FEVER) and build the splits
python -m src.data_prep.download --no-wiki
python -m src.data_prep.build_task1_data          # -> data/processed/task1_*.parquet

# 2. baseline (CPU, ~10 s)
python -m src.task1_claim_detection.baseline

# 3. train roberta-base
#    GPU (recommended, see notebooks/task1_colab.ipynb):
python -m src.task1_claim_detection.train --batch-size 32 --grad-accum 1
python -m src.task1_claim_detection.train --batch-size 32 --grad-accum 1 --label-col label_checkworthy
#    8 GB Apple M1 also works with the defaults (batch 16 x 2), but takes ~1.5-2 h per run.

# 4. try it on raw text (full pipeline: claim detection -> retrieval -> NLI)
python -m src.pipeline "Hello everyone! Barack Obama was born in Hawaii. I love pizza."

# 5. domain gap: how many real FEVER claims (paper_dev) pass the classifier
python -m src.task1_claim_detection.domain_gap
```

### Training on Google Colab

1. `bash notebooks/make_task1_bundle.sh` → `outputs/task1_colab.zip` (code + Task 1 data, ~1.4 MB)
2. Upload it to Google Drive in `MyDrive/nlp-final-project/`
3. Open `notebooks/task1_colab.ipynb` in Colab, choose a T4 GPU, then Run all
4. Download each `task1_<model>_<mapping>.zip` from the same Drive folder (one per run) and unzip them in the project root

Trained models are **not in git** (about 500 MB each). Share them through Drive or the Hugging Face Hub.

## Using it in the pipeline

```python
from src.task1_claim_detection.infer import ClaimDetector

detector = ClaimDetector()                 # loads models/task1_roberta-base_label/model
claims = detector.detect(raw_text)         # only sentences with P(claim) >= tuned threshold
# [{"index": 2, "sentence": "...", "p_claim": 0.97, "is_claim": True}, ...]
detector.detect(raw_text, return_all=True) # every sentence, with its score
```

The threshold is tuned on the validation set during training and saved in
`model/threshold.json`. `p_claim` maps to `check_worthy_confidence` in the final pipeline output.

## Training details

- Backbone `roberta-base` (`microsoft/deberta-v3-small` optional), with a 2-class head
- Weighted cross-entropy (class weights from the train split, in `task1_meta.json`)
- lr 2e-5, 3 epochs, warmup 6%, weight decay 0.01, max length 128, effective batch 32
- The best step is chosen by validation macro F1, then the threshold is tuned on validation
- Test is scored only once: at threshold 0.5 and at the tuned threshold

Output per run, in `models/task1_<model>_<label-col>/`: `model/` (weights, tokenizer,
`threshold.json`), `metrics.json`, `test_predictions.csv`.

## Results

### Classification (test = groundtruth.csv, at the threshold tuned on validation)

| Model | Mapping | Val macro F1 | Test macro F1 | Precision | Recall | ROC-AUC | Threshold |
|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | factual | 0.813 | 0.921 | 0.898 | 0.877 | 0.976 | 0.59 |
| TF-IDF + LogReg | check-worthy | 0.795 | 0.908 | 0.829 | 0.895 | 0.977 | 0.54 |
| **roberta-base** | **factual (main model)** | **0.882** | **0.976** | 0.958 | 0.973 | 0.998 | 0.60 |
| roberta-base | check-worthy | 0.863 | 0.969 | 0.939 | 0.966 | 0.997 | 0.53 |
| deberta-v3-small | factual | 0.869 | 0.979 | 0.970 | 0.970 | 0.997 | 0.58 |

- **Main model: `roberta-base`, factual mapping.** It has the best validation macro F1. Deberta-v3-small
  scores slightly higher on test (0.979 vs 0.976), but the model is chosen on validation, never on test.
- Transformers beat the TF-IDF baseline by about 5.5 macro F1 points.
- Validation is much lower than test for every model. Validation uses noisy crowd labels, while test
  uses expert labels.
- Leakage check: only 14 test sentences have a near-duplicate (TF-IDF cosine ≥ 0.8) in train/val.
  Removing them leaves the roberta score unchanged (0.9755 → 0.9753). All test sentences come from
  debates that also appear in training, so the split doesn't measure generalization to unseen debates.

### Domain gap (plan 1.5): FEVER `paper_dev` claims predicted as claim

All 9,999 FEVER claims are factual statements, so ideally all of them should pass.

| Model | Pass rate | SUPPORTS | REFUTES | NOT ENOUGH INFO |
|---|---|---|---|---|
| **roberta-base, factual** | **0.945** | 0.949 | 0.929 | 0.957 |
| deberta-v3-small, factual | 0.945 | 0.947 | 0.932 | 0.955 |
| roberta-base, check-worthy | **0.172** | 0.126 | 0.218 | 0.173 |

- With the main model, Task 1 caps end-to-end recall at about **94.5%**. The claims it drops are short,
  odd ones that barely look like statements, mostly generated variants (*"Wish Upon starred tomato."*,
  *"Morse Code is communication."*).
- **The check-worthy mapping drops 83% of FEVER claims** while scoring almost as well on its own test set.
  Encyclopedic facts are not "check-worthy" in the political sense, which supports the factual mapping
  for this pipeline.
- spaCy keeps 9,967 of 9,999 claims as a single sentence.

Full report: `models/task1_roberta-base_label/domain_gap_paper_dev.json`.

### Example (`infer.py --all`)

```
[  -  ] 0.003  Good evening everyone, and thank you for having me.
[  -  ] 0.002  I honestly think this city deserves better.
[CLAIM] 0.999  Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
[  -  ] 0.001  Can you believe that?
[CLAIM] 0.998  The Eiffel Tower is located in Berlin.      <- false, but still a claim (Task 3 judges truth)
[  -  ] 0.001  We must work together to build a brighter future.
[CLAIM] 0.999  Unemployment fell to 4.9 percent in 2016.
[CLAIM] 0.999  He won the award in 2010.                  <- passes, but unverifiable (pronoun)
```

## Known limitations

- **Domain gap:** the model is trained on spoken debate sentences, but the pipeline feeds it
  Wikipedia-style FEVER claims. `domain_gap.py` measures the pass rate, which caps the recall of
  the whole pipeline.
- **Pronoun claims** ("He won the award in 2010") pass Task 1 but cannot be verified without
  knowing who "he" is. Decontextualization is out of scope.
- **Label noise:** crowdsourced labels are inconsistent (e.g. "And I'm against price controls."
  is labelled check-worthy factual).
