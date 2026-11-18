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

# 4. try it on raw text
python -m src.task1_claim_detection.infer --all "Hello everyone! Barack Obama was born in Hawaii. I love pizza."

# 5. domain gap: how many real FEVER claims (paper_dev) pass the classifier
python -m src.task1_claim_detection.domain_gap
```

### Training on Google Colab

1. `bash notebooks/make_task1_bundle.sh` → `outputs/task1_colab.zip` (code + Task 1 data, ~1.4 MB)
2. Upload it to Google Drive in `MyDrive/nlp-final-project/`
3. Open `notebooks/task1_colab.ipynb` in Colab, choose a T4 GPU, then Run all
4. Download `task1_models.zip` from the same Drive folder and unzip it in the project root

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

## Results (test = groundtruth.csv)

| Model | Mapping | Macro F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| TF-IDF + LogReg | factual (main) | 0.921 | 0.898 | 0.877 | 0.976 |
| TF-IDF + LogReg | check-worthy | 0.908 | 0.829 | 0.895 | 0.977 |
| roberta-base | factual (main) | _pending_ | | | |
| roberta-base | check-worthy | _pending_ | | | |

Validation scores are much lower than test (TF-IDF macro F1 about 0.81 vs 0.92). Validation uses
noisy crowd labels, while test uses expert labels.

## Known limitations

- **Domain gap:** the model is trained on spoken debate sentences, but the pipeline feeds it
  Wikipedia-style FEVER claims. `domain_gap.py` measures the pass rate, which caps the recall of
  the whole pipeline.
- **Pronoun claims** ("He won the award in 2010") pass Task 1 but cannot be verified without
  knowing who "he" is. Decontextualization is out of scope.
- **Label noise:** crowdsourced labels are inconsistent (e.g. "And I'm against price controls."
  is labelled check-worthy factual).
