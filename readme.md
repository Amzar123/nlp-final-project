# Fact-Checking Pipeline

raw text -> **Task 1** claim detection ([src/task1_claim_detection](src/task1_claim_detection/README.md))
-> **Task 2** evidence retrieval & reranking ([src/task2_evidence_retrieval](src/task2_evidence_retrieval/README.md))
-> **Task 3** NLI verification ([src/task3_verification](src/task3_verification/README.md)).

```bash
python -m src.pipeline "Good evening everyone. The Eiffel Tower is located in Berlin."
```

## Project layout

```
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
