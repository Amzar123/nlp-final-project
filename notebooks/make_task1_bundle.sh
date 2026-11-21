#!/usr/bin/env bash
# Bundle the code + data that Task 1 training needs on Colab (~2 MB).
# Run from the project root:  bash notebooks/make_task1_bundle.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs
rm -f outputs/task1_colab.zip
zip -q outputs/task1_colab.zip \
    src/__init__.py \
    src/paths.py \
    src/task1_claim_detection/*.py \
    data/processed/task1_train.parquet \
    data/processed/task1_val.parquet \
    data/processed/task1_test.parquet \
    data/processed/task1_meta.json
echo "Created outputs/task1_colab.zip ($(du -h outputs/task1_colab.zip | cut -f1))"
unzip -l outputs/task1_colab.zip
