#!/usr/bin/env bash
# Download the FiNER and XBRL Formula splits released with ACE (https://github.com/ace-agent/ace).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/finance
base=https://raw.githubusercontent.com/ace-agent/ace/main/eval/finance/data
for f in finer_train_batched_1000_samples finer_val_batched_500_samples finer_test_subset_006_seed42 \
         formula_train_subset_500 formula_val_subset_300 formula_test; do
  curl -fsSL "$base/$f.jsonl" -o "data/finance/$f.jsonl"
done
echo "Saved to data/finance/"
