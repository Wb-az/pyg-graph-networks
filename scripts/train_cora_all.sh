#!/usr/bin/env bash
# Training parameters for the four node-classification models on Cora.
# Only the flags that differ from the script defaults are passed; every other
# hyperparameter (2 layers, 8 heads, weight decay 5e-4, 200 epochs, seeds 0-4,
# early stop 12) is the default in planetoid_node_classification.py.
# Run from the project root:  bash scripts/train_cora_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN="uv run python src/node_classification/planetoid_node_classification.py --dataset Cora"

$TRAIN --model GCN   --hidden_channels 16  --lr 0.01  --dropout 0.5
$TRAIN --model GConv --hidden_channels 32  --lr 0.005 --dropout 0.6 --weight_decay 5e-3
$TRAIN --model GATV2 --hidden_channels 16  --lr 0.01  --dropout 0.6
$TRAIN --model SAGE  --hidden_channels 128 --lr 0.001 --dropout 0.5
