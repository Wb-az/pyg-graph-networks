#!/usr/bin/env bash
# ogbn-arxiv experiment grid (docs/project_report.md, section 6.1).
# Every run uses the same baseline flags; only the variable under study changes.
# Flags not listed (Adam, weight decay 5e-4, scheduler on with threshold 0.005,
# ten seeds 0-4, 20, 42, 123, 1234, 12345) are the defaults in ogb_run.py.
# Run from the project root:  bash scripts/train_arxiv_all.sh
#
# Outputs are tagged <model>_<loss> (e.g. sage_cross_entropy_summary.csv,
# ogbn-arxiv_sage_cross_entropy_<seed>_best.pth), so runs never overwrite
# each other.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="uv run python -m src.node_classification.ogb_run --dataset ogbn-arxiv \
    --num_layers 3 --hidden_channels 256 --dropout 0.5 --lr 0.01 \
    --epochs 500 --early_stop 50 --no-scheduler --no-dataloader --log_steps 50"

# Grid settings follow the OGB reference for ogbn-arxiv: 3 layers, 256 hidden,
# dropout 0.5, lr 0.01, 500 epochs, 10 runs, constant learning rate (no
# scheduler; early stopping on validation loss only), plus weight decay 5e-4,
# needed without BatchNorm. First A1 pass (dropout 0.3, F1 scheduler) is kept
# as sage_cross_entropy_f1sched_*.
# A1: GraphSAGE, cross entropy.
$BASE --model SAGE --loss cross_entropy

# Remaining grid: GraphSAGE, GraphSAGE-BN and GATv2 under cross entropy and
# weighted CE (balanced --weight_power 1 on SAGE only; tempered 0.5 elsewhere) (GCN, GConv and focal loss dropped, September 2026; focal was 2x
# slower per epoch on MPS and no better than weighted CE in a 100-epoch pilot). The same runs are listed in
# notebooks/04_arxiv_run_node_classification.ipynb for Colab; uncomment here
# only the ones you run in the terminal. Nothing skips finished runs here:
# check outputs/metrics/ogbn-arxiv/<model>_<loss>_summary.csv first.
# $BASE --model SAGE   --loss weighted_ce
# $BASE --model SAGE   --loss weighted_ce --weight_power 0.5
# $BASE --model SAGEBN --loss cross_entropy
# $BASE --model SAGEBN --loss weighted_ce --weight_power 0.5
# GATv2: 8 x 32 = 256 total width, needs an L4/A100; 8 x 16 fits a T4 but is narrower
# $BASE --model GATV2 --heads 8 --hidden_channels 32 --loss cross_entropy
# $BASE --model GATV2 --heads 8 --hidden_channels 32 --loss weighted_ce --weight_power 0.5
