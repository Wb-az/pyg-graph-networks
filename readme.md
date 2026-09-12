# Graph Neural Networks for Node Classification

Comparing four message-passing architectures (GCN, GraphConv, GATv2,
GraphSAGE) on citation graphs, with a protocol built to say *whether* a
difference is real rather than just which number is higher: fixed seeds,
checkpoint selection on validation loss, confidence intervals, paired
statistical tests, calibration, embedding quality and explainability.

Built on PyTorch Geometric. Starts on Cora (2.7k nodes) where every choice is
cheap to verify, then scales the same pipeline to ogbn-arxiv (169k nodes,
2.3M edges, 40 imbalanced classes).

Full methodology and results: [docs/project_report.md](docs/project_report.md).

---

## Highlights

**Cora, five seeds, 1,000 test nodes, mean with 95 % CI**

| Model | Accuracy | Macro F1 | ROC AUC | ECE ↓ |
|---|---|---|---|---|
| GCN | 0.791 [0.772, 0.810] | 0.782 | 0.952 | 0.053 |
| GraphConv | 0.778 [0.753, 0.804] | 0.771 | 0.943 | 0.118 |
| **GATv2** | **0.813 [0.801, 0.825]** | **0.802** | **0.966** | 0.046 |
| GraphSAGE | 0.803 [0.796, 0.810] | 0.795 | 0.966 | **0.044** |

- GATv2 beats GCN and GraphConv on paired test nodes (McNemar, Holm-corrected
  p = 0.012); GATv2 vs GraphSAGE is not separable. Cochran's Q across all four:
  p = 0.004.
- Sum aggregation (GraphConv) is the least calibrated model by a wide margin.
- GNNExplainer on GATv2: removing the explanation subgraph flips 40 % of
  predictions, keeping only the explanation flips none.

**ogbn-arxiv** (in progress): ten seeds, three losses (cross entropy,
class-weighted, focal), a BatchNorm variant, and an Optuna search. Two findings
that shaped the protocol are already in: the citation graph must be
symmetrised or 64 % of test nodes receive no messages (accuracy stalls at
0.40), and without BatchNorm, weight decay is what keeps lr 0.01 stable.

<!-- TODO figures: copy from outputs/figures once final
![GATv2 t-SNE embeddings on Cora](docs/figures/cora/gatv2_tsne_embedding.svg)
![Training curves, all seeds](docs/figures/cora/gatv2_training_curve.svg)
-->

---

## What is in the box

| Piece | Where |
|---|---|
| Four models sharing one layer recipe, plus a BatchNorm GraphSAGE | `src/node_classification/node_models.py` |
| Cora training, Planetoid split | `src/node_classification/planetoid_node_classification.py` |
| OGB training: full batch or NeighborLoader, layer-wise inference, three losses, LR scheduling, flag validation | `src/node_classification/ogb_run.py` |
| Best-model selection, training curves, 3D embeddings, silhouette | `visualise_best.py`, `best_model.py` |
| Cochran's Q, McNemar, Friedman, Kruskal-Wallis, Wilcoxon with Holm | `compare_best.py`, `src/common/statistical_tests.py` |
| GNNExplainer with fidelity, unfaithfulness, characterisation | `explain_best.py`, `src/common/explainability.py` |
| Optuna search (TPE + pruning), CLI or notebook, resumable | `src/node_classification/optuna_search.py` |
| Metrics incl. Brier, ECE, ROC AUC, silhouette | `src/common/evaluation_metrics.py` |
| Graph EDA and plotting helpers (Altair, Plotly) | `src/common/graph_eda.py`, `altair_plots.py`, `plotly_plots.py` |
| Notebooks: Cora EDA, Cora GCN walkthrough, arxiv EDA, custom circuit data | `notebooks/` |
| Tests for the model recipe, loaders, statistics and the search | `test/` |

---

## Method in one paragraph

Every model applies `dropout → conv → activation` per layer and a linear
head; `encode` exposes the pre-head embeddings. Each seed fixes
initialisation, trains with early stopping, and reloads the checkpoint from
the epoch of lowest validation loss before touching the test set once.
Summaries report the mean over seeds with a t confidence interval. Models
are then compared at the node level (paired correctness of the best-seed
checkpoints) and at the seed level (one score per seed), with Holm
correction on the pairwise tests. Hyperparameters are searched on one seed
and confirmed on the full seed list; the tuned seed's score is never
reported.

---

## Quick start

```bash
uv sync                                   # macOS arm64 lockfile (pyg-lib wheel)

# Cora: train all four models, build the tables, compare, visualise, explain
bash scripts/train_cora_all.sh
uv run python scripts/cora_results_md.py
uv run python src/node_classification/compare_best.py --dataset Cora
uv run python src/node_classification/visualise_best.py --dataset Cora --metric f1
uv run python src/node_classification/explain_best.py --dataset Cora --metric f1

# ogbn-arxiv, ten seeds, full batch (the finished grid's SAGE cross-entropy tag)
uv run python -m src.node_classification.ogb_run --model SAGE --loss cross_entropy \
    --num_layers 3 --lr 0.01 --hidden_channels 256 --dropout 0.5 --weight_decay 0 \
    --no-dataloader --no-scheduler --epochs 500 --early_stop 501

uv run python src/node_classification/compare_best.py --dataset ogbn-arxiv \
    --tags sage_cross_entropy sagebn_cross_entropy gatv2_cross_entropy --label architectures
uv run python src/node_classification/visualise_best.py --dataset ogbn-arxiv \
    --model SAGEBN --loss weighted_ce_p0.5

# Hyperparameter search
uv run python -m src.node_classification.optuna_search --dataset ogbn-arxiv --model SAGE --n_trials 30

# Tests
uv run --with pytest python -m pytest test/
```

Python 3.12, PyTorch 2.13, PyG 2.8. Trained on Apple silicon (MPS); CUDA is
picked up automatically.

---

## Layout

`src/` (models, training, comparison, search), `notebooks/`, `scripts/`,
`docs/` (report and results), `test/`. `outputs/` (checkpoints, metrics,
figures) is git-ignored. See `docs/project_report.md` for what each piece
does and how the results were produced.

## Roadmap

- Hyperparameter search (Optuna) on the finished ogbn-arxiv grid.
- ogbn-products with the NeighborLoader and layer-wise inference.
- Graph classification and link prediction on the custom circuit dataset.

---

## Acknowledgments

**The project.** The idea, the experimental design, and the statistical
methodology, factorial architecture/loss ablations, grid search, nonparametric
paired comparison, balanced metrics under class imbalance, are the author's,
carried over from a background in explainability and compliance and standard
practice across prior work. An AI tool did not propose this project or its
research questions.

**How it was built.** Implementation went through iterative, reviewed
collaboration with AI coding assistants: a change was proposed, explained
(what it did and how it would affect other files), tested, and only then
accepted, never applied on a tool's own authority.

**Tools.** Claude (Anthropic) and Codex were used as coding assistants, to
accelerate writing and debugging the implementation, and as independent
second reviewers of each other's output. This is a solo project: reaching
this scope, a full ogbn-arxiv extension with resume-safe training,
statistical comparison, and explainability, would not have been possible
without that assistance. This also matters beyond attribution: the pipeline
is meant to be reusable, point it at a new dataset already in, or convertible
to, PyTorch Geometric's format, and the same model recipe, training loop,
selection rule, and statistical comparison apply. Debugging and testing were
AI-assisted specifically so that reuse is reliable, not just so the original
results were produced faster.
