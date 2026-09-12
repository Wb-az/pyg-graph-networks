# Graph neural networks for node classification: methodology and results

Living document. Sections marked **TODO** are placeholders that fill in as
experiments finish. Figures live under `docs/figures/<dataset>/`; copy them
from the git-ignored `outputs/figures/<dataset>/` when a result is final
(`cp outputs/figures/cora/gatv2_training_curve.svg docs/figures/cora/`).

Repository: https://github.com/Wb-az/pyg-graph-networks

---

## 1. Scope

The project compares four message-passing architectures on node
classification, first on a small citation graph (Cora) where every design
choice can be checked cheaply, then on a graph two orders of magnitude larger
(ogbn-arxiv) where memory, class imbalance and training budget start to
matter. The same models, training loop, selection rule and statistical
comparison are used on both, so differences in the results come from the
data, not from the pipeline.

| Stage | Dataset | Status |
|---|---|---|
| Exploratory analysis | Cora, ogbn-arxiv, custom circuit dataset | done (notebooks 01, 03, 04) |
| Node classification, 4 models, 5 seeds | Cora | done, `docs/cora_results.md` |
| Statistical model comparison | Cora | done, `docs/cora_model_comparison.md` |
| Embeddings and explainability | Cora | done (GATv2, the best model) |
| Node classification, GraphSAGE, 3 losses | ogbn-arxiv | done, section 6.1, 6.3 |
| BatchNorm and GATv2 variants | ogbn-arxiv | done, section 6.1, 6.4 |
| Statistical model comparison | ogbn-arxiv | done, `docs/ogbn-arxiv_model_comparison_*.md` |
| Hyperparameter search (Optuna) | ogbn-arxiv | script ready, runs pending, section 6.5 |
| Graph classification, link prediction | custom | **TODO**, stubs only |

---

## 2. Datasets

### 2.1 Cora (Planetoid split)

| Property | Value |
|---|---|
| Nodes / edges (undirected) | 2,708 / 5,278 |
| Features | 1,433 binary bag-of-words, ~18 non-zero per node |
| Classes | 7 |
| Train / val / test | 140 / 500 / 1,000 (fixed public split) |

Cora's features are wide and sparse, so the first layer acts as a
compressor and small hidden sizes (16 to 128) are enough. With 140 labelled
nodes, regularisation dominates every design decision.

*Figure placeholders*

- `![Cora class distribution per split](/Users/az-asc/PycharmProjects/GraphNetworks/outputs/figures/cora/data_dist.svg)` **TODO copy**
- `![Cora graph, ForceAtlas2 layout](figures/cora/cora_graph_atlas2.svg)` **TODO copy**

### 2.2 ogbn-arxiv

| Property | Value |
|---|---|
| Nodes | 169,343 arXiv CS papers |
| Edges | 1,166,243 directed citations, 2,332,486 after symmetrising |
| Features | 128-dim dense skip-gram embeddings of title and abstract |
| Classes | 40 arXiv subject areas, strongly imbalanced |
| Split | by year: train up to 2017 (90,941), val 2018 (29,799), test 2019+ (48,603) |

Two properties drive the methodology on this graph:

- **Direction.** Citations point from new papers to old ones. Left directed,
  37 % of all nodes and 64 % of *test* nodes have no incoming edge and receive
  no messages, so a GNN degrades to an MLP on them. The loader symmetrises
  the edge index, as the OGB reference does with `adj_t.to_symmetric()`.
- **Imbalance.** Accuracy is dominated by a handful of large classes. Macro
  F1 is the headline metric, and class-weighted and focal losses are
  compared against plain cross entropy.

*Figure placeholders*

- `![arxiv class distribution per split](figures/arxiv/data_dist.svg)` **TODO copy**
- `![arxiv graph sample, ForceAtlas2](figures/arxiv/ogbnarx_graph_atlas2.svg)` **TODO copy**

### 2.3 Custom dataset (electronic circuits)

Kaggle circuit-step data (`notebooks/04_eda_custom_dataset.ipynb`): one
graph per circuit, components as nodes. Intended tasks are next-component
prediction (graph classification) and missing-link prediction. **TODO**:
graph construction, splits, baselines.

---

## 3. Models

All four models live in `src/node_classification/node_models.py` and share
one layer recipe, so the comparison isolates the message-passing operator:

```
encode:  per layer  dropout -> conv -> activation
forward: encode -> dropout -> linear head
```

| Model | Operator | Aggregation | Notes |
|---|---|---|---|
| GCN | `GCNConv` | symmetric-normalised sum | Kipf & Welling |
| GConv | `GraphConv` | sum, separate root weight | Morris et al.; sum makes it distinct from SAGE |
| GATv2 | `GATv2Conv` | attention, multi-head | ELU activation; last layer single head |
| GraphSAGE | `SAGEConv` | mean, separate root weight | Hamilton et al. |
| GraphSAGE-BN | `SAGEConv` + `BatchNorm1d` | mean | arxiv only; `dropout -> conv -> norm -> act`, the OGB reference layout |

`encode` returns the pre-head hidden vectors, which are what the embedding
plots and silhouette scores use. Dropout on the input features is part of
the recipe (PyG example-script order). It suits Cora's sparse features; on
arxiv's dense features it is a stronger dose, which is why dropout 0.3 is
used there against 0.5 on Cora.

---

## 4. Training and evaluation protocol

| Element | Cora | ogbn-arxiv |
|---|---|---|
| Training | full batch | full batch (`--no-dataloader`); NeighborLoader available for ogbn-products |
| Width | per model (Cora tables) | 256 hidden for every model; GATv2 8 heads × 32 (8 × 16 on a T4) |
| Epoch budget | 200 | 500 (one gradient step per epoch) |
| Early stopping | 12 epochs on validation loss | 50 epochs on validation loss |
| LR schedule | none | none (`--no-scheduler`, constant lr 0.01 as in the OGB reference); the first A1 pass used ReduceLROnPlateau on macro F1 |
| Model selection | epoch of lowest validation loss, checkpoint reloaded for test | same |
| Seeds | 0 to 4 | 0 to 4, 20, 42, 123, 1234, 12345 |
| Interruption | rerun | rerun starts clean (old `<tag>_*` files deleted); `--resume` opts in to keep finished seeds of the same flags; per-seed timing in the results CSV |
| Loss | cross entropy | cross entropy, class-weighted CE, focal (γ = 2, weighted) |
| Test evaluation | once per seed, best checkpoint | same |

Each seed fixes initialisation and sampling; GPU scatter kernels are not
bit-deterministic, so seeds give repeatable *experiments*, not identical
weights. The seed list is identical across models and loss variants because
the seed-level statistical tests pair rows by seed.

### 4.1 Metrics

Per split and epoch (`src/common/evaluation_metrics.py`): loss, accuracy,
macro precision / recall / F1, balanced accuracy. On the final checkpoint
only: one-vs-rest macro ROC AUC and average precision, multiclass Brier
score, and top-label expected calibration error (10 bins). Summaries report
the mean over seeds with a t-based 95 % confidence interval.

Embedding quality: cosine silhouette of `encode` outputs against the true
labels, on all nodes and on test nodes.

### 4.2 Statistical comparison (`compare_best.py`)

- **Node level.** The best-seed checkpoint of each model predicts the test
  nodes; Cochran's Q tests whether the models' per-node correctness differs,
  then pairwise exact McNemar tests with Holm correction.
- **Seed level.** Friedman and Kruskal-Wallis across models on the per-seed
  score, then pairwise Wilcoxon with Holm correction. With five seeds the
  exact Wilcoxon floor is p = 0.0625, so these are descriptive on Cora; with
  ten seeds on arxiv the floor is 0.002.

### 4.3 Explainability (`explain_best.py`)

GNNExplainer on test nodes of the best model, reporting fidelity+,
fidelity−, unfaithfulness and the characterisation score, aggregate feature
importance, and one worked example per confidence group (high-confidence
correct, high-confidence incorrect, low-confidence correct).

### 4.4 Hyperparameter search (`optuna_search.py`)

TPE sampler with a median pruner, one seed per trial, full batch, optimising
a validation metric read at the best-validation-loss epoch (the same rule as
training). Search space: layers 2 to 3, hidden 128 to 512, dropout 0.1 to
0.6, lr 1e-3 to 3e-2 (log), Adam or AdamW, weight decay 0 to 1e-2, heads
for GATv2. The winner is confirmed over the full seed list with `ogb_run.py`;
the tuned seed's own score is never reported.

---

## 5. Results: Cora

Five seeds, test split of 1,000 nodes, mean with 95 % CI. Full tables and
per-model hyperparameters: `docs/cora_results.md`.

| Model | Accuracy | Macro F1 | ROC AUC | ECE |
|---|---|---|---|---|
| GCN | 0.791 [0.772, 0.810] | 0.782 [0.762, 0.802] | 0.952 | 0.053 |
| GConv | 0.778 [0.753, 0.804] | 0.771 [0.745, 0.797] | 0.943 | 0.118 |
| **GATv2** | **0.813 [0.801, 0.825]** | **0.802 [0.789, 0.815]** | **0.966** | 0.046 |
| GraphSAGE | 0.803 [0.796, 0.810] | 0.795 [0.787, 0.804] | 0.966 | **0.044** |

**Comparison.** Cochran's Q rejects equal accuracy across the four
best-seed checkpoints (p = 0.004). After Holm correction, GATv2 beats GCN
and GConv on paired test nodes (p = 0.012 each); GATv2 vs GraphSAGE and
GraphSAGE vs the two others are not separable (p > 0.2). At the seed level
Friedman on macro F1 gives p = 0.021 but no pairwise Wilcoxon can pass the
five-seed floor. Reading: GATv2 and GraphSAGE form a top pair, GCN and
GConv a lower pair, and the sum-aggregating GConv is the least calibrated
(ECE 0.118).

**Embeddings.** GATv2 cosine silhouette 0.435 on all nodes, 0.424 on test
nodes: classes are separated but overlap, consistent with 81 % accuracy.

**Explainability (GATv2, 30 test nodes).** Fidelity+ 0.40 ± 0.50,
fidelity− 0.00, unfaithfulness 0.40 ± 0.18, characterisation 0.40. Removing
the explanation subgraph flips the prediction for 40 % of nodes; keeping only
the explanation never does. Explanations are sufficient more often than
they are necessary.

*Figure placeholders*

- `![GATv2 training curves, all seeds](figures/cora/gatv2_training_curve.svg)` **TODO copy**
- `![GATv2 t-SNE of encode embeddings](figures/cora/gatv2_tsne_embedding.svg)` **TODO copy**
- `![GATv2 aggregate feature importance](figures/cora/gatv2_xai_aggregate_importance.svg)` **TODO copy**
- `![Explanation subgraph, high-confidence correct node](figures/cora/gatv2_xai_high_confidence_correct_node2220_subgraph.png)` **TODO copy**
- **TODO** per-seed accuracy strip plot across models (from `*_results.csv`)
- **TODO** McNemar contingency heatmap (from `comparison_mcnemar.csv`)

---

## 6. Results: ogbn-arxiv

### 6.1 Experiment grid

The plan below (runs A-D) was the starting point; the grid that actually
finished differs from it in two ways worth stating up front, since later
subsections report the finished numbers, not this plan. First, every SAGE
and SAGEBN tag ended up trained at `weight_decay=0`: a `weight_decay=5e-4`
pilot (6.2, 6.3) gave comparable results, and the zero-decay runs
superseded it as the tags actually kept. Second, GATv2's hyperparameters
were tuned separately rather than held to the shared baseline: `lr=0.001`,
`dropout=0.6` (baseline: `lr=0.01`, `dropout=0.5`), chosen from published
attention-network configurations rather than the SAGE/SAGEBN sweep, so
GATv2 is not a controlled ablation of the other two, only width and heads
were meant to be its sole difference at the planning stage.

Ten seeds each, identical flags apart from the variable under study.
Baseline flags follow the OGB reference for ogbn-arxiv: 3 layers, 256
hidden (GATv2: 8 heads × 32, the same total width), dropout 0.5, constant
lr 0.01 with no scheduler, 500 epochs, ten runs; early stop disabled
(`--early_stop 501`, so every seed trains the full budget), full batch,
undirected graph.

| Run | Model | Loss | Hidden | Weight decay | Status |
|---|---|---|---|---|---|
| A1 | GraphSAGE | cross entropy | 256 | 0 | done |
| A2 | GraphSAGE | weighted CE, untempered | 256 | 0 | done |
| A2b | GraphSAGE | weighted CE, tempered (p = 0.5) | 256 | 0 | done |
| A3 | GraphSAGE | focal | – | – | dropped after pilot (6.3) |
| B | GATv2 (8 heads × 32) | CE, tempered weighted CE | 256 (32/head) | 5e-4 | done, `lr=0.001` `dropout=0.6`, not fully converged (6.4) |
| C | GraphSAGE-BN | CE, tempered weighted CE | 256 | 0 | done |
| E1 | GraphSAGE | cross entropy (width ablation) | 128 | 0 | done |
| E2 | GraphSAGE-BN | cross entropy (width ablation) | 128 | 0 | done |
| D | Optuna on best model | best loss from A | – | – | not started (6.5) |

GCN and GConv were dropped from the arxiv grid in September 2026 to fit the
compute budget; B and C run under all three losses rather than waiting for
A. GATv2 uses 8 heads × 32 channels (256 total, the SAGE width) because
`GATv2Conv` materialises an `[edges, heads, hidden]` tensor per layer; this
needed an A100 (a CUDA out-of-memory error confirmed it on an L4). At
8 heads × 16 (128 total) it fits a T4 or L4, at reduced capacity. Runs are
split between the terminal (`scripts/train_arxiv_all.sh`) and Colab
(`notebooks/04_arxiv_run_node_classification.ipynb`), which backs up to
Drive after every run and skips finished tags.

`ogb_run.py` tags every output with `<model>_<loss>`, plus `_p<power>` when
the class weights are tempered and `_h<width>` for SAGE/SAGEBN off the
usual 256 (the E1/E2 width ablation), so the loss and width variants of one
model coexist in the per-dataset directories and nothing is moved by hand.
The post-training scripts read the same tags; `--model SAGE` alone resolves
when one loss variant exists, `--loss` picks among several.

**Final results, mean over ten seeds** (`outputs/metrics/ogbn-arxiv/<tag>_summary.csv`):

| Tag | acc | f1 | balanced accuracy | ECE | test loss |
|---|---|---|---|---|---|
| sage_cross_entropy | 0.6530 | 0.3363 | 0.3314 | 0.0295 | 1.1716 |
| sage_weighted_ce_p0.5 | 0.6491 | 0.4093 | 0.4254 | 0.0660 | 1.4494 |
| sage_weighted_ce | 0.5814 | 0.4280 | 0.5333 | 0.0504 | 1.5002 |
| sage_cross_entropy_h128 | 0.6175 | 0.2732 | 0.2861 | 0.0591 | 1.3349 |
| sagebn_cross_entropy | 0.7014 | 0.4734 | 0.4661 | 0.0324 | 0.9753 |
| sagebn_weighted_ce_p0.5 | 0.6795 | 0.4961 | 0.5278 | 0.0257 | 1.2252 |
| sagebn_cross_entropy_h128 | 0.7007 | 0.4636 | 0.4580 | 0.0365 | 0.9761 |
| gatv2_cross_entropy | 0.6340 | 0.2621 | 0.2691 | 0.0860 | 1.3044 |
| gatv2_weighted_ce_p0.5 | 0.6454 | 0.3691 | 0.3714 | 0.1929 | 1.6265 |

Balanced accuracy, not accuracy, is the metric that matters for choosing
among these: ogbn-arxiv's 40 classes are heavily imbalanced (train split
21 to 16,284 nodes per class, 775×), so a model can post a respectable
accuracy while doing badly on most of the class space. `sagebn_cross_entropy`
sits at accuracy 0.7014 with only 0.4661 balanced accuracy, close to
Node2vec's 0.7007 on the OGB leaderboard (rank 70/74, a shallow random-walk
embedding with no message passing), while every proper GNN entry above it
outperforms this project's unweighted baselines on accuracy. Unweighted
training here is not using the graph's signal much better than a decade-old
embedding method once class imbalance is accounted for; see 6.3.

### 6.2 Findings so far (single-seed diagnostics, validation split)

These runs were stopped early and are not reported as results; they are
recorded because they fixed the protocol.

| Change | Val acc at ~epoch 150 | Lesson |
|---|---|---|
| directed graph, dropout 0.5, wd 5e-4 | 0.40 | 64 % of test nodes had no in-edges; symmetrise the graph |
| undirected, dropout 0.5, wd 5e-4, early stop 12 | 0.64 (stopped at 172) | 12 full-batch epochs is 12 gradient steps; too short a leash |
| undirected, dropout 0.3, wd 5e-4 | 0.64, loss 1.17 | lower input dropout helps the dense features |
| undirected, dropout 0.3, wd 0 | 0.55, loss 1.59 | without BatchNorm, weight decay is what keeps lr 0.01 stable; keep it |
| old scheduler (÷10, no floor) | LR collapsed to 1e-8 | scheduler now halves and floors at lr/1000 |

Reference point: the OGB example GraphSAGE (BatchNorm, 3 × 256, lr 0.01, no
decay, 500 full-batch epochs, directed graph) reports 0.715 test accuracy.

### 6.3 Loss comparison

**Pilot (10 September 2026).** GraphSAGE, seed 42, 100 epochs, on the Mac,
other flags as A1. Files: `outputs/metrics/ogbn-arxiv/sage_weighted_ce_pilot100_*`
(256 hidden); the 128-hidden pilot's files were overwritten and only its
console log survives. Validation split at epoch 100, test at the
best-validation-loss epoch; minutes are per 100 epochs on the Mac's MPS
backend while another run shared the GPU:

| Loss | Hidden | Val acc | Val macro F1 | Test acc | Test macro F1 | Min / 100 ep |
|---|---|---|---|---|---|---|
| Cross entropy (A1, same seed, epoch 100) | 256 | 0.581 | 0.209 | – | – | ~12 |
| Weighted CE | 256 | 0.560 | 0.397 | 0.557 | 0.386 | 12.8 |
| Weighted CE | 128 | 0.566 | 0.417 | – | – | 6.0 |

Weighted CE nearly doubles macro F1 for 0.02 of accuracy at the same width.
Halving the width cost nothing at epoch 100 and halved the time, which is
the case for running the grid at 128 (open question, see 6.1). Focal loss
(γ = 2, class-weighted, 128 hidden) was started under the same flags and
stopped before its first seed finished: it tracked weighted CE at epoch 20
(val F1 0.122 vs 0.134) but ran at well under half the speed per epoch on
MPS, so it was dropped from the grid. Weighted CE already carries the
imbalance question at no extra cost. Focal remains future work, ideally on
a CUDA machine where the slowdown may not apply.

**Full run, first seed (10 September 2026, afternoon).** The pilot's lead
does not survive 500 epochs. SAGE at 256 with balanced weighted CE, seed 0,
epoch 370 on the Mac: validation accuracy 0.590, macro F1 0.442, against
A1's ten-seed average of 0.680 and 0.421 at the end of training. Nine
points of accuracy for two of F1. Cross entropy catches up on the rare
classes late in training while the weighting keeps paying its accuracy
cost. The cause is the weight spread: the arxiv training split runs from
21 to 16,284 nodes per class (775×), and the balanced formula copies that
spread into the loss (0.14 to 108), so the model over-predicts rare
classes. `--weight_power 0.5` tempers the weights to their square root
(spread 28×, mean weight per sample kept at 1); outputs carry the tag
suffix `_p0.5`. Plan: keep the balanced run as the over-correction row,
run the tempered variant on SAGE, and use the tempered variant as the
weighted loss for GATv2 and SAGEBN.

**Full grid, ten seeds each** (values in 6.1's table). Untempered weighted
CE reaches the best balanced accuracy of the whole grid (0.5333) but gives
up the most accuracy (0.5814) and calibration (ECE 0.0660 tempered →
0.0504 untempered, non-monotonic; test loss worsens to 1.5002), the
over-correction this comparison was designed to show. Tempered weighting
(p = 0.5) is the more usable middle ground on plain SAGE: 0.4254 balanced
accuracy for a much smaller accuracy cost (0.6491).

Node-level test: best-seed checkpoints of the three SAGE loss variants,
paired on the 48,603 test nodes (`compare_best.py --dataset ogbn-arxiv
--tags sage_cross_entropy sage_weighted_ce_p0.5 sage_weighted_ce --label
sage_losses`, full output in `docs/ogbn-arxiv_model_comparison_sage_losses.md`).
Cochran's Q = 2173.1 (df 2, p < 0.001); every pairwise McNemar is
significant after Holm correction (p_holm < 0.001 for all three pairs).
Seed-level (ten seeds, not Cora's five, so the Wilcoxon floor does not
apply here): Friedman p < 0.001 for both macro F1 and balanced accuracy;
every pairwise Wilcoxon significant after Holm correction.

### 6.4 Model comparison

**Architecture, cross entropy only** (`compare_best.py --dataset ogbn-arxiv
--tags sage_cross_entropy sagebn_cross_entropy gatv2_cross_entropy --label
architectures`, full output in `docs/ogbn-arxiv_model_comparison_architectures.md`).
Node-level: Cochran's Q = 1975.3 (df 2, p < 0.001); every pairwise McNemar
significant after Holm correction. Seed-level: Friedman p < 0.001 for macro
F1 and balanced accuracy; every pairwise Wilcoxon significant. SAGEBN beats
plain SAGE at every matched comparison in 6.1's table (CE at 256: 0.4661 vs
0.3314 balanced accuracy; CE at 128: 0.4580 vs 0.2861; tempered weighted at
256: 0.5278 vs 0.4254), so BatchNorm is doing real work here, not just
stabilising optimisation. GATv2 trails both SAGE variants at both losses,
but see 6.1's caveat: its hyperparameters were tuned separately and its
best epochs sit at 495-498 of the 500-epoch budget (seed 12345, cross
entropy), so validation loss had not fully plateaued; some of the gap may
be under-training rather than a real architectural disadvantage.

**GraphSAGE vs GraphSAGE-BN, the normalisation question.**
`sagebn_weighted_ce_p0.5` (tempered weighting + BatchNorm) reaches balanced
accuracy 0.5278, within 0.0055 of the grid's best (0.5333, untempered
weighting on plain SAGE, which sacrifices accuracy and calibration to get
there), while keeping accuracy at 0.6795 and posting the best calibration
in the grid (ECE 0.0257). BatchNorm is substituting for a large share of
what aggressive loss weighting would otherwise need to do: a gentler
correction on a normalised architecture reaches nearly the same
balanced-accuracy outcome as the strongest correction on an unnormalised
one, at a much lower cost to overall accuracy and calibration.

**Embeddings.** `visualise_best.py --dataset ogbn-arxiv --model SAGEBN
--loss weighted_ce_p0.5` (best seed 3, `sagebn_weighted_ce_p0.5`, the
model-comparison candidate above): cosine silhouette 0.1156 over all
169,343 nodes, 0.0924 on the 48,603 test nodes. Figures in
`outputs/figures/ogbn-arxiv/sagebn_weighted_ce_p0.5_training_curve.{html,svg}`
and `..._tsne_embedding.{html,svg}`; not yet copied to `docs/figures/`.

### 6.5 Hyperparameter search

**Not started.** `src/node_classification/optuna_search.py` already exists
for this (`uv run python -m src.node_classification.optuna_search --dataset
ogbn-arxiv --model SAGE --n_trials 30`); GATv2's lr/dropout/width were
tuned by hand instead against published attention-network configurations
(6.1), which is exactly the kind of manual search this script should
replace. Before running it for GATv2, confirm `suggest_params` and its
`--heads` handling cover that model, it was written with SAGE in mind.
**TODO**: best configuration per model, its confirmation over ten seeds,
and Optuna parameter-importance and optimisation-history plots.

*Figure placeholders*

- Training curve and t-SNE embedding for `sagebn_weighted_ce_p0.5` are
  generated (see 6.4); copy from `outputs/figures/ogbn-arxiv/` to
  `docs/figures/ogbn-arxiv/` when ready to link them here.
- **TODO** per-class F1 bar chart, plain vs weighted CE (shows where the F1 gain comes from)
- **TODO** reliability diagram from the calibration bins
- **TODO** Optuna `plot_param_importances` and `plot_optimization_history`

---

## 7. Reproducibility

```
# Cora: all four models with the reported hyperparameters, then the tables
bash scripts/train_cora_all.sh
uv run python scripts/cora_results_md.py
uv run python src/node_classification/compare_best.py --dataset Cora
uv run python src/node_classification/visualise_best.py --dataset Cora --metric f1
uv run python src/node_classification/explain_best.py --dataset Cora --metric f1

# ogbn-arxiv grid: SAGE (A1/A2/A2b), SAGEBN (C), GATv2 (B), the 128-hidden
# width ablation (E1/E2) -- all done, see 6.1. scripts/train_arxiv_all.sh only
# has A1 uncommented; the finished grid ran via the Colab notebook instead.
# A1 (SAGE, cross entropy), the finished config:
uv run python -m src.node_classification.ogb_run --model SAGE --loss cross_entropy \
    --num_layers 3 --lr 0.01 --hidden_channels 256 --dropout 0.5 --weight_decay 0 \
    --no-dataloader --no-scheduler --epochs 500 --early_stop 501
# GATv2 (B), tuned separately from the above, not a controlled ablation of it:
uv run python -m src.node_classification.ogb_run --model GATV2 --loss cross_entropy \
    --num_layers 3 --heads 8 --hidden_channels 32 --dropout 0.6 --lr 0.001 \
    --no-dataloader --no-scheduler --epochs 500 --early_stop 501

# Post-training scripts, ogbn-arxiv (see 6.4 for the actual runs and results)
uv run python src/node_classification/compare_best.py --dataset ogbn-arxiv \
    --tags sage_cross_entropy sage_weighted_ce_p0.5 sage_weighted_ce --label sage_losses
uv run python src/node_classification/compare_best.py --dataset ogbn-arxiv \
    --tags sage_cross_entropy sagebn_cross_entropy gatv2_cross_entropy --label architectures
uv run python src/node_classification/visualise_best.py --dataset ogbn-arxiv \
    --model SAGEBN --loss weighted_ce_p0.5
# explain_best.py is not yet updated for OGB (still Planetoid-only); on a
# large dataset --max_samples must be set explicitly regardless (6.5/8).

# Hyperparameter search (D), not started (6.5)
uv run python -m src.node_classification.optuna_search --dataset ogbn-arxiv --model SAGE --n_trials 30

# Tests
uv run --with pytest python -m pytest test/node_models_test.py test/utils_loader_test.py test/optuna_search_test.py
```

Environment: Python 3.12, PyTorch 2.13, PyG 2.8, `uv` lockfile pinned to
macOS arm64 (pyg-lib wheel). Training so far on Apple silicon (MPS).

---

## 8. Open items

- ~~`compare_best.py` on arxiv currently compares every `<model>_<loss>` tag
  found; a loss-level and a model-level comparison (6.3 and 6.4) need a
  filter on the tag.~~ Done: `compare_best.py` and `visualise_best.py` now
  dispatch to `load_ogb_node` for OGB datasets (`best_model.load_dataset`),
  and `compare_best.py` takes `--tags` (which tags to compare) and `--label`
  (output filename suffix), used for the 6.3/6.4 loss-level and
  model-level comparisons.
- Eval-mode train accuracy in the training log (currently measured with
  dropout active, which understates it).
- AdamW as an alternative to Adam L2 when decay is needed; flag exists,
  untested.
- ogbn-products with the NeighborLoader and layer-wise inference, once the
  arxiv protocol is settled.
- Custom circuit dataset: graph construction and the two prediction tasks.
- The `readme.md` tree predates the current layout and should be regenerated.
- **Temporal / inductive framing of ogbn-arxiv.** The standard split (train on
  papers through 2017, validate on 2018, test from 2019) is transductive, not
  inductive: `load_ogb_node` loads one static graph, and full-batch training
  passes messages over the whole citation network, test-year edges and
  features included, on every forward pass. Only the loss is time-restricted.
  A genuinely inductive setup would train on the train-year subgraph only,
  then add the validation-year and finally the test-year nodes and edges
  incrementally, classifying each using only the graph as it existed before
  that node arrived, closer to GraphSAGE's original inductive framing than to
  how it's used here. That needs temporal graph snapshots `load_ogb_node`
  doesn't build, and points toward dynamic/temporal GNN architectures
  (EvolveGCN, DySAT, Temporal Graph Networks) rather than a config change to
  the current models. `GraphSAGEBN` in the current grid is a step in that
  direction already: BatchNorm between layers helps stabilise training under
  the feature-distribution drift across publication years that an
  incremental curriculum would make explicit, even though today's grid
  trains on one static snapshot. Not started; would need a new data pipeline
  before any model change. For scale, the OGB leaderboard's current top
  entry, SimTeG+TAPE+RevGAT (uses external data, LLM-derived text features),
  reports validation accuracy 0.7846 ± 0.0004 and test accuracy 0.7803 ± 0.0007
  on the standard transductive split; this project's grid uses no external
  data and is not aiming to compete with that number, it's a reference point
  for where full-batch GraphSAGE/GATv2 with class-weighted losses sit in the
  field.
