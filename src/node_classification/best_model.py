"""Pick the best trained node-classification model and rebuild it from its checkpoint.

Shared by visualise_best.py and explain_best.py. Selection reads the
``<model>_summary.csv`` files written by planetoid_node_classification.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.node_classification.node_models import GCN, GConv, GATV2, GraphSAGE, GraphSAGEBN
from src.common.utils import load_checkpoint

MODEL_CLASSES = {"GCN": GCN, "GConv": GConv, "GATV2": GATV2, "SAGE": GraphSAGE,
                 "SAGEBN": GraphSAGEBN}

# Metrics where a smaller value is better.
LOWER_IS_BETTER = {"loss", "brier", "ece"}

CORA_LABELS = {
    0: "Theory",
    1: "Reinforcement Learning",
    2: "Genetic Algorithms",
    3: "Neural Networks",
    4: "Probabilistic Methods",
    5: "Case Based",
    6: "Rule Learning",
}
LABEL_MAPS = {"Cora": CORA_LABELS}


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    """CLI flags shared by the post-training scripts."""
    parser.add_argument("--dataset", type=str, default="Cora",
                        help="Dataset name: Cora, CiteSeer, PubMed")
    parser.add_argument("--metric", type=str, default="f1",
                        help="Summary metric used to pick the best model "
                             "(f1, balanced_accuracy, acc, roc_auc, loss, ...)")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODEL_CLASSES),
                        help="Skip selection and use this model instead")
    parser.add_argument("--loss", type=str, default=None,
                        help="With --model on OGB runs, which loss variant to use "
                             "(ogb_run.py tags its outputs <model>_<loss>)")


def load_summaries(metrics_dir: Path) -> pd.DataFrame:
    """One row per trained model, one column per summary metric (means)."""
    rows = {}
    for path in sorted(metrics_dir.glob("*_summary.csv")):
        tag = path.name.removesuffix("_summary.csv")
        # Only training runs write a history file; skips e.g. <tag>_xai_summary.csv.
        if not (metrics_dir / f"{tag}_history.csv").exists():
            continue
        summary = pd.read_csv(path, index_col=0)
        rows[tag] = summary["mean"].rename(lambda k: k.removeprefix("Test "))
    if not rows:
        raise FileNotFoundError(f"No *_summary.csv files in {metrics_dir}")
    return pd.DataFrame(rows).T


def select_best_model(metrics_dir: Path, metric: str = "f1") -> tuple[str, pd.DataFrame]:
    """Return the tag (e.g. 'gatv2') of the model with the best mean metric."""
    table = load_summaries(metrics_dir)
    if metric not in table.columns:
        raise ValueError(f"Unknown metric {metric!r}. Choose from {list(table.columns)}")
    ascending = metric in LOWER_IS_BETTER
    tag = table[metric].sort_values(ascending=ascending).index[0]
    return tag, table


def select_best_seed(metrics_dir: Path, tag: str) -> int:
    """Seed whose run reached the lowest validation loss."""
    results = pd.read_csv(metrics_dir / f"{tag}_results.csv")
    return int(results.loc[results["val_loss"].idxmin(), "seed"])


def build_model(config: dict, num_features: int, num_classes: int) -> torch.nn.Module:
    """Instantiate the architecture described by a checkpoint's config dict."""
    cls = MODEL_CLASSES[config["model"]]
    kwargs = dict(num_layers=config["num_layers"], in_feat=num_features,
                  hid_feat=config["hidden_channels"], num_classes=num_classes,
                  dropout=config["dropout"])
    if cls is GATV2:
        kwargs["heads"] = config["heads"]
    return cls(**kwargs)


def load_best_checkpoint(checkpoint_dir: Path, dataset_name: str, tag: str, seed: int,
                         num_features: int, num_classes: int,
                         device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Rebuild the model from its saved config and load the best weights."""
    path = checkpoint_dir / f"{dataset_name.lower()}_{tag}_{seed}_best.pth"
    config = torch.load(path, map_location=device, weights_only=False)["config"]
    model = build_model(config, num_features, num_classes)
    model = load_checkpoint(model, path, device=device, state_key="model")
    return model, config


def resolve_model(args, metrics_dir: Path) -> tuple[str, pd.DataFrame]:
    """Apply --model override or pick the best by --metric; print the table."""
    tag, table = select_best_model(metrics_dir, args.metric)
    print(f"Models trained on {args.dataset} (mean over seeds):")
    print(table.round(4).to_string())
    if args.model is not None:
        tag = args.model.lower()
        loss = getattr(args, "loss", None)
        if loss is not None:
            tag = f"{tag}_{loss}"
        if tag not in table.index:
            # OGB runs are tagged <model>_<loss>; accept a bare model name when
            # only one loss variant of it was trained.
            matches = [t for t in table.index if t.split("_")[0] == tag]
            if len(matches) != 1:
                raise ValueError(f"No run tagged {tag!r} in {metrics_dir}; candidates: "
                                 f"{list(table.index)}. Use --loss to pick one.")
            tag = matches[0]
        print(f"\nUsing model override: {tag}")
    else:
        print(f"\nBest model by {args.metric}: {tag}")
    return tag, table
