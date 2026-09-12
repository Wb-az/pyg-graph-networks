"""
Optuna hyperparameter search for one node-classification model.

Each trial trains the model full batch on one or more seeds with sampled
hyperparameters, selects the epoch with the lowest validation loss (the same
rule as ogb_run.py) and returns the validation metric at that epoch, averaged
over the seeds. Unpromising trials are pruned with Optuna's median pruner.
The study is stored in SQLite under ``outputs/optuna`` so a search can be
interrupted and resumed, and the best configuration is written next to the
training metrics as ``optuna_<model>_best.json`` together with the equivalent
``ogb_run.py`` command line.

Command line::

    uv run python -m src.node_classification.optuna_search \\
        --dataset ogbn-arxiv --model SAGE --n_trials 30 --loss weighted_ce

Notebook (local kernel, from the project root)::

    from src.node_classification.optuna_search import run_study
    study = run_study(dataset="ogbn-arxiv", model="SAGE", n_trials=30)
    study.best_params
    study.trials_dataframe().sort_values("value", ascending=False).head()

Colab::

    !git clone <repo-url> GraphNetworks && cd GraphNetworks
    %cd GraphNetworks
    !pip install -q torch_geometric ogb optuna pytorch-focalloss
    from src.node_classification.optuna_search import run_study
    study = run_study(dataset="ogbn-arxiv", model="SAGE", n_trials=30, seeds=(0,))

Only full-batch training is supported (ogbn-arxiv fits on any GPU and on
Apple silicon); use ogb_run.py with the NeighborLoader for ogbn-products.
Edit ``suggest_params`` to change the search space.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):  # run as a file (%run or python path/to/file.py)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import copy

import optuna
import torch

from src.common.datasets import load_ogb_node, load_planetoid
from src.common.paths import get_project_root
from src.common.utils import evaluate, get_device, set_seed, train_one_epoch
from src.node_classification.ogb_run import build_criterion, build_model

MODELS = ("GCN", "GConv", "GATV2", "SAGE", "SAGEBN")
METRICS = {"f1": "maximize", "acc": "maximize", "balanced_accuracy": "maximize",
           "loss": "minimize"}
OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}


def suggest_params(trial: optuna.Trial, model: str) -> dict:
    """Search space. Everything here maps onto an ogb_run.py flag.

    num_layers and hidden_channels are searched, not fixed: an earlier
    narrower version fixed both at the finished grid's values (2 layers, 256
    hidden), but a broader search found 3 layers/512 hidden meaningfully
    better (val_balanced_accuracy 0.5192 vs 0.5014), so capacity matters more
    than that assumption held. dropout is reduced to 3 values (see below);
    that narrowing held up and stays. weight_decay is on one shared log
    range: Optuna requires one parameter name to keep the same distribution
    across every trial in a study, so it can't switch between a fixed 0.0 for
    Adam and a fixed 0.01 for AdamW depending on the trial's own optimizer
    choice. The range spans Adam's effective no-decay preference (near the
    1e-6 floor) and AdamW's own PyTorch default of 0.01 (comfortably inside
    the range), letting the sampler find what each optimizer actually prefers.
    """
    params = {
        "num_layers": trial.suggest_int("num_layers", 2, 3),
        "hidden_channels": trial.suggest_categorical("hidden_channels", [128, 256, 512]),
        # 0.5 is BASE's default (the whole finished SAGE/SAGEBN grid), 0.6 is
        # what GATv2 used, 0.3 is what the early 6.2 diagnostics found helped
        # the dense input features before the grid settled on 0.5.
        "dropout": trial.suggest_categorical("dropout", [0.3, 0.5, 0.6]),
        "lr": trial.suggest_float("lr", 1e-3, 3e-2, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
    }
    if model == "GATV2":
        params["heads"] = trial.suggest_categorical("heads", [2, 4, 8])
    return params


def load_data(dataset: str, root: Path):
    """Return (data, num_features, num_classes) for an OGB or Planetoid dataset."""
    if dataset.lower().startswith("ogbn-"):
        ds, data, _ = load_ogb_node(name=dataset, root=root / "data")
    else:
        ds, data = load_planetoid(root / "data", dataset)
    return data, ds.num_node_features, ds.num_classes


def make_config(params: dict, model: str, loss: str, gamma: float) -> SimpleNamespace:
    """Namespace with the fields build_model / build_criterion read from args."""
    return SimpleNamespace(model=model, loss=loss, gamma=gamma, heads=params.get("heads", 8),
                           **{k: v for k, v in params.items() if k != "heads"})


def train_trial(config, data_dev, num_feat, num_class, device, seed: int,
                epochs: int, early_stop: int, metric: str,
                trial: optuna.Trial | None = None) -> dict:
    """Train one seed full batch; return validation metrics at the best-val-loss epoch.

    Reports the running metric to Optuna every epoch when ``trial`` is given,
    so the pruner can stop hopeless configurations early.
    """
    set_seed(seed)
    model = build_model(config, num_feat, num_class).to(device)
    criterion = build_criterion(config, data_dev, num_class, device)
    optimizer = OPTIMIZERS[config.optimizer](model.parameters(), lr=config.lr,
                                             weight_decay=config.weight_decay)

    best = {"val_loss": float("inf"), "epoch": 0}
    patience = 0
    for epoch in range(1, epochs + 1):
        train_one_epoch(data_dev, model, criterion, optimizer, data_dev.train_mask)
        val = evaluate(data_dev, model, criterion, data_dev.val_mask)
        if val.loss < best["val_loss"]:
            best = {"val_loss": val.loss, "epoch": epoch, "acc": val.acc, "f1": val.f1,
                    "balanced_accuracy": val.balanced_accuracy, "loss": val.loss}
            patience = 0
        else:
            patience += 1
        if trial is not None:
            trial.report(best[metric], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if patience >= early_stop:
            break
    return best


def make_objective(data, num_feat, num_class, device, *, model, loss, gamma, seeds,
                   epochs, early_stop, metric):
    data_dev = copy.copy(data).to(device)

    def objective(trial: optuna.Trial) -> float:
        config = make_config(suggest_params(trial, model), model, loss, gamma)
        results = []
        for i, seed in enumerate(seeds):
            # Only the first seed reports to the pruner; later seeds are the
            # confirmation and would confuse the per-step median otherwise.
            results.append(train_trial(config, data_dev, num_feat, num_class, device, seed,
                                       epochs, early_stop, metric,
                                       trial=trial if i == 0 else None))
        for key in ("epoch", "val_loss", "acc", "f1", "balanced_accuracy"):
            trial.set_user_attr(key, float(sum(r[key] for r in results) / len(results)))
        return float(sum(r[metric] for r in results) / len(results))

    return objective


def ogb_run_command(params: dict, dataset: str, model: str, loss: str, gamma: float) -> str:
    """The ogb_run.py invocation that reproduces a set of parameters."""
    flags = [f"--dataset {dataset}", f"--model {model}", f"--loss {loss}",
             f"--num_layers {params['num_layers']}",
             f"--hidden_channels {params['hidden_channels']}",
             f"--dropout {params['dropout']:.2f}", f"--lr {params['lr']:.2e}",
             f"--optimizer {params['optimizer']}",
             f"--weight_decay {params['weight_decay']:g}", "--no-dataloader"]
    if "heads" in params:
        flags.append(f"--heads {params['heads']}")
    if loss == "focal":
        flags.append(f"--gamma {gamma}")
    return "uv run python -m src.node_classification.ogb_run " + " ".join(flags)


def run_study(dataset: str = "ogbn-arxiv", model: str = "SAGE", n_trials: int = 30,
              seeds=(0,), epochs: int = 300, early_stop: int = 30, metric: str = "f1",
              loss: str = "cross_entropy", gamma: float = 2.0, timeout: float | None = None,
              study_name: str | None = None, persist: bool = True, resume: bool = True,
              prune: bool = True, data=None, root: Path | None = None,
              show_progress_bar: bool = False) -> optuna.Study:
    """Run (or resume) a search and write the best configuration to disk.

    :param seeds: Seeds averaged per trial. One is enough to rank
        configurations; confirm the winner with the full seed list in ogb_run.
    :param metric: Validation metric to optimise; ``loss`` is minimised, the
        others maximised. Always read at the epoch of lowest validation loss.
    :param persist: Store the study in ``outputs/optuna/<study_name>.db``.
        False keeps it in memory (tests, throwaway runs).
    :param data: Preloaded graph, to skip the load in a notebook session.
    :return: The Optuna study; ``study.best_params`` and
        ``study.trials_dataframe()`` are the useful attributes.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {list(METRICS)}, got {metric!r}")

    root = Path(root) if root is not None else get_project_root()
    device = get_device()
    if data is None:
        data, num_feat, num_class = load_data(dataset, root)
    else:
        num_feat, num_class = data.num_node_features, int(data.y.max()) + 1
    print(f"Optuna search: {model} on {dataset}  metric=val_{metric}  loss={loss}  "
          f"seeds={list(seeds)}  device={device}")

    study_name = study_name or f"{dataset.lower()}_{model.lower()}_{loss}_{metric}"
    storage = None
    if persist:
        db_dir = root / "outputs" / "optuna"
        db_dir.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db_dir / study_name}.db"
    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=resume,
        direction=METRICS[metric], sampler=optuna.samplers.TPESampler(seed=0),
        pruner=(optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
                if prune else optuna.pruners.NopPruner()))
    if resume and len(study.trials):
        print(f"Resuming {study_name}: {len(study.trials)} trials already stored")

    start = time.time()
    study.optimize(
        make_objective(data, num_feat, num_class, device, model=model, loss=loss,
                       gamma=gamma, seeds=tuple(seeds), epochs=epochs,
                       early_stop=early_stop, metric=metric),
        n_trials=n_trials, timeout=timeout, show_progress_bar=show_progress_bar,
        gc_after_trial=True)
    elapsed = (time.time() - start) / 60

    best = study.best_trial
    command = ogb_run_command(best.params, dataset, model, loss, gamma)
    print(f"\nBest of {len(study.trials)} trials ({elapsed:.1f} min): "
          f"val_{metric}={best.value:.4f} at epoch {best.user_attrs.get('epoch', 0):.0f}")
    for key, value in best.params.items():
        print(f"  {key}: {value}")
    print(f"\nReproduce over all seeds with:\n  {command}")

    out_dir = root / "outputs" / "metrics" / dataset.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out_dir / f"optuna_{model.lower()}_trials.csv", index=False)
    with open(out_dir / f"optuna_{model.lower()}_best.json", "w") as fh:
        json.dump({"study": study_name, "metric": f"val_{metric}", "value": best.value,
                   "params": best.params, "user_attrs": best.user_attrs,
                   "ogb_run_command": command}, fh, indent=2)
    return study


def main(argv=None):
    parser = argparse.ArgumentParser(description="Optuna search for node classification")
    parser.add_argument("--dataset", default="ogbn-arxiv",
                        help="ogbn-arxiv, Cora, CiteSeer, PubMed (full batch only)")
    parser.add_argument("--model", choices=MODELS, default="SAGE")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=None,
                        help="stop after this many seconds even if trials remain")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="seeds averaged per trial (one is enough to rank configs)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--early_stop", type=int, default=30)
    parser.add_argument("--metric", choices=list(METRICS), default="f1",
                        help="validation metric to optimise, read at the best-val-loss epoch")
    parser.add_argument("--loss", choices=["cross_entropy", "weighted_ce", "focal"],
                        default="cross_entropy")
    parser.add_argument("--gamma", type=float, default=2.0, help="focal loss exponent")
    parser.add_argument("--study_name", default=None,
                        help="default <dataset>_<model>_<loss>_<metric>")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="continue a stored study with the same name")
    parser.add_argument("--prune", action=argparse.BooleanOptionalAction, default=True,
                        help="median pruning after 20 epochs (off: every trial runs to early stop)")
    args = parser.parse_args(argv)
    run_study(dataset=args.dataset, model=args.model, n_trials=args.n_trials,
              seeds=tuple(args.seeds), epochs=args.epochs, early_stop=args.early_stop,
              metric=args.metric, loss=args.loss, gamma=args.gamma, timeout=args.timeout,
              study_name=args.study_name, resume=args.resume, prune=args.prune)


if __name__ == "__main__":
    main()
