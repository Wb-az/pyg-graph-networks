"""Smoke tests for optuna_search on a small synthetic graph (no dataset download)."""
import json

import pytest
import torch
from torch_geometric.data import Data

from src.node_classification.optuna_search import (ogb_run_command, run_study,
                                                   suggest_params)

N, F, C = 200, 6, 3


@pytest.fixture
def graph():
    torch.manual_seed(0)
    edge_index = torch.randint(0, N, (2, 1600))
    y = torch.randint(0, C, (N,))
    data = Data(x=torch.randn(N, F), edge_index=edge_index, y=y)
    perm = torch.randperm(N)
    for name, idx in (("train_mask", perm[:120]), ("val_mask", perm[120:160]),
                      ("test_mask", perm[160:])):
        mask = torch.zeros(N, dtype=torch.bool)
        mask[idx] = True
        setattr(data, name, mask)
    return data


@pytest.mark.parametrize("model", ["SAGE", "GATV2", "SAGEBN"])
def test_study_runs_and_writes_outputs(graph, tmp_path, model):
    study = run_study(dataset="synthetic", model=model, n_trials=2, epochs=3,
                      early_stop=3, persist=False, prune=False, data=graph, root=tmp_path)
    assert len(study.trials) == 2
    assert study.best_trial.value is not None
    assert "epoch" in study.best_trial.user_attrs
    out = tmp_path / "outputs" / "metrics" / "synthetic"
    assert (out / f"optuna_{model.lower()}_trials.csv").exists()
    best = json.loads((out / f"optuna_{model.lower()}_best.json").read_text())
    assert best["params"] == study.best_params
    assert best["ogb_run_command"].startswith("uv run python -m src.node_classification.ogb_run")


def test_two_seeds_average(graph, tmp_path):
    study = run_study(dataset="synthetic", model="SAGE", n_trials=1, seeds=(0, 1), epochs=2,
                      persist=False, prune=False, data=graph, root=tmp_path)
    assert len(study.trials) == 1


def test_loss_metric_is_minimised(graph, tmp_path):
    study = run_study(dataset="synthetic", model="SAGE", n_trials=2, epochs=2, metric="loss",
                      persist=False, prune=False, data=graph, root=tmp_path)
    assert study.direction.name == "MINIMIZE"


def test_command_covers_every_searched_param():
    import optuna
    study = optuna.create_study()
    trial = study.ask()
    params = suggest_params(trial, "GATV2")
    cmd = ogb_run_command(params, "ogbn-arxiv", "GATV2", "focal", 2.0)
    for flag in ("--num_layers", "--hidden_channels", "--dropout", "--lr", "--optimizer",
                 "--weight_decay", "--heads", "--gamma", "--no-dataloader"):
        assert flag in cmd
