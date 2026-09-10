"""Behaviour of create_dataloaders and the loader-based epoch helpers."""
import copy

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.common.utils import (create_dataloaders, evaluate, evaluate_loader,
                              inference_layerwise, train_one_epoch,
                              train_one_epoch_loader)
from src.node_classification.node_models import GATV2, GConv, GraphSAGE, GraphSAGEBN

N, F, C = 300, 6, 3


@pytest.fixture
def graph():
    torch.manual_seed(0)
    edge_index = torch.randint(0, N, (2, 2400))
    # Imbalanced labels: 70% / 20% / 10%.
    y = torch.cat([torch.zeros(210), torch.ones(60), torch.full((30,), 2)]).long()
    y = y[torch.randperm(N)]
    data = Data(x=torch.randn(N, F), edge_index=edge_index, y=y)
    perm = torch.randperm(N)
    for name, idx in (("train_mask", perm[:180]), ("val_mask", perm[180:240]),
                      ("test_mask", perm[240:])):
        mask = torch.zeros(N, dtype=torch.bool)
        mask[idx] = True
        setattr(data, name, mask)
    return data


def sage(seed=0):
    torch.manual_seed(seed)
    return GraphSAGE(num_layers=2, in_feat=F, hid_feat=8, num_classes=C, dropout=0.0)


def test_neighbor_train_loader_visits_every_train_seed_once(graph):
    loaders = create_dataloaders(graph, "neighbor", batch_size=64, num_neighbors=(5, 5))
    assert loaders["val"] is None and loaders["test"] is None
    seen = torch.cat([b.input_id for b in loaders["train"]])
    assert torch.equal(seen.sort().values, graph.train_mask.nonzero().squeeze())
    batch = next(iter(loaders["train"]))
    assert batch.batch_size == 64
    assert torch.equal(batch.y[:batch.batch_size], graph.y[batch.input_id])


def test_int_fanout_expands_to_num_layers(graph):
    loader = create_dataloaders(graph, "neighbor", num_neighbors=7, num_layers=3)["train"]
    assert loader.node_sampler.num_neighbors.values == [7, 7, 7]


def test_balanced_sampler_raises_minority_share(graph):
    torch.manual_seed(1)
    loaders = create_dataloaders(graph, "neighbor", batch_size=64, num_neighbors=(5, 5),
                                 balanced=True)
    y = torch.cat([b.y[:b.batch_size] for b in loaders["train"]])
    assert y.numel() == int(graph.train_mask.sum())
    original = torch.bincount(graph.y[graph.train_mask], minlength=C).float()
    resampled = torch.bincount(y, minlength=C).float()
    assert resampled[2] / resampled.sum() > original[2] / original.sum()


def test_full_loader_yields_whole_graph_once(graph):
    loaders = create_dataloaders(graph, "full", eval_loaders=True)
    batches = list(loaders["train"])
    assert len(batches) == 1 and batches[0].num_nodes == N
    assert loaders["val"] is loaders["train"]


def test_full_loader_epoch_matches_full_batch_training(graph):
    model_a, model_b = sage(), copy.deepcopy(sage())
    criterion = nn.CrossEntropyLoss()
    opt_a = torch.optim.SGD(model_a.parameters(), lr=0.1)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.1)
    loader = create_dataloaders(graph, "full")["train"]

    ref = train_one_epoch(graph, model_a, criterion, opt_a, graph.train_mask)
    got = train_one_epoch_loader(loader, model_b, criterion, opt_b, "cpu")
    assert got.loss == pytest.approx(ref.loss)
    assert got.f1 == pytest.approx(ref.f1)
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(pa, pb)


def test_exact_eval_loader_matches_full_batch_for_sage(graph):
    model = sage()
    criterion = nn.CrossEntropyLoss()
    loaders = create_dataloaders(graph, "neighbor", num_neighbors=(5, 5),
                                 eval_loaders=True, eval_batch_size=50)
    for split in ("val", "test"):
        mask = getattr(graph, f"{split}_mask")
        ref = evaluate(graph, model, criterion, mask, include_auc=True)
        got = evaluate_loader(loaders[split], model, criterion, "cpu",
                              f"{split}_mask", include_auc=True)
        assert got.loss == pytest.approx(ref.loss, rel=1e-5)
        assert got.acc == pytest.approx(ref.acc)
        assert got.roc_auc == pytest.approx(ref.roc_auc, rel=1e-5)


def test_neighbor_epoch_trains(graph):
    model = sage()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loader = create_dataloaders(graph, "neighbor", batch_size=64, num_neighbors=(5, 5))["train"]
    first = train_one_epoch_loader(loader, model, criterion, opt, "cpu")
    for _ in range(20):
        last = train_one_epoch_loader(loader, model, criterion, opt, "cpu")
    assert last.loss < first.loss


def test_unknown_loader_type(graph):
    with pytest.raises(ValueError):
        create_dataloaders(graph, "simple")


@pytest.mark.parametrize("cls,extra", [(GraphSAGE, {}), (GraphSAGEBN, {}), (GATV2, {"heads": 2}),
                                       (GConv, {})])
def test_layerwise_inference_matches_forward(graph, cls, extra):
    torch.manual_seed(0)
    model = cls(num_layers=2, in_feat=F, hid_feat=8, num_classes=C, dropout=0.0, **extra)
    model.eval()
    ref = model(graph.x, graph.edge_index)
    got = inference_layerwise(model, graph, "cpu", batch_size=64)
    assert torch.allclose(got, ref, atol=1e-5)


def test_evaluate_accepts_precomputed_logits(graph):
    model = sage()
    criterion = nn.CrossEntropyLoss()
    ref = evaluate(graph, model, criterion, graph.test_mask, include_auc=True)
    logits = inference_layerwise(model, graph, "cpu", batch_size=64)
    got = evaluate(graph, model, criterion, graph.test_mask, include_auc=True, out=logits)
    assert got.loss == pytest.approx(ref.loss, rel=1e-5)
    assert got.roc_auc == pytest.approx(ref.roc_auc, rel=1e-5)
