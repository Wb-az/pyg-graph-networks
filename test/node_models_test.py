"""Guards the shared layer recipe documented in node_models.py and CLAUDE.md."""
import torch
import torch.nn as nn
import pytest

from src.node_classification.node_models import GCN, GConv, GATV2, GraphSAGE

MODELS = [
    (GCN, {}),
    (GConv, {}),
    (GATV2, {"heads": 2}),
    (GraphSAGE, {}),
]


def build(cls, extra):
    torch.manual_seed(0)
    return cls(num_layers=2, in_feat=5, hid_feat=8, num_classes=3,
               dropout=0.5, **extra)


@pytest.fixture
def graph():
    torch.manual_seed(0)
    x = torch.randn(10, 5)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    return x, edge_index


@pytest.mark.parametrize("cls,extra", MODELS)
def test_common_interface(cls, extra):
    model = build(cls, extra)
    assert isinstance(model.layers, nn.ModuleList) and len(model.layers) == 2
    assert isinstance(model.lin1, nn.Module)
    assert callable(model.encode) and callable(model.forward)


@pytest.mark.parametrize("cls,extra", MODELS)
def test_forward_is_head_over_encode(cls, extra, graph):
    x, ei = graph
    model = build(cls, extra).eval()
    emb = model.encode(x, ei)
    assert emb.shape == (10, 8)
    assert torch.allclose(model(x, ei), model.lin1(emb))


@pytest.mark.parametrize("cls,extra", MODELS)
def test_encode_is_deterministic_in_eval(cls, extra, graph):
    x, ei = graph
    model = build(cls, extra).eval()
    assert torch.equal(model.encode(x, ei), model.encode(x, ei))


def _call_order(model, fn, x, ei):
    """Record the order in which dropout, activation and conv layers are called."""
    order = []
    handles = [
        model.dropout.register_forward_hook(lambda *_: order.append("dropout")),
        model.activation.register_forward_hook(lambda *_: order.append("act")),
        *[layer.register_forward_hook(lambda *_, i=i: order.append(f"conv{i}"))
          for i, layer in enumerate(model.layers)],
        model.lin1.register_forward_hook(lambda *_: order.append("head")),
    ]
    with torch.no_grad():
        fn(x, ei)
    for h in handles:
        h.remove()
    return order


@pytest.mark.parametrize("cls,extra", MODELS)
def test_encode_layer_recipe(cls, extra, graph):
    """encode is dropout -> conv -> activation per layer and never ends with dropout."""
    x, ei = graph
    model = build(cls, extra).train()
    order = _call_order(model, model.encode, x, ei)
    expected = []
    for i in range(len(model.layers)):
        expected += ["dropout", f"conv{i}", "act"]
    assert order == expected


@pytest.mark.parametrize("cls,extra", MODELS)
def test_forward_layer_recipe(cls, extra, graph):
    """forward is encode -> dropout -> head."""
    x, ei = graph
    model = build(cls, extra).train()
    order = _call_order(model, model.forward, x, ei)
    encode_part = _call_order(model, model.encode, x, ei)
    assert order == encode_part + ["dropout", "head"]


@pytest.mark.parametrize("cls,extra", MODELS)
def test_dropout_active_in_train_mode(cls, extra, graph):
    x, ei = graph
    model = build(cls, extra).train()
    torch.manual_seed(1)
    out1 = model(x, ei)
    torch.manual_seed(2)
    out2 = model(x, ei)
    assert not torch.allclose(out1, out2), "dropout should make train outputs differ"
