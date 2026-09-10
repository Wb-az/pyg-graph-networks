import random
from collections.abc import Sequence

import numpy as np

import torch
from torch_geometric.loader import DataLoader, ImbalancedSampler, NeighborLoader

from src.common.evaluation_metrics import Metrics


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_class_weights(y: torch.Tensor, mask: torch.Tensor | None = None,
                          num_classes: int | None = None) -> torch.Tensor:
    """
    Inverse-frequency class weights for nn.CrossEntropyLoss(weight=...).
    weight[c] = n_samples / (num_classes * count[c]) -- sklearn's
    "balanced" formula. Pass mask=data.train_mask so validation/test class
    balance never leaks into the training loss.

    :param y: Full label tensor (e.g. data.y).
    :param mask: Boolean mask selecting which labels to compute weights
        from (e.g. data.train_mask). Uses all of `y` if None.
    :param num_classes: Number of classes. Inferred from the labels if None
        -- pass it explicitly if a class could be entirely absent from
        `mask` (e.g. a very rare arxiv class), since it can't be inferred
        from labels that never appear.
    """
    labels = y[mask] if mask is not None else y
    num_classes = num_classes or int(labels.max().item()) + 1
    counts = torch.bincount(labels, minlength=num_classes).float().clamp(min=1)
    return labels.numel() / (num_classes * counts)


def create_dataloaders(data, loader_type: str = "neighbor", batch_size: int = 1024,
                       num_neighbors: int | Sequence[int] = (15, 10),
                       num_layers: int = 2, balanced: bool = False,
                       eval_loaders: bool = False, eval_batch_size: int = 4096,
                       num_workers: int = 0) -> dict[str, DataLoader | NeighborLoader | None]:
    """
    Loaders for a single-graph node-classification dataset, returned as
    ``{"train", "val", "test"}``. "val"/"test" are None unless
    ``eval_loaders=True``; when the graph fits in memory (Cora, ogbn-arxiv)
    use the full-batch ``evaluate`` instead.

    ``"neighbor"``: NeighborLoader over the split's seed nodes (first
    ``batch_size`` rows of each batch) with sampled ``num_neighbors`` per
    hop for training and the full neighbourhood for eval. Keep ``data`` on
    the CPU; batches are moved to the device in the epoch functions.
    ``"full"``: one batch holding the whole graph, so the same loop can run
    full-batch training for a like-for-like comparison.

    :param num_neighbors: Fan-out per layer, e.g. (15, 10); -1 = all. An
        int is expanded to ``num_layers`` entries.
    :param balanced: Sample training seeds with ImbalancedSampler
        (inverse class frequency, with replacement).
    :param num_workers: Keep 0 on macOS; process spawn outweighs sampling.
    """
    if loader_type == "full":
        loader = DataLoader([data], batch_size=1, shuffle=False)
        eval_loader = loader if eval_loaders else None
        return {"train": loader, "val": eval_loader, "test": eval_loader}
    if loader_type != "neighbor":
        raise ValueError(f"loader_type must be 'neighbor' or 'full', got {loader_type!r}")

    if isinstance(num_neighbors, int):
        num_neighbors = [num_neighbors] * num_layers
    num_neighbors = list(num_neighbors)

    sampler = ImbalancedSampler(data, input_nodes=data.train_mask) if balanced else None
    loaders = {
        "train": NeighborLoader(
            data, num_neighbors=num_neighbors, input_nodes=data.train_mask,
            batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
            num_workers=num_workers,
        ),
        "val": None,
        "test": None,
    }
    if eval_loaders:
        # Full neighbourhood reproduces full-batch predictions for SAGE/GAT,
        # not GCN (its symmetric norm also sees the sampled neighbours' degrees).
        for split in ("val", "test"):
            loaders[split] = NeighborLoader(
                data, num_neighbors=[-1] * len(num_neighbors),
                input_nodes=getattr(data, f"{split}_mask"),
                batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
            )
    return loaders


def _target_index(batch, mask_name: str):
    """
    Rows to score: the seed nodes (first ``batch_size`` rows) of a
    NeighborLoader batch, else ``batch[mask_name]`` for a full-graph batch,
    whose ``batch_size`` is the number of graphs.
    """
    if "n_id" in batch:
        return slice(0, batch.batch_size)
    return batch[mask_name]


def train_one_epoch(data, model, criterion, optimizer, mask) -> Metrics:
    """
    Full-batch training step on `mask` (e.g. data.train_mask).
    Caller is responsible for `model.to(device)` / `data.to(device)` once
    before the epoch loop -- doing it every epoch here would be wasted work.
    """
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()

    pred = out[mask].argmax(dim=1)
    return Metrics.compute(data.y[mask], pred, loss.item())


@torch.no_grad()
def evaluate(data, model, criterion, mask, include_auc: bool = False,
             include_cal: bool = False, out: torch.Tensor | None = None) -> Metrics:
    """Evaluates `model` on `mask` (e.g. data.val_mask / data.test_mask).
    Set include_auc=True only for the final best-checkpoint report (see
    Metrics' docstring for why it's off by default). Pass precomputed
    full-graph logits as `out` (e.g. from `inference_layerwise`) to skip
    the forward pass.
    """
    if out is None:
        model.eval()
        out = model(data.x, data.edge_index)
    y = data.y[mask].to(out.device)  # data may be on the CPU when `out` is given
    loss = criterion(out[mask], y)
    probs = torch.softmax(out[mask], dim=1) if include_auc else None
    pred = out[mask].argmax(dim=1)
    return Metrics.compute(y, pred, loss.item(), y_proba=probs,
                           include_auc=include_auc, include_cal=include_cal)


@torch.no_grad()
def inference_layerwise(model, data, device, batch_size: int = 4096) -> torch.Tensor:
    """
    Full-graph logits computed one layer at a time, for graphs whose
    edge-level messages don't fit in memory (e.g. ogbn-products). Each pass
    loads a batch of nodes with its full one-hop neighbourhood, so memory
    scales with batch_size * degree rather than with the edge count.
    Exact for SAGE, SAGEBN, GATv2 and GConv; GCN's symmetric normalisation needs
    source-node degrees the one-hop subgraph doesn't have, so don't use it
    for GCN. `data` stays on the CPU; `model` must already be on `device`.
    """
    model.eval()
    loader = NeighborLoader(data, num_neighbors=[-1], input_nodes=None,
                            batch_size=batch_size, shuffle=False)
    x_all = data.x
    norms = getattr(model, "norms", None)  # GraphSAGEBN: norm between conv and activation
    for i, layer in enumerate(model.layers):
        outs = []
        for batch in loader:
            x = x_all[batch.n_id].to(device)
            x = layer(x, batch.edge_index.to(device))[:batch.batch_size]
            if norms is not None:
                x = norms[i](x)
            outs.append(model.activation(x).cpu())
        x_all = torch.cat(outs)
    return model.lin1(x_all.to(device))


def train_one_epoch_loader(loader, model, criterion, optimizer, device,
                           mask_name: str = "train_mask") -> Metrics:
    """One training epoch over a ``create_dataloaders`` loader. Loss and
    metrics use the target rows only, pooled over the epoch so macro
    averages reflect the epoch's class balance.
    """
    model.train()
    total_loss = 0.0
    total_targets = 0
    all_y, all_pred = [], []

    for batch in loader:
        batch = batch.to(device)
        idx = _target_index(batch, mask_name)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)[idx]
        y = batch.y[idx]

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.numel()
        total_targets += y.numel()
        all_y.append(y.detach())
        all_pred.append(out.argmax(dim=1).detach())

    return Metrics.compute(torch.cat(all_y), torch.cat(all_pred),
                           total_loss / total_targets)


@torch.no_grad()
def evaluate_loader(loader, model, criterion, device, mask_name: str,
                    include_auc: bool = False, include_cal: bool = False) -> Metrics:
    """``evaluate`` over a loader; ``mask_name`` (e.g. "val_mask") only
    matters for full-graph batches.
    """
    model.eval()
    total_loss = 0.0
    total_targets = 0
    all_y, all_pred, all_probs = [], [], []

    for batch in loader:
        batch = batch.to(device)
        idx = _target_index(batch, mask_name)
        out = model(batch.x, batch.edge_index)[idx]
        y = batch.y[idx]

        total_loss += criterion(out, y).item() * y.numel()
        total_targets += y.numel()
        all_y.append(y)
        all_pred.append(out.argmax(dim=1))
        if include_auc:
            all_probs.append(torch.softmax(out, dim=1))

    probs = torch.cat(all_probs) if include_auc else None
    return Metrics.compute(torch.cat(all_y), torch.cat(all_pred),
                           total_loss / total_targets, y_proba=probs,
                           include_auc=include_auc, include_cal=include_cal)


def load_checkpoint(model, checkpoint_path, device, state_key="best_model"):
    """
    Loads a saved state dict into `model` (already instantiated with the
    matching architecture args) and moves it to `device` in eval mode.
    :param model: An instantiated (untrained or arbitrary-weight) model of
        the same architecture the checkpoint was saved from.
    :param checkpoint_path: Path to the saved checkpoint file.
    :param device: Device to move the model to after loading.
    :param state_key: Key of the checkpoint dict holding the state dict.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint[state_key])
    model.to(device)
    model.eval()
    return model