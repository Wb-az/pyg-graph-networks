import random
import numpy as np

import torch
from torch_geometric.loader import NeighborLoader

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


def make_neighbor_loader(data, input_nodes, num_neighbors=(15, 10),
                         batch_size=1024, shuffle=True) -> NeighborLoader:
    """
    Mini-batch NeighborLoader for a single large graph -- not needed for
    Cora or ogbn-arxiv (both fit full-batch comfortably), kept ready for a
    future larger dataset (e.g. ogbn-products).

    :param data: A single-graph Data object.
    :param input_nodes: Seed/target nodes to sample around, e.g. data.train_mask.
    :param num_neighbors: Fanout per hop, one entry per GNN layer
        (e.g. (15, 10) for a 2-layer model).
    :param batch_size: Number of seed nodes per mini-batch.
    :param shuffle: A boolean flag to shuffle the data before each epoch.
    """
    return NeighborLoader(
        data, num_neighbors=list(num_neighbors), input_nodes=input_nodes,
        batch_size=batch_size, shuffle=shuffle,
    )


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
def evaluate(data, model, criterion, mask,
                include_auc: bool = False, include_cal=False) -> Metrics:
    """Evaluates `model` on `mask` (e.g. data.val_mask / data.test_mask).
    Set include_auc=True only for the final best-checkpoint report (see
    Metrics' docstring for why it's off by default).
    """
    model.eval()
    out = model(data.x, data.edge_index)
    loss = criterion(out[mask], data.y[mask])
    probs = torch.softmax(out[mask], dim=1) if include_auc else None
    pred = out[mask].argmax(dim=1)
    return Metrics.compute(data.y[mask], pred, loss.item(), y_proba=probs,
                           include_auc=include_auc, include_cal=include_cal)


def train_one_epoch_loader(loader: NeighborLoader, model, criterion,
                           optimizer, device) -> Metrics:
    """Mini-batch training epoch over a NeighborLoader (or any NodeLoader).

    Each mini-batch's first `batch.batch_size` nodes are the seed/target
    nodes being trained on; the rest are sampled k-hop neighbours included
    only for message passing -- per NeighborLoader's contract, loss and
    metrics must be computed on that seed-node slice only, not the full
    mini-batch.

    Predictions/labels are pooled across all mini-batches, and Metrics are
    computed once at the end (not averaged per-batch), so macro-averaged
    metrics reflect the whole epoch's class balance correctly.
    """
    model.train()
    total_loss = 0.0
    total_seeds = 0
    all_y, all_pred = [], []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        seed_out = out[:batch.batch_size]
        seed_y = batch.y[:batch.batch_size]

        loss = criterion(seed_out, seed_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.batch_size
        total_seeds += batch.batch_size
        all_y.append(seed_y.detach())
        all_pred.append(seed_out.argmax(dim=1).detach())

    y_true = torch.cat(all_y)
    y_pred = torch.cat(all_pred)
    return Metrics.compute(y_true, y_pred, total_loss / total_seeds)


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



