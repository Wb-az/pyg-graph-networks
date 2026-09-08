"""
Unified loading for node-classification datasets.

- Planetoid datasets such as Cora provide boolean
  train_mask/val_mask/test_mask directly.
- OGB datasets provide index-based train/valid/test splits and labels
  shaped [N, 1].

This module normalizes OGB labels and optionally creates boolean masks
for a consistent node-classification interface.
"""

from __future__ import annotations

import pathlib
import warnings

import torch
from torch_geometric.datasets import Planetoid


__all__ = ["load_planetoid", "load_ogb_node"]


def load_planetoid(path: str | pathlib, dataset_name: str = "Cora") -> tuple[Planetoid, InMemoryDataset]:
    """
    :param path: Path to the directory where the dataset is stored or will be stored.
    :param dataset_name: Name of the dataset to load (default is "Cora").
    :return: A tuple containing the dataset object and the first data item in the dataset.
    """

    dataset = Planetoid(root=path, name=dataset_name)

    return dataset, dataset[0]


def _masks_from_index_split(num_nodes:int, split_idx: dict) -> dict[str, torch.Tensor]:
    """
    Convert index-based splits into boolean masks.
    :param num_nodes: The number of nodes in the graph
    :param split_idx: A dictionary containing the indices of the train, validation, and test sets
    :return: A dictionary of boolean masks
    """
    split_mapping = {
        "train_mask": "train",
        "val_mask": "valid",
        "test_mask": "test"}

    masks = {}
    for mask_name, split_name in split_mapping.items():
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[split_idx[split_name]] = True
        masks[mask_name] = mask

    return masks


def _configure_ogb_compatibility() -> None:
    """Configure compatibility for cached OGB/PyG datasets."""

    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage

    torch.serialization.add_safe_globals(
        [DataEdgeAttr, DataTensorAttr, GlobalStorage]
    )

    warnings.filterwarnings(
        "ignore",
        message="The given NumPy array is not writable.*",
        category=UserWarning
    )


def load_ogb_node(name: str, root: str | pathlib):
    """
    Load and normalize the OGB Node datasets

    OGB provides:
    - Labels shaped [N, 1]
    - Index-based train/valid/test splits
    - Mappings from labels to node indices

    The labels are converted to 1-D labels and the indexes to boolean masks, so the dataset
    follows the same interface used for PyGeometric's datasets.

    :param name: A string with the dataset name from the OGB nodes datasets
    :param root: A string or path to the dataset
    :return: a tuple of (data, dataset)
    """
    _configure_ogb_compatibility()

    from ogb.nodeproppred import PygNodePropPredDataset

    dataset = PygNodePropPredDataset(name=name, root=root)
    data = dataset[0]
    data.y = data.y.squeeze(-1)
    split_idx = dataset.get_idx_split()
    for split, mask in _masks_from_index_split(data.num_nodes, split_idx).items():
        setattr(data, split, mask)

    return dataset, data, split_idx





