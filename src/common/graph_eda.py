from collections import Counter

import numpy as np
from torch_geometric.utils import (assortativity,
    degree,
    homophily,
    to_networkx,
    to_scipy_sparse_matrix,
)

from scipy.sparse.csgraph import connected_components
import networkx as nx


class GraphEDA:
    """Exploratory metrics for a single PyG graph (node classification).

    Usage:
        eda = GraphEDA(graph, label_dict=label_dict)
        eda.summary()
        metrics = eda.as_dict() # if you want the raw numbers
    """

    def __init__(self, graph, label_dict=None):
        self.g = graph
        self.label_dict = label_dict
        self._compute()

    def _compute(self):
        g = self.g
        n = g.num_nodes
        e = g.num_edges  # PyG counts each undirected edge twice

        self.num_nodes = n
        self.num_edges = e
        self.num_node_features = g.num_node_features
        self.num_classes = int(np.unique(g.y.numpy()).size)

        # Class balance
        self.class_counts = Counter(g.y.tolist())

        # Degree stats (out-degree; equals total degree for undirected PyG graphs)
        deg = degree(g.edge_index[0], num_nodes=n)
        self.avg_degree = e / n
        self.deg_min = int(deg.min())
        self.deg_max = int(deg.max())
        self.deg_std = float(deg.std())

        # Density: 2E_undirected / (n(n-1)); PyG num_edges already = 2E_undirected
        self.density = e / (n * (n - 1)) if n > 1 else 0.0

        # Homophily
        self.edge_homophily = homophily(g.edge_index, g.y, method='edge')
        self.node_homophily = homophily(g.edge_index, g.y, method='node')
        self.edge_insensitive_homophily = homophily(
            g.edge_index, g.y, method='edge_insensitive'
        )

        # Assortativity (degree)
        self.assortativity = (
            float(assortativity(g.edge_index)) if assortativity is not None else None
        )

        # Connectivity
        adj = to_scipy_sparse_matrix(g.edge_index, num_nodes=n)
        n_comp, comp_labels = connected_components(adj, directed=False)
        self.num_components = int(n_comp)
        self.largest_component = int(np.bincount(comp_labels).max())

        # Structural flags
        self.has_isolated_nodes = bool(g.has_isolated_nodes())
        self.has_self_loops = bool(g.has_self_loops())
        self.is_undirected = bool(g.is_undirected())

        # Splits (guarded, not every dataset has all masks)
        self.train_nodes = self._mask_sum('train_mask')
        self.val_nodes = self._mask_sum('val_mask')
        self.test_nodes = self._mask_sum('test_mask')

    def _mask_sum(self, name):
        mask = getattr(self.g, name, None)
        return int(mask.sum()) if mask is not None else None

    def as_dict(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in ('g', 'label_dict')
        }

    def summary(self):
        nodes = int(self.num_nodes)
        print('=== Graph structure ===')
        print(f'Number of nodes: {nodes}')
        print(f'Number of edges: {self.num_edges}')
        print(f'Number of node features: {self.num_node_features}')
        print(f'Number of classes: {self.num_classes}')
        print(f'Graph density: {float(self.density):.6f}')
        print()

        print('=== Degree ===')
        print(f'Average node degree: {self.avg_degree:.2f}')
        print(f'Degree min / max / std: '
              f'{self.deg_min} / {self.deg_max} / {self.deg_std:.2f}')
        print()

        print('=== Class balance ===')
        for cls, count in sorted(self.class_counts.items()):
            name = self.label_dict[cls] if self.label_dict else cls
            print(f'{name}: {count} ({count / int(nodes):.2%})')
        print()

        print('=== Homophily / mixing ===')
        print(f'Edge homophily: {self.edge_homophily:.2f}')
        print(f'Node homophily: {self.node_homophily:.2f}')
        print(f'Edge-insensitive homophily: {self.edge_insensitive_homophily:.2f}')
        if self.assortativity is not None:
            print(f'Degree assortativity: {self.assortativity:.2f}')
        print()

        print('=== Connectivity ===')
        print(f'Connected components: {self.num_components}')
        print(f'Largest component: {self.largest_component} '
              f'({self.largest_component / nodes:.2%} of nodes)')
        print(f'Has isolated nodes: {self.has_isolated_nodes}')
        print(f'Has self-loops: {self.has_self_loops}')
        print(f'Is undirected: {self.is_undirected}')
        print()

        print('=== Splits ===')
        if self.train_nodes is not None:
            print(f'Training nodes: {self.train_nodes} '
                  f'(label rate {self.train_nodes / nodes:.2f})')
        if self.val_nodes is not None:
            print(f'Validation nodes: {self.val_nodes}')
        if self.test_nodes is not None:
            print(f'Test nodes: {self.test_nodes}')


def _stats(values):
    """min / max / mean / std helper."""
    a = np.asarray(values, dtype=float)
    return {
        'min': float(a.min()),
        'max': float(a.max()),
        'mean': float(a.mean()),
        'std': float(a.std()),
    }


class DatasetEDA:
    """Exploratory metrics for a DATASET of graphs (graph classification).

    Works on any iterable of PyG `Data` objects (e.g. a TUDataset).

    Usage:
        eda = DatasetEDA(dataset)
        eda.summary()
    """

    def __init__(self, dataset, label_dict=None):
        self.dataset = dataset
        self.label_dict = label_dict
        self._compute()

    def _compute(self):
        ds = self.dataset
        self.num_graphs = len(ds)

        nodes, edges, densities, labels = [], [], [], []
        for data in ds:
            n = data.num_nodes
            e = data.num_edges  # PyG double-counts undirected edges
            nodes.append(n)
            edges.append(e)
            densities.append(e / (n * (n - 1)) if n > 1 else 0.0)
            if getattr(data, 'y', None) is not None:
                labels.append(int(data.y.view(-1)[0]))

        self.nodes_per_graph = _stats(nodes)
        self.edges_per_graph = _stats(edges)
        self.density_per_graph = _stats(densities)

        # Feature / class metadata (prefer dataset attributes when present)
        self.num_node_features = getattr(ds, 'num_node_features', None)
        if self.num_node_features is None:
            self.num_node_features = ds[0].num_node_features

        self.graph_label_counts = Counter(labels) if labels else Counter()
        self.num_classes = (
            getattr(ds, 'num_classes', None) or len(self.graph_label_counts)
        )

    def summary(self):
        print('=== Dataset (graph classification) ===')
        print(f'Number of graphs: {self.num_graphs}')
        print(f'Number of node features: {self.num_node_features}')
        print(f'Number of classes: {self.num_classes}')
        print()

        for name, s, dec in (
            ('Nodes per graph', self.nodes_per_graph),
            ('Edges per graph', self.edges_per_graph),
            ('Density per graph', self.density_per_graph),
        ):
            vals = ' / '.join(
                format(s[k], f'.{dec}f') for k in ('min', 'max', 'mean', 'std'))
            print(f'{name} (min/max/mean/std): {vals}')
        print()

        print('=== Graph-label balance ===')
        total = sum(self.graph_label_counts.values()) or 1
        for cls, count in sorted(self.graph_label_counts.items()):
            name = self.label_dict[cls] if self.label_dict else cls
            print(f'  {name}: {count} ({count / total:.2%})')


class LinkEDA:
    """Exploratory metrics for a single graph, framed for LINK prediction.

    Focus: degree, clustering (feeds common-neighbour baselines), and the
    positive- / negative-edge balance.

    Usage:
        eda = LinkEDA(graph)
        eda.summary()
    """

    def __init__(self, graph, num_pos_edges=None):
        # num_pos_edges: override if you pass a split (e.g. only train edges).
        self.g = graph
        self._num_pos_override = num_pos_edges
        self._compute()

    def _compute(self):
        g = self.g
        n = g.num_nodes
        e_dir = g.num_edges                      # directed count in PyG
        e_undir = e_dir // 2 if bool(g.is_undirected()) else e_dir

        self.num_nodes = n
        self.num_undirected_edges = e_undir
        self.density = e_dir / (n * (n - 1)) if n > 1 else 0.0

        # Degree
        deg = degree(g.edge_index[0], num_nodes=n)
        self.deg_min = int(deg.min())
        self.deg_max = int(deg.max())
        self.deg_mean = float(deg.mean())
        self.deg_std = float(deg.std())

        # Clustering (needs networkx); this drives neighbour-based link baselines
        self.avg_clustering = None
        self.transitivity = None
        if HAS_NX:
            gx = to_networkx(g, to_undirected=True)
            self.avg_clustering = float(nx.average_clustering(gx))
            self.transitivity = float(nx.transitivity(gx))

        # Positive- / negative-edge balance
        pos = self._num_pos_override if self._num_pos_override is not None else e_undir
        total_possible = n * (n - 1) // 2
        neg = total_possible - pos
        self.num_pos = int(pos)
        self.num_neg = int(neg)
        self.pos_neg_ratio = pos / neg if neg else float('inf')

    def summary(self):
        print('=== Link prediction EDA ===')
        print(f'Number of nodes: {self.num_nodes}')
        print(f'Undirected edges (positives): {self.num_undirected_edges}')
        print(f'Graph density: {float(self.density):.6f}')
        print()

        print('=== Degree ===')
        print(f'Degree min / max / mean / std: '
              f'{self.deg_min} / {self.deg_max} / '
              f'{self.deg_mean:.2f} / {self.deg_std:.2f}')
        print()

        print('=== Clustering (neighbour-based baselines) ===')
        if self.avg_clustering is not None:
            print(f'Average clustering coefficient: {float(self.avg_clustering):.4f}')
            print(f'Transitivity: {float(self.transitivity):.4f}')
        else:
            print('networkx not installed; skipped.')
        print()

        print('=== Edge balance ===')
        print(f'Positive edges: {self.num_pos}')
        print(f'Possible negatives: {self.num_neg}')
        print(f'Pos/neg ratio: {float(self.pos_neg_ratio):.2e}')
        print('(Sample negatives ~1:1 with positives for training/eval.)')