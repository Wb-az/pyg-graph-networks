"""Node-classification models sharing one layer recipe.

Design decision (September 2026): every model applies, per message-passing layer,
``dropout -> conv -> activation`` inside ``encode``, and ``forward`` is
``encode -> dropout -> lin1``. This is the PyG example-script recipe
(examples/gcn.py, examples/gat.py) and the original Kipf & Welling GCN setup,
where dropout also masks the sparse input features. The alternative PyG
recipe (Colab tutorial, ``torch_geometric.nn.models.BasicGNN``) is
``conv -> activation -> dropout`` with no input dropout; the two differ only in
that input dropout, and are identical in eval mode.

Invariants relied on elsewhere (see test/node_models_test.py):
- all four models expose ``layers`` (ModuleList), ``lin1``, ``encode`` and ``forward``;
- ``encode`` never ends with dropout, so it returns clean embeddings in any mode;
- ``forward(x, ei) == lin1(encode(x, ei))`` in eval mode.
Do not change the order in one model without changing all four, otherwise the
model comparison mixes regularisation schemes.
"""
import torch.nn as nn
from torch_geometric.nn import GCNConv,GATv2Conv, SAGEConv, GraphConv, Linear


__all__ = ['GCN', 'GConv', 'GATV2', 'GraphSAGE']


class GCN(nn.Module):
    def __init__(self, num_layers : int, in_feat: int, hid_feat: int,
                 num_classes:int, dropout:float=0.3):
        """
        Initializes the model with the specified number of layers, input and output
        channel dimensions, dataset schema, class categories, and dropout rate.
        The model is designed to use a multi-layer GCN architecture where each
        layer is built using `GCNConv`, and an activation function (`ReLU`) is
        applied after each layer. Dropout regularization can also be applied
        through the specified dropout rate. A final `Linear` head maps the last
        hidden representation to class logits.

        :param num_layers: Number of layers in the GCN model.
        :type num_layers: An integer with the number of layers.
        :param in_feat: Dimension of the input channels.
        :type in_feat: An integer with the input channels.
        :param hid_feat: Dimension of the output channels of each layer.
        :type hid_feat: An integer with the output channels of each layer.
        :param num_classes: Number of output classes for the classification task.
        :type num_classes: An integer with the number of output classes.
        :param dropout: Dropout rate for regularization during training.
            Default value: 0.3.
        :type dropout: float
        """

        super().__init__()
        self.num_layers = num_layers
        self.in_channels = in_feat
        self.out_channels = hid_feat
        self.num_classes = num_classes

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            hid_in = self.out_channels if i > 0 else self.in_channels
            self.layers.append(GCNConv(in_channels=hid_in, out_channels=self.out_channels))

        self.lin1 = Linear(in_channels=self.out_channels, out_channels=self.num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.lin1.reset_parameters()

    def encode(self, x, edge_index):
        """Returns learned node embeddings before the classification head."""
        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x, edge_index)
            x = self.activation(x)
        return x

    def forward(self, x, edge_index):
        x = self.encode(x, edge_index)
        x = self.dropout(x)
        x = self.lin1(x)
        return x


class GConv(nn.Module):
    def __init__(self, num_layers: int, in_feat: int, hid_feat: int,
                 num_classes: int, dropout: float = 0.3
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.in_channels = in_feat
        self.out_channels = hid_feat
        self.num_classes = num_classes

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            hid_in = self.out_channels if i > 0 else self.in_channels
            # Sum aggregation is the Morris et al. definition of GraphConv and is
            # what distinguishes it from SAGEConv; with aggr='mean' the two
            # layers compute the same function and train identically.
            self.layers.append(GraphConv(in_channels=hid_in, out_channels=self.out_channels,
                                         aggr='add'))

        self.lin1 = Linear(in_channels=self.out_channels, out_channels=self.num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.lin1.reset_parameters()

    def encode(self, x, edge_index):
        """Returns learned node embeddings before the classification head."""
        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x, edge_index)
            x = self.activation(x)
        return x

    def forward(self, x, edge_index):
        x = self.encode(x, edge_index)
        x = self.dropout(x)
        x = self.lin1(x)
        return x


class GATV2(nn.Module):
    def __init__(self, num_layers : int, in_feat: int, hid_feat: int,
                 num_classes:int, heads:int, dropout:float=0.3):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.num_layers = num_layers
        self.in_channels = in_feat
        self.out_channels = hid_feat
        self.num_classes = num_classes
        self.heads = heads

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            is_last = i == self.num_layers - 1
            num_heads = 1 if is_last else self.heads
            hid_in = self.in_channels if i == 0 else self.out_channels * self.heads
            self.layers.append(GATv2Conv(in_channels=hid_in, out_channels=self.out_channels,
                                         heads=num_heads, concat=True,
                                         dropout=dropout))

        self.lin1 = Linear(in_channels=self.out_channels, out_channels=self.num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ELU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.lin1.reset_parameters()

    def encode(self, x, edge_index):
        """Returns learned node embeddings before the classification head."""
        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x, edge_index)
            x = self.activation(x)
        return x

    def forward(self, x, edge_index):
        x = self.encode(x, edge_index)
        x = self.dropout(x)
        x = self.lin1(x)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, num_layers: int, in_feat: int, hid_feat: int,
                 num_classes: int, dropout: float = 0.3
                 ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.num_layers = num_layers
        self.in_channels = in_feat
        self.out_channels = hid_feat
        self.num_classes = num_classes

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            hid_in = self.out_channels if i > 0 else self.in_channels
            self.layers.append(SAGEConv(in_channels=hid_in, out_channels=self.out_channels))

        self.lin1 = Linear(in_channels=self.out_channels, out_channels=self.num_classes)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.lin1.reset_parameters()

    def encode(self, x, edge_index):
        """Returns learned node embeddings before the classification head."""
        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x, edge_index)
            x = self.activation(x)
        return x

    def forward(self, x, edge_index):
        x = self.encode(x, edge_index)
        x = self.dropout(x)
        x = self.lin1(x)
        return x
