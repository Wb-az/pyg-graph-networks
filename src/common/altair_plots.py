import os
import pathlib
import random
from collections import Counter
import pandas as pd

import networkx as nx
from torch_geometric.utils import to_networkx

import altair as alt
import altair_nx as anx


def plot_label_distribution(
    labels, path_to_graph: pathlib, graph_name : str | None = None,  label_dict=None, title= "Type of papers distribution",
        subtitle: str | bool = '', split: str | bool = False, display= True):
    """
    Counts graph labels and saves an interactive, portfolio-ready Altair chart to HTML.

    Args:
        labels: A PyTorch tensor, numpy array, or list of node labels (e.g.
          graph.y).
        :param labels: A PyTorch tensor, numpy array, or list of node labels
        :param path_to_graph: A directory path where the graph will be saved.
        :param graph_name: A string representing the name of the graph.
        :param title: A string with the graph title.
        :param subtitle: The subtitle for the chart.
        :param split: The type of split (e.g. 'Train', 'Test') or a boolean flag.
        :param labels: A PyTorch tensor, numpy array, or list of node labels
        :param display: A boolean flag to determine whether to display the chart.
        :param label_dict: A dictionary mapping labels to their corresponding names.
    """
    # Convert tensor/array to a Python list
    if label_dict is None:
        label_dict = {}
    if hasattr(labels, "tolist"):
        labels_list = labels.tolist()

    else:
        labels_list = list(labels)

    # Process data into a clean Pandas DataFrame
    counts_dict = Counter(labels_list)

    df = pd.DataFrame([
            {"Category": label_dict.get(cls, f"Class {cls}"),
                "Count": count}
            for cls, count in counts_dict.items()]
            ).sort_values("Count", ascending=False)

    # Create the interactive Altair Chart

    parts = [str(p) for p in (subtitle, f'- {split}' if split else None) if p]
    subtitle = '\n'.join(parts)

    chart = (
        alt.Chart(df)
        .mark_bar(
            cornerRadiusTopLeft=4,
            cornerRadiusTopRight=4,
            color="#5778A4",  # Clean, professional portfolio blue
        )
        .encode(
            x=alt.X(
                "Category:N",
                sort="-y",
                axis=alt.Axis(labelAngle=-30, labelPadding=10),
            ),
            y=alt.Y("Count:Q", axis=alt.Axis(gridColor="#eaeaea")),
            text=alt.Text("Count:Q", format='.0f'),
            # Add dynamic hover effects
            tooltip=["Category", "Count"],
            opacity=alt.condition(
                alt.datum.active, alt.value(1.0), alt.value(0.85)
            ),
        )
        .properties(
            title=alt.TitleParams(
                text=title,
                subtitle=subtitle,
                anchor="start",  # Clean, left-aligned title architecture
                fontSize=16,
                subtitleFontSize=13,
                subtitleColor="#666666",
            ),
            width=300,
            height=250,
        )
        .configure_view(strokeWidth=0)  # Removes ugly outer box borders
        .configure_axis(
            labelFont="Inter, system-ui, sans-serif",
            titleFont="Inter, system-ui, sans-serif",
            titleFontSize=12,
            titlePadding=15,
        )
    )

    # Save to standalone HTML
    if not path_to_graph.exists():
        os.makedirs(path_to_graph)
    chart.save(f'{path_to_graph}/{graph_name}.html')
    chart.save(f'{path_to_graph}/{graph_name}.svg')
    if display:
        chart.display()


def convert_to_networkx(graph, n_sample=None, label_dict=None, named_node='paper_topic'):

    g = to_networkx(graph, node_attrs=["x"])
    nx.set_node_attributes(g, {n: g.nodes[n].pop("x") for n in g.nodes}, name="features")
    y = graph.y.numpy()

    if n_sample is not None:
        sampled_nodes = random.sample(list(g.nodes), n_sample)
        sampled_nodes.sort()
        g = g.subgraph(sampled_nodes).copy()
        y = y[sampled_nodes]

    for i, node in enumerate(g.nodes):
        g.nodes[node][named_node] = label_dict[y[i]]

    return g


def compute_layout(g, graph_type: str | None=None, seed=142):

    if graph_type == 'spring' or graph_type is None:
        pos = nx.spring_layout(g, seed=seed)
    elif graph_type == 'forceatlas2':
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pos = nx.forceatlas2_layout(g, linlog=True, seed=seed)
    else:
        raise ValueError(
            f"Unknown layout type {graph_type!r}. Available types: 'spring' and 'forceatlas2'"
        )

    return pos


def plot_graph(g, pos, node_tooltip: str | list[str] | None, node_colour="paper_topic", nodes=120,
               sampling_type="spring", dataset_name="Cora", node_size=120,
               node_label:str|None=None, edge_colour="#9C9C9C", edge_width=0.8,
               edge_legend=None, alpha=0.7, curved_edges=True, path_to_plots='plots',
               plot_name='cora_graph', display=True, cmap="set2"):

    chart = anx.draw_networkx(g,
    pos=pos,
    node_label=node_label,
    node_colour=node_colour,
    node_cmap=cmap,
    node_alpha=alpha,
    node_size=node_size,
    edge_colour=edge_colour,
    edge_width=edge_width,
    edge_legend=edge_legend,
    node_tooltip=node_tooltip,
    curved_edges=curved_edges).properties(width=800,
    height=500,
    title={"text":f"Graph Visualization - Dataset: {dataset_name}",
           "subtitle": f"Nodes samples: {nodes} {sampling_type} algorithm",
            "subtitleColor": "#9C9C9C"}).configure_legend(
    titleFontSize=16,
    labelFontSize=12)

    # Make the interactive layout zoomable and pannable
    chart = chart.interactive()

    # Save chart
    os.makedirs(path_to_plots, exist_ok=True)
    chart.save(f'{path_to_plots}/{plot_name}.html')
    chart.save(f'{path_to_plots}/{plot_name}.svg')
    if display:
        chart.display()