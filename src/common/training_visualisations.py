# Quick-look embedding visualization for node classification.
# Complementary to altair_plots.py (interactive graph-structure charts) and
# plotly_plots.py (interactive, shareable projections)
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
import numpy as np
import pandas as pd

import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


@torch.no_grad()
def get_embeddings(model: torch.nn.Module, x: torch.Tensor,
                   edge_index: torch.Tensor) -> torch.Tensor:
    """
    Node embeddings for UMAP/TSNE/silhouette use.
    """
    model.eval()
    return model.encode(x, edge_index)


@torch.no_grad()
def compute_embedding(model, x, edge_index, method="tsne",
                      n_components=2, random_state=0):
    """
    Low-dimensional projection of node embeddings.
    """
    features = get_embeddings(model, x, edge_index).detach().cpu().numpy()

    if method == "tsne":
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            init="pca",
        )
    elif method == "pca":
        reducer = PCA(
            n_components=n_components,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'tsne' or 'pca'.")

    return reducer.fit_transform(features)


def plot_embeddings(embeddings, labels, label_map,
                    figures_dir, titles=None, point_size=40,
                    alpha=0.8, panel_size=(400, 400),
                    name="embeddings",):
    """Plots one or more 2D node embeddings using Altair."""

    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    else:
        labels = np.asarray(labels)

    categorical_labels = [
        label_map[int(label)]
        for label in labels]

    titles = titles or [None] * len(embeddings)

    charts = []

    for embedding, title in zip(embeddings, titles):
        plot_df = pd.DataFrame({
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "class": categorical_labels})

        chart = (
            alt.Chart(plot_df)
            .mark_circle(
                size=point_size,
                opacity=alpha)
            .encode(
                x=alt.X("x:Q", axis=None),
                y=alt.Y("y:Q", axis=None),
                color=alt.Color(
                    "class:N",
                    title="Class",
                    scale=alt.Scale(scheme="tableau10")),
                tooltip=[
                    alt.Tooltip("class:N", title="Class")])
            .properties(
                width=panel_size[0],
                height=panel_size[1],
                title=title))
        charts.append(chart)

    combined_chart = alt.hconcat(*charts).resolve_scale(color="shared")

    combined_chart.save(figures_dir / f"{name}_embeddings.html")
    combined_chart.save(figures_dir / f"{name}_embeddings.svg")

    return combined_chart


def plot_embeddings_3d(embedding, labels, label_map,
                       fig_dir, name="embeddings",
                       title="3D embedding"):
    """
    Interactive 3D embedding plot with Plotly.
    """
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    else:
        labels = np.asarray(labels)

    categorical_labels = [
        label_map[int(label)]
        for label in labels
    ]

    plot_df = pd.DataFrame({
        "x": embedding[:, 0],
        "y": embedding[:, 1],
        "z": embedding[:, 2],
        "class": categorical_labels})

    fig = px.scatter_3d(
        plot_df,
        x="x",
        y="y",
        z="z",
        color="class",
        color_discrete_sequence=px.colors.qualitative.T10,
        title=title,
        opacity=0.8)

    fig.update_traces(marker=dict(size=4))

    fig.update_layout(
        width=900,
        height=750,
        scene=dict(
            aspectmode="cube"),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=50))

    fig.write_html(fig_dir / f"{name}_embedding.html")
    fig.write_image(fig_dir / f"{name}_embedding.svg")

    return fig


def plot_metric_curves(history_df, fig_dir, name,
                       metrics=("loss", "acc", "f1"),
                       splits=("train", "val")):
    """
    Plots training curves for each seed by metric and split.
    """

    plot_df = history_df[
        history_df["split"].isin(splits)].copy()

    plot_df = plot_df.melt(
        id_vars=["epoch", "split", "seed"],
        value_vars=list(metrics),
        var_name="metric",
        value_name="value")

    chart = (
        alt.Chart(plot_df)
        .mark_line()
        .encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y("value:Q", title=None),
            color=alt.Color("seed:N", title="Seed",
                            scale=alt.Scale(scheme="dark2")),
            detail="seed:N",
            column=alt.Column(
                "metric:N",
                title=None,
                sort=list(metrics),
            ),
            row=alt.Row(
                "split:N",
                title=None,
                sort=list(splits),
            ),
            tooltip=[
                "seed:N",
                "split:N",
                "metric:N",
                "epoch:Q",
                alt.Tooltip("value:Q", format=".4f"),])
        .properties(
            width=280,
            height=220).resolve_scale(
            y="independent"))

    chart.save(fig_dir / f"{name}_training_curve.html")
    chart.save(fig_dir / f"{name}_training_curve.svg")

    return chart


def get_confusion_matrix(y_true, y_pred, labels, title):
    cm =confusion_matrix(y_true, y_pred, labels, normalize=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.ax_.set_title(title)
    disp.plot(cmap='Blues')
    plt.show()
    plt.show()