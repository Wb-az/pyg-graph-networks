from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import altair as alt

import torch
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import (GNNExplainer,
                                               CaptumExplainer)
from torch_geometric.explain.metric import (
    fidelity,
    unfaithfulness,
    characterization_score)

__all__ = ["create_explainer", "eval_explanations", "get_local_feature_importance",
           "get_top_k_features", "aggregate_feature_importance", "plot_feature_importance",
           "plot_aggregate_feature_importance", "select_xai_examples"]


def create_explainer(
    model:torch.nn.Module,
    num_features: int,
    method: Literal[
        "auto",
        "gnn_explainer",
        "captum",
    ] = "auto",
    attribution_method: str | None = None,
    shapley_threshold: int = 50,
    **algorithm_kwargs,
) -> Explainer:
    """
    Create a PyG node-classification explainer.
    Auto behaviour: num_features <= shapley_threshold: then use captum when auto is selected with
        Captum ShapleyValueSampling.
    :param model: A trained torch model.
    :param num_features: Number of features in the dataset.
    :param method: Method to create the explainer.
    :param attribution_method: Attribution method for captum explainer.
    :param shapley_threshold: The threshold for the shapley explainer to
    be used (max num of features to prevent long computation time).
    :param algorithm_kwargs: Arguments to pass to the algorithm.
    :return: An explainer.
    """

    if method == "auto":
        if num_features <= shapley_threshold:
            method = "captum"
            attribution_method = "ShapleyValueSampling"
        else:
            method = "gnn_explainer"

    if method == "gnn_explainer":
        algorithm_kwargs.setdefault("epochs", 200)

        algorithm = GNNExplainer(
            **algorithm_kwargs
        )
        edge_mask_type = "object"

    elif method == "captum":

        if attribution_method is None:
            attribution_method = (
                "ShapleyValueSampling"
                if num_features <= shapley_threshold
                else "InputXGradient"
            )

        algorithm = CaptumExplainer(
            attribution_method=attribution_method,
            **algorithm_kwargs,
        )
        # Captum used primarily for feature attribution.
        edge_mask_type = None

    else:
        raise ValueError(
            f"Unknown method: {method!r}"
        )

    return Explainer(
        model=model,
        algorithm=algorithm,
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type=edge_mask_type,
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )


def eval_explanations(
    explainer,
    data,
    sample_indices=None,
    max_samples=100,
    seed=123,
):
    """
    Evaluate explanations across a sample of test nodes.

    Returns
    -------
    results_df:
        Per-node explanation metrics.

    summary_df:
        Mean/std and valid/invalid counts for each metric.

    feature_importance_df:
        Mean normalized feature importance across sampled nodes.

    sample_indices:
        Indices used in the evaluation.

    explanations:
        Stored explanation objects for the sampled nodes.
    """

    # Default to test nodes.
    if sample_indices is None:
        sample_indices = (
            data.test_mask
            .nonzero(as_tuple=False)
            .view(-1)
        )

    sample_indices = sample_indices.detach().cpu()

    # Reproducible sampling.
    if max_samples is not None and len(sample_indices) > max_samples:
        generator = torch.Generator().manual_seed(seed)

        selected = torch.randperm(
            len(sample_indices),
            generator=generator,
        )[:max_samples]

        sample_indices = sample_indices[selected]

    explainer.model.eval()

    results = []

    feature_sum = torch.zeros(
        data.num_features,
        dtype=torch.float32,
        device=data.x.device,
    )

    with torch.no_grad():
        logits = explainer.model(data.x, data.edge_index)
        probs = torch.softmax(logits, dim=1)

    valid_feature_explanations = 0
    explanations = {}

    # Reproducible GNNExplainer optimization.
    torch.manual_seed(seed)

    for sample_idx in sample_indices:
        sample_idx = int(sample_idx)

        explanation = explainer(
            data.x,
            data.edge_index,
            index=sample_idx,
        )

        explanations[sample_idx] = explanation

        # Explanation metrics.
        fidelity_plus, fidelity_minus = fidelity(
            explainer,
            explanation,
        )

        unf = unfaithfulness(
            explainer,
            explanation,
        )

        char_score = characterization_score(
            pos_fidelity=torch.tensor(fidelity_plus),
            neg_fidelity=torch.tensor(fidelity_minus),
        ).item()

        confidence, predicted_class = probs[sample_idx].max(dim=0)
        true_class = data.y[sample_idx]

        correct = (predicted_class == true_class).item()

        results.append({
            "node": sample_idx,
            "true_class": int(true_class),
            "predicted_class": int(predicted_class),
            "confidence": float(confidence),
            "correct": bool(correct),
            "fidelity_plus": float(fidelity_plus),
            "fidelity_minus": float(fidelity_minus),
            "unfaithfulness": float(unf),
            "characterization": char_score,
        })

        # Feature importance.
        if explanation.node_mask is not None:
            feature_scores = (
                explanation.node_mask
                .detach()
                .sum(dim=0)
            )

            feature_scores = torch.nan_to_num(
                feature_scores,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            total = feature_scores.sum()

            if total > 0:
                feature_scores = feature_scores / total
                feature_sum += feature_scores
                valid_feature_explanations += 1

    # Metric summary.
    results_df = pd.DataFrame(results)

    metric_cols = [
        "fidelity_plus",
        "fidelity_minus",
        "unfaithfulness",
        "characterization"]

    clean_metrics = (
        results_df[metric_cols]
        .replace([np.inf, -np.inf], np.nan)
    )

    summary_df = pd.DataFrame({
        "mean": clean_metrics.mean(),
        "std": clean_metrics.std(),
        "valid": clean_metrics.count(),
        "invalid": clean_metrics.isna().sum()})

    # Aggregate feature importance.
    if valid_feature_explanations:
        mean_importance = (
            feature_sum / valid_feature_explanations)

        feature_importance_df = (
            pd.DataFrame({
                "feature": np.arange(data.num_features),
                "importance": (
                    mean_importance
                    .detach()
                    .cpu()
                    .numpy())})
            .sort_values(
                "importance",
                ascending=False)
            .reset_index(drop=True))

    else:
        feature_importance_df = pd.DataFrame(
            columns=["feature", "importance"])

    return (
        results_df,
        summary_df,
        feature_importance_df,
        sample_indices,
        explanations)


def get_local_feature_importance(explanation, node_idx: int):
    """
    Returns feature importance for one explained node.
    """
    node_mask = explanation.node_mask.detach().cpu()

    if node_mask.ndim == 1:
        return node_mask.numpy()

    if node_mask.ndim == 2:
        return node_mask[node_idx].numpy()

    raise ValueError(f"Unexpected node_mask shape: {tuple(node_mask.shape)}")


def get_top_k_features(feature_importance, k=10, feature_names=None):
    """
    Returns top-k most important features.
    """
    feature_importance = np.asarray(feature_importance)
    top_idx = np.argsort(feature_importance)[::-1][:k]

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(feature_importance))]

    return pd.DataFrame({
        "feature_idx": top_idx,
        "feature_name": [feature_names[i] for i in top_idx],
        "importance": feature_importance[top_idx],
    })


def aggregate_feature_importance(explainer, data, node_indices):
    """
    Mean feature importance across a set of explained nodes.
    """
    all_importance = []

    for node_idx in node_indices:
        node_idx = int(node_idx)

        explanation = explainer(data.x, data.edge_index, index=node_idx)
        importance = get_local_feature_importance(explanation, node_idx=node_idx)

        all_importance.append(importance)

    all_importance = np.asarray(np.vstack(all_importance), dtype=np.float32)
    mean_importance = all_importance.mean(axis=0)

    return mean_importance, all_importance


def _save_chart(chart, fig_dir, name):
    """Save an Altair chart as HTML and SVG under fig_dir when both are given."""
    if fig_dir is None or name is None:
        return
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    chart.save(fig_dir / f"{name}.html")
    chart.save(fig_dir / f"{name}.svg")


def plot_feature_importance(
    explanation,
    top_k=10,
    title="Top feature importance",
    fig_dir=None,
    name=None,
):
    node_mask = explanation.node_mask.detach().cpu().numpy()

    if node_mask.ndim == 2:
        importance = node_mask.sum(axis=0)
    else:
        importance = node_mask

    top_idx = np.argsort(importance)[::-1][:top_k]

    plot_df = pd.DataFrame({
        "feature": [f"Feature {i}" for i in top_idx],
        "importance": importance[top_idx],
    })

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "importance:Q",
                title="Importance",
            ),
            y=alt.Y(
                "feature:N",
                sort="-x",
                title=None,
            ),
            tooltip=[
                "feature:N",
                alt.Tooltip("importance:Q", format=".4f"),
            ],
        )
        .properties(
            width=500,
            height=300,
            title=title,
        )
    )

    _save_chart(chart, fig_dir, name)
    return chart


def plot_aggregate_feature_importance(
    feature_importance_df,
    top_k=10,
    title="Aggregate feature importance",
    fig_dir=None,
    name=None,
):
    plot_df = (
        feature_importance_df
        .head(top_k)
        .copy()
    )

    plot_df["feature"] = plot_df["feature"].astype(str)

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "importance:Q",
                title="Mean normalized importance",
            ),
            y=alt.Y(
                "feature:N",
                sort="-x",
                title="Feature",
            ),
            tooltip=[
                alt.Tooltip(
                    "feature:N",
                    title="Feature",
                ),
                alt.Tooltip(
                    "importance:Q",
                    title="Mean importance",
                    format=".4f",
                ),
            ],
        )
        .properties(
            width=550,
            height=350,
            title=title,
        )
    )

    _save_chart(chart, fig_dir, name)
    return chart


def get_node_sample_predictions(
    model,
    data,
    node_indices,
):
    """
    Return prediction, confidence, and correctness
    for the supplied nodes.
    """
    model.eval()

    node_indices = torch.as_tensor(
        node_indices,
        device=data.x.device,
    )

    with torch.no_grad():
        logits = model(
            data.x,
            data.edge_index,
        )

        probs = torch.softmax(
            logits[node_indices],
            dim=1,
        )

        confidence, predicted_class = probs.max(dim=1)

    df = pd.DataFrame({
        "node": node_indices.cpu().numpy(),
        "true_class": data.y[node_indices].cpu().numpy(),
        "predicted_class": predicted_class.cpu().numpy(),
        "confidence": confidence.cpu().numpy(),
    })

    df["correct"] = (
        df["true_class"]
        == df["predicted_class"])

    return df


def select_xai_examples(
    prediction_df,
    n=3,
    seed=123,
):
    """
    Sample representative XAI examples from confidence groups.
    """

    correct = prediction_df[
        prediction_df["correct"]
    ]

    incorrect = prediction_df[
        ~prediction_df["correct"]
    ]

    # Define confidence bands.
    correct_low = correct["confidence"].quantile(0.25)
    correct_high = correct["confidence"].quantile(0.75)

    incorrect_high = incorrect["confidence"].quantile(0.75)

    groups = {
        "High-confidence correct": correct[
            correct["confidence"] >= correct_high
        ],

        "Low-confidence correct": correct[
            correct["confidence"] <= correct_low
        ],

        "High-confidence incorrect": incorrect[
            incorrect["confidence"] >= incorrect_high
        ],
    }

    selected = {}

    for name, group in groups.items():
        selected[name] = group.sample(
            n=min(n, len(group)),
            random_state=seed,
        )

    return selected