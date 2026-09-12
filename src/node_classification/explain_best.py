"""GNNExplainer evaluation and example figures for the best model.

Usage (from the project root, after training):
    uv run python src/node_classification/explain_best.py --dataset Cora --metric f1
    uv run python src/node_classification/explain_best.py --dataset ogbn-arxiv --model SAGEBN \
        --loss cross_entropy --max_samples 100
Writes per-node metrics, a summary and the aggregate feature importance to
outputs/metrics/<dataset>, and figures to outputs/figures/<dataset>. On a dataset
whose test set exceeds MAX_SAMPLES_THRESHOLD, --max_samples must be given
explicitly: GNNExplainer runs its own --explainer_epochs optimisation per node,
so explaining every test node (e.g. 48,603 for ogbn-arxiv) by default is
impractical the way it is for Cora's ~1,000.
"""
import argparse

import matplotlib
matplotlib.use("Agg")

from src.common.paths import get_project_root
from src.common.utils import get_device
from src.common import explainability as xai
from src.node_classification.best_model import (add_selection_args, resolve_model,
                                                select_best_seed, load_best_checkpoint,
                                                load_dataset, label_map_for, figures_dirname)

MAX_SAMPLES_THRESHOLD = 5000


def slugify(text: str) -> str:
    return text.lower().replace("-", "_").replace(" ", "_")


def main(args):
    root = get_project_root()
    name = args.dataset.lower()
    metrics_dir = root / "outputs" / "metrics" / name
    checkpoint_dir = root / "outputs" / "checkpoints" / name
    fig_dir = root / "outputs" / "figures" / figures_dirname(args.dataset)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tag, _ = resolve_model(args, metrics_dir)
    seed = select_best_seed(metrics_dir, tag)
    print(f"Best seed by validation loss: {seed}\n")

    device = get_device()
    num_features, num_classes, data = load_dataset(args.dataset, root)
    labels = label_map_for(args.dataset, num_classes)
    data = data.to(device)
    model, config = load_best_checkpoint(checkpoint_dir, args.dataset, tag, seed,
                                         num_features, num_classes, device)

    explainer = xai.create_explainer(model=model, num_features=num_features,
                                     method="gnn_explainer", epochs=args.explainer_epochs)

    n_test = int(data.test_mask.sum())
    if args.max_samples is None and n_test > MAX_SAMPLES_THRESHOLD:
        raise ValueError(
            f"{args.dataset} has {n_test} test nodes; --max_samples must be set "
            f"explicitly above {MAX_SAMPLES_THRESHOLD} test nodes, GNNExplainer runs "
            f"its own {args.explainer_epochs}-epoch optimisation per node, unbounded "
            "is impractical here. Pass e.g. --max_samples 100.")
    max_samples = n_test if args.max_samples is None else min(args.max_samples, n_test)
    print(f"Explaining {max_samples} of {n_test} test nodes with GNNExplainer "
          f"({args.explainer_epochs} epochs each)...")
    results, summary, feat_importance, sampled, explanations = xai.eval_explanations(
        explainer=explainer, data=data, max_samples=max_samples, seed=args.seed)

    results.to_csv(metrics_dir / f"{tag}_xai_results.csv", index=False)
    summary.to_csv(metrics_dir / f"{tag}_xai_summary.csv")
    feat_importance.to_csv(metrics_dir / f"{tag}_xai_feature_importance.csv", index=False)
    print("\nExplanation quality (mean over sampled nodes):")
    print(summary.round(4).to_string())

    xai.plot_aggregate_feature_importance(
        feat_importance, top_k=args.top_k,
        title=f"{config['model']} aggregate feature importance",
        fig_dir=fig_dir, name=f"{tag}_xai_aggregate_importance")
    print(f"\nSaved {tag}_xai_aggregate_importance.html/.svg")

    # One or more example nodes per confidence group.
    examples = xai.select_xai_examples(results, n=args.n_examples, seed=args.seed)
    for group_name, group_df in examples.items():
        for _, row in group_df.iterrows():
            node_idx = int(row["node"])
            stem = f"{tag}_xai_{slugify(group_name)}_node{node_idx}"
            print(f"\n{group_name}: node {node_idx}  "
                  f"true={labels[int(row['true_class'])]}  "
                  f"pred={labels[int(row['predicted_class'])]}  "
                  f"conf={row['confidence']:.3f}")
            explanation = explanations[node_idx]
            xai.plot_feature_importance(
                explanation, top_k=args.top_k,
                title=f"{group_name} — node {node_idx}",
                fig_dir=fig_dir, name=f"{stem}_features")
            explanation.visualize_graph(path=str(fig_dir / f"{stem}_subgraph.png"))
            print(f"  saved {stem}_features.html/.svg and {stem}_subgraph.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explain the best node-classification model")
    add_selection_args(parser)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Test nodes to explain (default: all)")
    parser.add_argument("--explainer_epochs", type=int, default=100)
    parser.add_argument("--n_examples", type=int, default=1,
                        help="Example nodes per confidence group")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    main(parser.parse_args())
