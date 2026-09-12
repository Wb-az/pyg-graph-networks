"""Training curves, embedding projection and silhouette scores for the best model.

Usage (from the project root, after training):
    uv run python src/node_classification/visualise_best.py --dataset Cora --metric f1
    uv run python src/node_classification/visualise_best.py --dataset ogbn-arxiv --model SAGE --loss cross_entropy
Outputs go to outputs/figures/<dataset> and outputs/metrics/<dataset>. ogbn-arxiv's
40 classes have no curated names here (see best_model.label_map_for), so the
embedding plot's legend falls back to "class <i>".
"""
import argparse

import pandas as pd

from src.common.paths import get_project_root
from src.common.utils import get_device
from src.common.evaluation_metrics import EmbeddingMetrics
from src.common.training_visualisations import (plot_metric_curves, compute_embedding,
                                                plot_embeddings_3d, get_embeddings)
from src.node_classification.best_model import (add_selection_args, resolve_model,
                                                select_best_seed, load_best_checkpoint,
                                                load_dataset, label_map_for, figures_dirname)


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

    # Training curves for every seed of the chosen model.
    history = pd.read_csv(metrics_dir / f"{tag}_history.csv")
    plot_metric_curves(history, fig_dir, name=tag, metrics=("loss", "acc", "f1"))
    print(f"Saved {tag}_training_curve.html/.svg")

    # Rebuild the model from its checkpoint.
    device = get_device()
    num_features, num_classes, data = load_dataset(args.dataset, root)
    data = data.to(device)
    model, config = load_best_checkpoint(checkpoint_dir, args.dataset, tag, seed,
                                         num_features, num_classes, device)
    print(f"Loaded {config['model']} checkpoint (hidden={config['hidden_channels']}, "
          f"lr={config['lr']}, dropout={config['dropout']})")

    # Embedding projection.
    projection = compute_embedding(model, data.x, data.edge_index,
                                   method=args.embedding, n_components=3)
    plot_embeddings_3d(projection, data.y, label_map_for(args.dataset, num_classes), fig_dir,
                       name=f"{tag}_{args.embedding}",
                       title=f"{args.dataset} node embeddings ({config['model']})")
    print(f"Saved {tag}_{args.embedding}_embedding.html/.svg")

    # Silhouette on the raw embeddings, cosine distance.
    embeddings = get_embeddings(model, data.x, data.edge_index)
    scores = {
        "all_nodes": EmbeddingMetrics.compute(embeddings, data.y, metric="cosine").silhouette,
        "test_nodes": EmbeddingMetrics.compute(embeddings[data.test_mask],
                                               data.y[data.test_mask],
                                               metric="cosine").silhouette,
    }
    scores_df = pd.DataFrame({"silhouette_cosine": scores})
    scores_df.to_csv(metrics_dir / f"{tag}_embedding_metrics.csv")
    print("\nSilhouette (cosine):")
    print(scores_df.round(4).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise the best node-classification model")
    add_selection_args(parser)
    parser.add_argument("--embedding", choices=["tsne", "pca"], default="tsne")
    main(parser.parse_args())
