"""Statistical comparison of the trained node-classification models.

Node level: loads the best-seed checkpoint of every model found in
outputs/metrics/<dataset>, predicts the test nodes, and runs Cochran's Q plus
pairwise McNemar tests on the per-node correct/incorrect outcomes. The
node-level table reports macro F1 and balanced accuracy alongside accuracy;
the paired tests themselves operate on per-node correctness, so they compare
accuracy by construction.
Seed level: Friedman, Kruskal-Wallis and pairwise Wilcoxon on the per-seed
test scores from <model>_results.csv (default metrics: f1, balanced_accuracy).

Usage (from the project root, after training):
    uv run python src/node_classification/compare_best.py --dataset Cora
    uv run python src/node_classification/compare_best.py --dataset ogbn-arxiv \
        --tags sage_cross_entropy sage_weighted_ce_p0.5 sage_weighted_ce --label sage_losses
Writes test_predictions[_<label>].csv, comparison_mcnemar[_<label>].csv,
comparison_seeds_<metric>[_<label>].csv to outputs/metrics/<dataset> and a
summary to docs/<dataset>_model_comparison[_<label>].md. --tags restricts the
comparison to the given tags (default: every tag with a summary CSV); the
OGB grid has architecture and loss variants mixed in one folder, so a
loss-level comparison (one architecture, several losses) and a model-level
one (several architectures, one loss each) need --tags to stay meaningful,
and --label keeps their output files from overwriting each other.
"""
import argparse

import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.common.paths import get_project_root
from src.common.utils import get_device
from src.common import statistical_tests as st
from src.node_classification.best_model import (load_summaries, select_best_seed,
                                                load_best_checkpoint, load_dataset)


def predict_test_nodes(model, data) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    return logits.argmax(dim=1)[data.test_mask].cpu()


def node_level_scores(predictions: pd.DataFrame, tags: list[str]) -> pd.DataFrame:
    """Accuracy, macro F1 and balanced accuracy of each model on the test nodes."""
    y_true = predictions["true"].to_numpy()
    rows = []
    for tag in tags:
        y_pred = predictions[f"pred_{tag}"].to_numpy()
        rows.append({"model": tag,
                     "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                     "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                     "accuracy": predictions[f"correct_{tag}"].mean()})
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    # itertuples keeps per-column dtypes; iterrows would upcast int columns
    # (e.g. seed) to float when the rest of the row is float.
    for row in df.itertuples(index=False):
        cells = [f"{v:{floatfmt}}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(args):
    root = get_project_root()
    name = args.dataset.lower()
    metrics_dir = root / "outputs" / "metrics" / name
    checkpoint_dir = root / "outputs" / "checkpoints" / name
    device = get_device()
    suffix = f"_{args.label}" if args.label else ""

    table = load_summaries(metrics_dir)
    if args.tags:
        missing = [t for t in args.tags if t not in table.index]
        if missing:
            raise ValueError(f"--tags {missing} not found in {metrics_dir}; "
                             f"available: {list(table.index)}")
        tags = args.tags
    else:
        tags = list(table.index)
    num_features, num_classes, data = load_dataset(args.dataset, root)
    data = data.to(device)
    y_test = data.y[data.test_mask].cpu()

    # ---- node level -------------------------------------------------------
    predictions = pd.DataFrame({"node": data.test_mask.nonzero().view(-1).cpu().numpy(),
                                "true": y_test.numpy()})
    seeds = {}
    for tag in tags:
        seeds[tag] = select_best_seed(metrics_dir, tag)
        model, _ = load_best_checkpoint(checkpoint_dir, args.dataset, tag, seeds[tag],
                                        num_features, num_classes, device)
        predictions[f"pred_{tag}"] = predict_test_nodes(model, data).numpy()
        predictions[f"correct_{tag}"] = predictions[f"pred_{tag}"] == predictions["true"]
    predictions.to_csv(metrics_dir / f"test_predictions{suffix}.csv", index=False)

    node_scores = node_level_scores(predictions, tags)
    correct = predictions[[f"correct_{t}" for t in tags]].rename(columns=lambda c: c[8:])
    q = st.cochran_q_test(correct.to_numpy())
    mcnemar = st.pairwise_mcnemar(correct, exact=not args.chi2)
    mcnemar.to_csv(metrics_dir / f"comparison_mcnemar{suffix}.csv", index=False)

    print(f"Best seeds: {seeds}")
    print(f"\nTest scores of the best-seed checkpoints ({len(correct)} nodes):")
    print(node_scores.round(4).to_string(index=False))
    print(f"\nCochran's Q = {q['statistic']:.3f}, df = {q['df']}, p = {q['p_value']:.4g}")
    print("\nPairwise McNemar (Holm-adjusted):")
    print(mcnemar.round(4).to_string(index=False))

    # ---- seed level -------------------------------------------------------
    seed_sections = []
    for metric in args.seed_metrics:
        scores = pd.DataFrame({tag: pd.read_csv(metrics_dir / f"{tag}_results.csv")
                               .set_index("seed")[metric] for tag in tags})
        friedman = st.friedman_test(scores)
        kruskal = st.kruskal_test(scores)
        wilcoxon = st.pairwise_wilcoxon(scores)
        wilcoxon.to_csv(metrics_dir / f"comparison_seeds_{metric}{suffix}.csv", index=False)
        print(f"\nSeed-level {metric} ({len(scores)} seeds): "
              f"Friedman p = {friedman['p_value']:.4g}, Kruskal-Wallis p = {kruskal['p_value']:.4g}")
        print(wilcoxon.round(4).to_string(index=False))
        seed_sections.append((metric, scores, friedman, kruskal, wilcoxon))

    # ---- markdown ---------------------------------------------------------
    lines = [f"# {args.dataset} model comparison", "",
             "Generated by `src/node_classification/compare_best.py`.", "",
             "## Node level (best-seed checkpoint per model, paired on test nodes)", "",
             "Best seed by validation loss: " + ", ".join(f"{t} = {s}" for t, s in seeds.items()),
             "", markdown_table(node_scores), "",
             "Cochran's Q and McNemar are paired tests on per-node correctness, so they",
             "compare accuracy; macro F1 and balanced accuracy are reported for the same",
             "checkpoints but have no per-node analogue.", "",
             f"Cochran's Q = {q['statistic']:.3f} (df {q['df']}), p = {q['p_value']:.4g}", "",
             "### Pairwise McNemar" + (" (chi-square)" if args.chi2 else " (exact)"), "",
             markdown_table(mcnemar), "",
             "## Seed level (one score per seed per model)", "",
             "With five seeds an exact Wilcoxon test cannot go below p = 0.0625, so",
             "these are descriptive only.", ""]
    for metric, scores, friedman, kruskal, wilcoxon in seed_sections:
        lines += [f"### {metric}", "",
                  markdown_table(scores.reset_index()), "",
                  f"Friedman p = {friedman['p_value']:.4g}; Kruskal-Wallis p = {kruskal['p_value']:.4g}",
                  "", markdown_table(wilcoxon), ""]
    out = root / "docs" / f"{name}_model_comparison{suffix}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"\nwrote {out.relative_to(root)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare trained node-classification models")
    parser.add_argument("--dataset", type=str, default="Cora",
                        help="Cora, CiteSeer, PubMed, ogbn-arxiv, ogbn-products")
    parser.add_argument("--tags", nargs="+", default=None,
                        help="Restrict the comparison to these tags (default: every tag with "
                             "a summary CSV). The OGB grid mixes architectures and losses in "
                             "one folder, e.g. --tags sage_cross_entropy sagebn_cross_entropy "
                             "gatv2_cross_entropy for a model-level comparison, or "
                             "--tags sage_cross_entropy sage_weighted_ce_p0.5 sage_weighted_ce "
                             "for a loss-level one.")
    parser.add_argument("--label", type=str, default=None,
                        help="Suffix for output filenames, so different --tags subsets (e.g. "
                             "a loss-level and a model-level comparison) don't overwrite each "
                             "other's files.")
    parser.add_argument("--seed_metrics", nargs="+", default=["f1", "balanced_accuracy"],
                        help="Columns of <model>_results.csv to compare across seeds "
                             "(e.g. f1 balanced_accuracy acc roc_auc)")
    parser.add_argument("--chi2", action="store_true",
                        help="Use the chi-square McNemar instead of the exact test")
    main(parser.parse_args())
