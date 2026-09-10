import os
import copy
import argparse

import pandas as pd
import numpy as np

import torch
from torch.nn import CrossEntropyLoss

from src.common.datasets import load_planetoid
from src.common.paths import get_project_root
from src.common.statistical_tests import summarise_metric
from src.common.utils import evaluate, get_device, train_one_epoch, set_seed
from src.node_classification.node_models import GCN, GConv, GATV2, GraphSAGE


def main(args):

    print(f"Running {args.model} on {args.dataset} dataset")

    root_dir = get_project_root()
    device = get_device()

    checkpoint_dir = root_dir / "outputs" / "checkpoints" / args.dataset.lower()
    metrics_dir = root_dir / "outputs" / "metrics" / args.dataset.lower()
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    dataset, data = load_planetoid(path=root_dir / "data", dataset_name=args.dataset)
    data = data.to(device)

    num_feat, num_class = dataset.num_node_features, dataset.num_classes

    print(f"Device: {device}  Features: {num_feat}  Classes: {num_class}\n")

    if args.model == "GCN":
        model = GCN(num_layers=args.num_layers, in_feat=num_feat, hid_feat=args.hidden_channels,
                     num_classes=num_class, dropout=args.dropout).to(device)

    elif args.model == "GConv":
        model = GConv(num_layers=args.num_layers, in_feat=num_feat, hid_feat=args.hidden_channels,
                     num_classes=num_class, dropout=args.dropout).to(device)

    elif args.model == "GATV2":
        model = GATV2(num_layers=args.num_layers, in_feat=num_feat,
                      hid_feat=args.hidden_channels,
                     num_classes=num_class, heads=args.heads,
                      dropout=args.dropout).to(device)

    elif args.model == "SAGE":
        model = GraphSAGE(num_layers=args.num_layers, in_feat=num_feat,
                          hid_feat=args.hidden_channels,
                          num_classes=num_class,
                          dropout=args.dropout).to(device)
    else:
        raise ValueError("Invalid model name")

    all_history = []
    run_results = []

    for seed in args.seeds:
        set_seed(seed)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        criterion = CrossEntropyLoss()

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        early_stop = args.early_stop
        best_model = None

        best_checkpoint_path = checkpoint_dir / f"{args.dataset.lower()}_{args.model.lower()}_{seed}_best.pth"

        for epoch in range(1, 1 + args.epochs):
            train_metrics = train_one_epoch(
                data,
                model,
                criterion,
                optimizer,
                data.train_mask)

            val_metrics = evaluate(
                data,
                model,
                criterion,
                data.val_mask)

            all_history.append(train_metrics.to_row(
                epoch=epoch, split="train", seed=seed))

            all_history.append(val_metrics.to_row(
                    epoch=epoch,
                    split="val",
                    seed=seed))

            if val_metrics.loss < best_val_loss:
                best_val_loss = val_metrics.loss
                best_epoch = epoch
                best_model = copy.deepcopy(model.state_dict())

                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "seed": seed,
                        "val_loss": best_val_loss,
                        "config": vars(args).copy(),
                    },
                    best_checkpoint_path,
                )
                patience_counter = 0

            else:
                patience_counter += 1

            if epoch % args.log_steps == 0:
                print(
                    f"Seed: {seed}  |  "
                    f"epoch {epoch:03d}  |  "
                    f"train_loss={train_metrics.loss:.4f}  |  "
                    f"train_acc={train_metrics.acc:.4f}  |  "
                    f"val_loss={val_metrics.loss:.4f}  |  "
                    f"val_acc={val_metrics.acc:.4f}  |  "
                    # f"lr={optimizer.param_groups[0]['lr']:.2e}  |  "
                )

            if patience_counter >= early_stop:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no val_loss improvement for "
                    f"{early_stop} epochs)")
                break

        if best_model is not None:
            model.load_state_dict(best_model)
            test_metrics = evaluate(data, model, criterion, data.test_mask,
                                    include_auc=True,
                                    include_cal=True)
            run_results.append({"seed": seed,
            "best_epoch": best_epoch,
            "val_loss": best_val_loss,
            **test_metrics.to_row()})

            print(f"Completed runs with seed: {seed}  | "
                f"best_epoch={best_epoch}  |  "
                f"val_loss={best_val_loss:.4f}  |  "
                f"test_acc={test_metrics.acc:.4f}"
            )
            print("=========================="*4)

    history_df = pd.DataFrame(all_history)
    results_df = pd.DataFrame(run_results)
    results_df.to_csv(
        metrics_dir / f"{args.model.lower()}_results.csv",
        index=False)

    history_df.to_csv(
        metrics_dir / f"{args.model.lower()}_history.csv",
        index=False)

    summary_metrics = ["loss", "acc", "precision", "recall",
                       "f1", "balanced_accuracy", "brier",
                       "roc_auc", "avg_precision", "ece"]

    summary = {
        f"Test {metric}": summarise_metric(results_df[metric], confidence=0.95)
        for metric in summary_metrics}

    summary_df = pd.DataFrame(summary).T

    summary_df.to_csv(
        metrics_dir / f"{args.model.lower()}_summary.csv",
        index=True,
    )

    return summary_df, history_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Node classification')
    parser.add_argument("--dataset",
                        choices=["Cora", "CiteSeer", "PubMed"], default="Cora")
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument("--model", choices=["GCN", "GATV2", "GConv", "SAGE"],
                        default="GCN")
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=16,
                        help="number of hidden channels features")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--early_stop', type=int, default=12)
    args = parser.parse_args()
    print(args)

    metrics_summary, history = main(args)

    print(metrics_summary)




