"""Train one node-classification model on an OGB node dataset over several seeds.

Training runs full batch (``--no-dataloader``) or through a NeighborLoader
(default). Evaluation is full batch (``--eval full``, fine for ogbn-arxiv) or
layer-wise (``--eval layerwise``, required for ogbn-products, where the full
forward pass does not fit in memory).
"""
import os
import copy
import json
import time
import argparse
import warnings
from datetime import datetime

import pandas as pd

import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_focalloss import MultiClassFocalLoss

from src.common.datasets import load_ogb_node
from src.common.paths import get_project_root
from src.common.statistical_tests import summarise_metric
from src.common.utils import (compute_class_weights, create_dataloaders, evaluate,
                              get_device, inference_layerwise, set_seed,
                              train_one_epoch, train_one_epoch_loader)
from src.node_classification.node_models import GCN, GConv, GATV2, GraphSAGE, GraphSAGEBN


def build_model(args, num_feat, num_class):
    common = dict(num_layers=args.num_layers, in_feat=num_feat,
                  hid_feat=args.hidden_channels, num_classes=num_class,
                  dropout=args.dropout)
    if args.model == "GCN":
        return GCN(**common)
    if args.model == "GConv":
        return GConv(**common)
    if args.model == "GATV2":
        return GATV2(heads=args.heads, **common)
    if args.model == "SAGE":
        return GraphSAGE(**common)
    if args.model == "SAGEBN":
        return GraphSAGEBN(**common)
    raise ValueError(f"Invalid model name: {args.model}")


def build_criterion(args, data, num_class, device):
    if args.loss == "cross_entropy":
        return CrossEntropyLoss()
    weights = compute_class_weights(data.y, mask=data.train_mask, num_classes=num_class,
                                    power=args.weight_power).to(device)
    if args.loss == "weighted_ce":
        return CrossEntropyLoss(weight=weights)
    if args.loss == "focal":
        return MultiClassFocalLoss(gamma=args.gamma, weight=weights)
    raise ValueError(f"Invalid loss function: {args.loss}")


def main(args):
    print(f"Running {args.model} on {args.dataset} dataset")

    root_dir = get_project_root()
    device = get_device()

    checkpoint_dir = root_dir / "outputs" / "checkpoints" / args.dataset.lower()
    metrics_dir = root_dir / "outputs" / "metrics" / args.dataset.lower()
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    # One model is trained under several losses on the same dataset, so the
    # loss is part of every output name; nothing has to be moved between runs.
    # The post-training scripts (best_model.py) read the same <tag>_* names.
    tag = output_tag(args)

    # `data` stays on the CPU: the loaders sample there and layer-wise
    # inference reads from it. `data_dev` is a device copy for full-batch work.
    dataset, data, _ = load_ogb_node(name=args.dataset, root=root_dir / "data")
    num_feat, num_class = dataset.num_node_features, dataset.num_classes
    print(f"Device: {device}  Features: {num_feat}  Classes: {num_class}\n")

    if args.eval == "full" or not args.dataloader:
        data_dev = copy.copy(data).to(device)

    loaders = None
    if args.dataloader:
        fanout = args.fanout if len(args.fanout) > 1 else args.fanout[0]
        loaders = create_dataloaders(
            data, "neighbor", batch_size=args.batch_size, num_neighbors=fanout,
            num_layers=args.num_layers, balanced=args.balanced,
            num_workers=args.num_workers)

    model = build_model(args, num_feat, num_class).to(device)

    def evaluate_split(mask_name, **kw):
        if args.eval == "layerwise":
            logits = inference_layerwise(model, data, device, batch_size=args.eval_batch_size)
            return evaluate(data, model, criterion, getattr(data, mask_name), out=logits, **kw)
        return evaluate(data_dev, model, criterion, getattr(data_dev, mask_name), **kw)

    results_path = metrics_dir / f"{tag}_results.csv"
    history_path = metrics_dir / f"{tag}_history.csv"
    summary_path = metrics_dir / f"{tag}_summary.csv"
    config_path = metrics_dir / f"{tag}_config.json"

    # A run owns every <tag>_* file: a fresh run deletes them before seed 0 so
    # nothing left by an earlier attempt can be read as this run's result.
    # --resume (opt-in) keeps the finished seeds of an interrupted run with the
    # same flags; it compares flags only, not code, hence the banner.
    all_history, run_results, done_seeds = [], [], set()
    resumed = False
    if args.resume:
        problem = resume_problem(config_path, results_path, args)
        if problem is None:
            with open(config_path) as f:
                previous = json.load(f)
            run_results = pd.read_csv(results_path).to_dict("records")
            all_history = pd.read_csv(history_path).to_dict("records")
            done_seeds = {int(r["seed"]) for r in run_results}
            resumed = True
            print("=" * 100)
            print(f"RESUMING {tag}: {len(done_seeds)} seed(s) carried over from the run "
                  f"started {previous.get('started')} (same flags; code changes are NOT checked)")
            for r in run_results:
                print(f"  seed {int(r['seed']):>6}  acc={r['acc']:.4f}  finished {r['finished']}")
            print(f"  seeds still to train: {[s for s in args.seeds if s not in done_seeds]}")
            print("=" * 100)
        elif not results_path.exists():
            # No seed finished (a checkpoint saved mid-seed cannot be continued):
            # nothing worth keeping, so the leftovers go the way of a fresh run.
            print(f"Nothing to resume for {tag} ({problem}); starting fresh")
        else:
            # --resume was asked for and cannot be honoured: stop rather than
            # delete the partial seeds. The user drops --resume to start over.
            raise SystemExit(f"Cannot resume {tag}: {problem}\n"
                             f"Existing {tag}_* files were left untouched. Fix the flags, or "
                             f"drop --resume to start over (deletes them).")
    if not resumed:
        removed = clear_tag_outputs(metrics_dir, checkpoint_dir, args.dataset.lower(), tag)
        if removed:
            print(f"Fresh run: removed {len(removed)} old {tag} file(s):")
            for path in removed:
                print(f"  {path}")

    run_started = datetime.now()
    run_t0 = time.time()
    write_run_config(config_path, args, status="running", started=run_started)

    for seed in args.seeds:
        if seed in done_seeds:
            print(f"Seed {seed} already finished, skipping")
            continue
        seed_started = datetime.now()
        seed_t0 = time.time()
        set_seed(seed)
        model.reset_parameters()
        # AdamW decouples the decay from Adam's gradient scaling; its useful
        # coefficients (1e-2 .. 1e-1) are far larger than Adam's L2 (5e-4).
        optimizer_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
        optimizer = optimizer_cls(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
        criterion = build_criterion(args, data, num_class, device)
     
        scheduler = (ReduceLROnPlateau(optimizer, "min", factor=0.5, 
                                       patience=args.scheduler_patience,
                                       threshold=args.scheduler_threshold,
                                       min_lr=args.lr * 1e-3)
                     if args.scheduler else None)

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        best_model = None
        best_checkpoint_path = checkpoint_dir / f"{args.dataset.lower()}_{tag}_{seed}_best.pth"

        for epoch in range(1, 1 + args.epochs):
            if loaders is not None:
                train_metrics = train_one_epoch_loader(
                    loaders["train"], model, criterion, optimizer, device)
            else:
                train_metrics = train_one_epoch(
                    data_dev, model, criterion, optimizer, data_dev.train_mask)

            val_metrics = evaluate_split("val_mask")

            if scheduler is not None:
                scheduler.step(val_metrics.loss)

            all_history.append(train_metrics.to_row(epoch=epoch, split="train", seed=seed))
            all_history.append(val_metrics.to_row(epoch=epoch, split="val", seed=seed))

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
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            if patience_counter >= args.early_stop:
                print(f"Early stopping at epoch {epoch} "
                      f"(no val_loss improvement for {args.early_stop} epochs)")
                break

        if best_model is not None:
            model.load_state_dict(best_model)
            test_metrics = evaluate_split("test_mask", include_auc=True, include_cal=True)
            seed_minutes = (time.time() - seed_t0) / 60
            run_results.append({"seed": seed,
                                "best_epoch": best_epoch,
                                "val_loss": best_val_loss,
                                **test_metrics.to_row(),
                                "started": seed_started.isoformat(timespec="seconds"),
                                "finished": datetime.now().isoformat(timespec="seconds"),
                                "minutes": round(seed_minutes, 2)})
            # Persist after every seed, summary included, so every file on
            # disk always describes this run's finished seeds and nothing else.
            pd.DataFrame(run_results).to_csv(results_path, index=False)
            pd.DataFrame(all_history).to_csv(history_path, index=False)
            summary_df = write_summary(run_results, summary_path)
            print(f"Completed runs with seed: {seed}  | "
                  f"best_epoch={best_epoch}  |  "
                  f"val_loss={best_val_loss:.4f}  |  "
                  f"test_acc={test_metrics.acc:.4f}  |  "
                  f"test_f1={test_metrics.f1:.4f}  |  "
                  f"{seed_minutes:.1f} min")
            print(f"Summary over {len(run_results)} seed(s): "
                  f"test_acc mean={summary_df.loc['Test acc', 'mean']:.4f}  -> {summary_path}")
            print("=" * 100)

    history_df = pd.DataFrame(all_history)
    results_df = pd.DataFrame(run_results).sort_values("seed").reset_index(drop=True)
    results_df.to_csv(results_path, index=False)
    history_df.to_csv(history_path, index=False)
    summary_df = write_summary(results_df.to_dict("records"), summary_path)

    total_minutes = (time.time() - run_t0) / 60
    write_run_config(config_path, args, status="done", started=run_started,
                     finished=datetime.now(), minutes=total_minutes,
                     best_epoch_per_seed={int(r.seed): int(r.best_epoch)
                                          for r in results_df.itertuples()},
                     minutes_per_seed={int(r.seed): float(r.minutes)
                                       for r in results_df.itertuples()})
    print(f"Run {tag} finished: {len(results_df)} seeds, "
          f"{total_minutes:.1f} min this session, "
          f"{results_df['minutes'].sum():.1f} min of training in total")

    return summary_df, history_df


def output_tag(args) -> str:
    """<model>_<loss>, plus _p<power> when weights are tempered, plus _h<width>
    for SAGE/SAGEBN off their usual 256 (GATV2's width varies by design, per
    --heads, so it is never suffixed here; a GATV2 width experiment needs its
    own scheme)."""
    tag = f"{args.model.lower()}_{args.loss}"
    if args.loss != "cross_entropy" and args.weight_power != 1.0:
        tag += f"_p{args.weight_power:g}"
    if args.model in ("SAGE", "SAGEBN") and args.hidden_channels != 256:
        tag += f"_h{args.hidden_channels}"
    return tag


SUMMARY_METRICS = ["loss", "acc", "precision", "recall", "f1", "balanced_accuracy",
                   "brier", "roc_auc", "avg_precision", "ece"]


def write_summary(run_results: list[dict], path) -> pd.DataFrame:
    """Mean / std / 95% t-interval of every test metric over the finished seeds."""
    results_df = pd.DataFrame(run_results)
    with warnings.catch_warnings():
        # one finished seed: std and CI are NaN, numpy warns about ddof
        warnings.simplefilter("ignore", RuntimeWarning)
        summary = {f"Test {metric}": summarise_metric(results_df[metric], confidence=0.95)
                   for metric in SUMMARY_METRICS}
    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(path, index=True)
    return summary_df


def clear_tag_outputs(metrics_dir, checkpoint_dir, dataset: str, tag: str) -> list:
    """Delete every file a previous run of exactly this tag left behind.

    Names are matched exactly (checkpoints as <dataset>_<tag>_<digits>_best.pth),
    so ``sage_cross_entropy`` never touches ``sage_cross_entropy_f1sched``.
    """
    removed = []
    for suffix in ("results.csv", "history.csv", "summary.csv", "config.json"):
        path = metrics_dir / f"{tag}_{suffix}"
        if path.exists():
            path.unlink()
            removed.append(path)
    prefix, suffix = f"{dataset}_{tag}_", "_best.pth"
    for path in sorted(checkpoint_dir.iterdir()):
        name = path.name
        if (path.is_file() and name.startswith(prefix) and name.endswith(suffix)
                and name[len(prefix):-len(suffix)].isdigit()):
            path.unlink()
            removed.append(path)
    return removed


def resumable_config(args) -> dict:
    """The flags that define a run; two runs with equal dicts may share outputs."""
    return {k: v for k, v in vars(args).items()
            if k not in ("resume", "log_steps", "num_workers")}


def resume_problem(config_path, results_path, args):
    """Why the outputs at these paths cannot continue under ``args``; None if they can.

    Used by ``main`` for ``--resume`` and by the notebook as a pre-flight check,
    so a mismatch is visible before anything is deleted. A missing results file
    means no seed finished: the caller may start fresh, there is nothing to keep.
    """
    if not results_path.exists():
        return f"{results_path.name} is missing (no seed finished)"
    if not config_path.exists():
        return f"{config_path.name} is missing"
    with open(config_path) as f:
        previous = json.load(f)
    status = previous.get("status")
    if status == "done":
        return f"the run already finished on {previous.get('finished')}"
    if status != "running":
        return f"unknown status {status!r} in {config_path.name}"
    old, new = previous.get("config") or {}, resumable_config(args)
    diff = {k: (old.get(k, "<absent>"), new.get(k, "<absent>"))
            for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)}
    if diff:
        lines = [f"  --{k}: on disk {a!r}, now {b!r}" for k, (a, b) in diff.items()]
        return "the flags differ from the interrupted run\n" + "\n".join(lines)
    return None


def write_run_config(path, args, status: str, started: datetime, finished=None,
                     minutes=None, **extra) -> None:
    payload = {"status": status,
               "config": resumable_config(args),
               "started": started.isoformat(timespec="seconds"),
               "finished": finished.isoformat(timespec="seconds") if finished else None,
               "minutes": round(minutes, 2) if minutes is not None else None,
               **extra}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def validate_args(parser, args):
    """Reject contradictory flag combinations and warn about ignored ones.

    argparse cannot express these dependencies, so they are checked here
    after parsing. Hard errors go through ``parser.error`` (exit code 2).
    """
    def is_set(name):
        return getattr(args, name) != parser.get_default(name)

    if args.dataloader and len(args.fanout) > 1 and len(args.fanout) != args.num_layers:
        parser.error(f"--fanout has {len(args.fanout)} values but --num_layers is "
                     f"{args.num_layers}; give one value or one per layer")

    if not args.dataloader:
        if args.eval == "layerwise":
            parser.error("--no-dataloader already needs the full graph on the device; "
                         "--eval layerwise adds nothing but time. Use --eval full")
        ignored = [f"--{n}" for n in ("batch_size", "fanout", "balanced", "num_workers")
                   if is_set(n)]
        if ignored:
            print(f"Warning: {', '.join(ignored)} ignored with --no-dataloader")

    if args.eval == "full" and is_set("eval_batch_size"):
        print("Warning: --eval_batch_size ignored with --eval full")
    if args.model != "GATV2" and is_set("heads"):
        print("Warning: --heads ignored; only GATV2 uses it")
    if args.loss == "cross_entropy" and is_set("weight_power"):
        print("Warning: --weight_power ignored with --loss cross_entropy")
    if args.loss != "focal" and is_set("gamma"):
        print("Warning: --gamma ignored; only --loss focal uses it")

    if args.balanced and args.loss != "cross_entropy":
        print("Warning: --balanced resamples the training seeds AND the loss is "
              "class-weighted; the imbalance is corrected twice.")

    if args.dataset == "ogbn-products" and args.eval == "full":
        print("Warning: full-batch evaluation of ogbn-products may not fit in memory; "
              "consider --eval layerwise")

    patience_given = args.scheduler_patience is not None
    if not patience_given:
        args.scheduler_patience = max(1, args.early_stop // 2)
    if args.scheduler:
        if args.scheduler_patience < 1:
            parser.error("--scheduler_patience must be at least 1")
        if args.scheduler_patience >= args.early_stop:
            parser.error(f"--scheduler_patience ({args.scheduler_patience}) must be smaller "
                         f"than --early_stop ({args.early_stop}), otherwise early stopping "
                         "ends the run before the learning rate is ever reduced")
    elif patience_given or is_set("scheduler_metric"):
        print("Warning: --scheduler_patience / --scheduler_metric ignored with --no-scheduler")
    return args


def build_parser() -> argparse.ArgumentParser:
    """CLI of this script; shared with the notebook runner (``run_from_flags``)."""
    parser = argparse.ArgumentParser(description="OGB node classification")
    parser.add_argument("--dataset", choices=["ogbn-arxiv", "ogbn-products"],
                        default="ogbn-arxiv")
    parser.add_argument("--model", choices=["GCN", "GATV2", "GConv", "SAGE", "SAGEBN"],
                        default="SAGE", help="SAGEBN adds BatchNorm after every conv")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--heads", type=int, default=8, help="GATV2 only")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam",
                        help="adamw applies weight decay directly to the weights")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early_stop", type=int, default=50)
    parser.add_argument("--scheduler", action=argparse.BooleanOptionalAction, default=True,
                        help="ReduceLROnPlateau on --scheduler_metric: halves the LR after "
                             "--scheduler_patience stagnant epochs, floor lr/1000")
    parser.add_argument("--scheduler_metric", choices=["loss", "f1"], default="loss",
                        help="validation quantity the scheduler watches: loss (minimised, the "
                             "same signal as early stopping) or f1 (macro F1, maximised)")
    parser.add_argument("--scheduler_threshold", type=float, default=0.005,
                        help="relative change that counts as an improvement for the scheduler "
                             "(0.005 = half a percent; PyTorch default 1e-4 never fires on noisy F1)")
    parser.add_argument("--scheduler_patience", type=int, default=None,
                        help="stagnant epochs before the LR is reduced; must be smaller than "
                             "--early_stop (default: early_stop // 2)")
    parser.add_argument("--loss", choices=["cross_entropy", "weighted_ce", "focal"],
                        default="cross_entropy")
    parser.add_argument("--gamma", type=float, default=2.0, help="focal loss exponent")
    parser.add_argument("--weight_power", type=float, default=1.0,
                        help="temper the inverse-frequency class weights of weighted_ce / focal: "
                             "1 = balanced (sklearn), 0.5 = square root; outputs get the tag "
                             "suffix _p<power> when it is not 1")
    # Mini-batching
    parser.add_argument("--dataloader", action=argparse.BooleanOptionalAction, default=True,
                        help="train with a NeighborLoader; --no-dataloader trains full batch")
    parser.add_argument("--batch_size", type=int, default=1024, help="seed nodes per batch")
    parser.add_argument("--fanout", type=int, nargs="+", default=[15, 10],
                        help="neighbours sampled per layer, e.g. --fanout 15 10 5; "
                             "one value is repeated for every layer")
    parser.add_argument("--balanced", action=argparse.BooleanOptionalAction, default=False,
                        help="resample training seeds by inverse class frequency")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval", choices=["full", "layerwise"], default="full",
                        help="full-batch evaluation, or layer-wise for graphs that don't fit")
    parser.add_argument("--eval_batch_size", type=int, default=4096, help="layer-wise eval only")
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False,
                        help="opt in to keep the seeds already in <tag>_results.csv when the "
                             "previous run of the same flags was interrupted (e.g. a dead VM). "
                             "Default: start over and delete every old <tag>_* file first")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 20, 42, 123, 1234, 12345],
                        help="one run per seed; keep the same list for every model and variant "
                             "you compare, the seed-level tests pair rows by seed")
    return parser


def run_from_flags(flags: str | list[str]):
    """Parse a flag string exactly as the CLI would and train; for notebooks.

    Example: ``run_from_flags("--model SAGE --loss focal --no-dataloader")``.
    Returns ``(summary_df, history_df)`` like ``main``.
    """
    import shlex
    argv = shlex.split(flags) if isinstance(flags, str) else list(flags)
    parser = build_parser()
    args = validate_args(parser, parser.parse_args(argv))
    print(args)
    return main(args)


if __name__ == "__main__":
    parser = build_parser()
    args = validate_args(parser, parser.parse_args())
    print(args)

    metrics_summary, history = main(args)
    print(metrics_summary)
