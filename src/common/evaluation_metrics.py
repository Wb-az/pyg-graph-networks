from __future__ import annotations
from typing import NamedTuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score)

def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Accepts a torch.Tensor or a numpy array/array-like and returns numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class Metrics(NamedTuple):
    """
    Stores classification metrics for one data split and evaluation step.
    Brier score is computed whenever class probabilities are provided.
    ROC-AUC and average precision are optional because multiclass
    one-vs-rest evaluation is more expensive and is better suited to
    final-model assessment than per-epoch tracking.
    Use calibration metrics and reliability plots alongside Brier score
    when assessing probability calibration.
    """
    loss: float
    acc: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    brier: float | None = None
    roc_auc: float | None = None
    avg_precision: float | None = None
    ece: float| None = None

    @classmethod
    def compute(
        cls,
        y_true: torch.Tensor | np.ndarray,
        y_pred: torch.Tensor | np.ndarray,
        loss: float,
        y_proba: torch.Tensor | np.ndarray | None = None,
        include_auc: bool = False,
        include_cal: bool = False,
        average: str = "macro",
    ) -> "Metrics":
        """Computes metrics for one split and returns a Metrics instance.

        :param include_cal:
        :param y_true: Ground-truth class indices, shape [N]. Tensor or numpy.
        :param y_pred: Predicted class indices (argmax of logits), shape [N].
        :param loss: Scalar loss for this split/epoch.
        :param y_proba: Class probabilities, shape [N, num_classes]. Enables
            `brier` whenever given; also required when include_auc=True.
        :param include_auc: Whether to also compute roc_auc/avg precision (one-vs-rest,
            macro-averaged). Expensive on many classes/nodes — reserve for the
            final best-checkpoint evaluation, not every epoch.
        :param include_cal: Whether to also compute calibration metrics.
        :param average: Averaging strategy for precision/recall/f1/auc
            (default "macro", appropriate for class-imbalanced datasets like
            ogbn-arxiv).
        """
        y_true_np = _to_numpy(y_true)
        y_pred_np = _to_numpy(y_pred)
        y_proba_np = _to_numpy(y_proba) if y_proba is not None else None

        if len(y_true_np) == 0:
            raise ValueError("Cannot compute metrics on an empty split.")

        brier = None
        if y_proba_np is not None:
            # Use all probability columns so classes absent from this split
            # are still represented consistently in the multiclass Brier score.
            brier = brier_score_loss(
                y_true_np,
                y_proba_np,
                labels=list(range(y_proba_np.shape[1])),
                scale_by_half=True)

        roc_auc = None
        average_precision: float | None = None
        if include_auc:
            if y_proba_np is None:
                raise ValueError("y_proba is required when include_auc=True.")
            present_classes = np.unique(y_true_np)

            if len(present_classes) < 2:
                raise ValueError(
                    "ROC-AUC and average precision require at least two classes."
                )

            y_true_bin = np.column_stack([
                (y_true_np == cls).astype(int)
                for cls in present_classes])
            y_score = y_proba_np[:, present_classes]

            roc_auc = roc_auc_score(y_true_bin, y_score, average=average)
            average_precision = average_precision_score(
                y_true_bin,
                y_score,
                average=average)

        ece:float | None = None
        if include_cal:
            cal = CalibrationMetrics.compute(y_true, y_proba)
            ece = cal.ece

        return cls(
            loss=loss,
            acc=accuracy_score(y_true_np, y_pred_np),
            precision=precision_score(y_true_np, y_pred_np, average=average, zero_division=0),
            recall=recall_score(y_true_np, y_pred_np, average=average, zero_division=0),
            f1=f1_score(y_true_np, y_pred_np, average=average, zero_division=0),
            balanced_accuracy=balanced_accuracy_score(y_true_np, y_pred_np),
            brier=brier,
            roc_auc=roc_auc,
            avg_precision=average_precision,
            ece=ece
        )


    def to_row(self, **extra) -> dict:
        """Flattens this record into a plain dict tagged with extra context
        (e.g. epoch=.., split=..) -- for accumulating per-epoch history
        across a training run into a pandas.DataFrame
        """
        return {**extra, **self._asdict()}


class CalibrationMetrics(NamedTuple):
    """
    Measures top-label calibration using Expected Calibration Error (ECE)
    and reliability-diagram bins. Compares predicted confidence with observed
    accuracy across confidence bins. Intended for final-model evaluation alongside
    the Brier score and reliability plots, rather than as a training-loop metric.
    """
    ece: float
    bin_confidence: list[float]  # mean predicted confidence per bin
    bin_accuracy: list[float]  # observed accuracy per bin
    bin_count: list[int]  # number of samples per bin

    @classmethod
    def compute(cls, y_true: torch.Tensor | np.ndarray,
        y_proba: torch.Tensor | np.ndarray, n_bins: int = 10) -> "CalibrationMetrics":
        """
        Top-label ECE: bins samples by their predicted (argmax) class
        confidence into `n_bins` equal-width bins, then measures the
        weighted-average gap between each bin's mean confidence and its
        observed accuracy.
        :param y_true: Ground-truth class indices, shape [N].
        :param y_proba: Class probabilities, shape [N, num_classes].
        :param n_bins: Number of equal-width confidence bins (10 is the
            standard choice in the calibration literature).
        :return: CalibrationMetrics
        """
        if n_bins < 1:
            raise ValueError("n_bins must be >= 1.")

        y_true_np = np.asarray(_to_numpy(y_true), dtype=np.int64)
        y_proba_np = np.asarray(_to_numpy(y_proba), dtype=np.float64)

        if len(y_true_np) != y_proba_np.shape[0]:
            raise ValueError(
                "y_true and y_proba must contain the same number of samples."
            )

        confidence = y_proba_np.max(axis=1)
        correct = (y_proba_np.argmax(axis=1) == y_true_np).astype(float)

        n = len(y_true_np)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_confidence, bin_accuracy, bin_count = [], [], []
        ece = 0.0

        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            in_bin = (confidence >= lo) & (confidence <= hi if i == n_bins - 1 else confidence < hi)
            count = int(in_bin.sum())
            avg_conf = float(confidence[in_bin].mean()) if count else np.nan
            avg_acc = float(correct[in_bin].mean()) if count else np.nan
            if count:
                ece += (count / n) * abs(avg_acc - avg_conf)
            bin_confidence.append(avg_conf)
            bin_accuracy.append(avg_acc)
            bin_count.append(count)

        return cls(ece=float(ece), bin_confidence=bin_confidence,
                   bin_accuracy=bin_accuracy, bin_count=bin_count)


class EmbeddingMetrics(NamedTuple):
    """
    Measures class separation in the model's learned node embeddings.
    Complements embedding visualizations with a quantitative silhouette score.
    Intended for final-model evaluation; use sampling for large datasets due
    to the cost of pairwise distance calculations.
    """
    silhouette: float

    @classmethod
    def compute(
        cls,
        embeddings: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
        metric: str = "cosine",
        sample_size: int | None = None,
        random_state: int | None = None) -> "EmbeddingMetrics":

        embeddings_np = _to_numpy(embeddings)
        labels_np = _to_numpy(labels)

        return cls(
            silhouette=silhouette_score(
                embeddings_np, labels_np, metric=metric,
                sample_size=sample_size, random_state=random_state,
            )
        )