import pandas as pd
import pytest

from src.node_classification.ogb_run import clear_tag_outputs, write_summary, SUMMARY_METRICS


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    return path


def test_clear_tag_outputs_removes_only_exact_tag(tmp_path):
    metrics, ckpt = tmp_path / "metrics", tmp_path / "checkpoints"
    mine = [
        _touch(metrics / "sage_cross_entropy_results.csv"),
        _touch(metrics / "sage_cross_entropy_history.csv"),
        _touch(metrics / "sage_cross_entropy_summary.csv"),
        _touch(metrics / "sage_cross_entropy_config.json"),
        _touch(ckpt / "ogbn-arxiv_sage_cross_entropy_0_best.pth"),
        _touch(ckpt / "ogbn-arxiv_sage_cross_entropy_12345_best.pth"),
    ]
    others = [
        _touch(metrics / "sage_cross_entropy_f1sched_results.csv"),
        _touch(metrics / "sage_cross_entropy_f1sched_summary.csv"),
        _touch(metrics / "sagebn_cross_entropy_results.csv"),
        _touch(metrics / "sage_weighted_ce_p0.5_results.csv"),
        _touch(ckpt / "ogbn-arxiv_sage_cross_entropy_f1sched_0_best.pth"),
        _touch(ckpt / "ogbn-arxiv_sagebn_cross_entropy_0_best.pth"),
        _touch(ckpt / "ogbn-products_sage_cross_entropy_0_best.pth"),
    ]

    removed = clear_tag_outputs(metrics, ckpt, "ogbn-arxiv", "sage_cross_entropy")

    assert sorted(removed) == sorted(mine)
    assert not any(p.exists() for p in mine)
    assert all(p.exists() for p in others)


def test_clear_tag_outputs_on_empty_dirs(tmp_path):
    metrics, ckpt = tmp_path / "metrics", tmp_path / "checkpoints"
    metrics.mkdir()
    ckpt.mkdir()
    assert clear_tag_outputs(metrics, ckpt, "ogbn-arxiv", "sage_cross_entropy") == []


def _row(seed, acc):
    row = {metric: acc for metric in SUMMARY_METRICS}
    row.update(seed=seed, best_epoch=1)
    return row


def test_write_summary_single_seed_has_mean_only(tmp_path, recwarn):
    path = tmp_path / "s.csv"
    summary = write_summary([_row(0, 0.7)], path)
    assert summary.loc["Test acc", "mean"] == pytest.approx(0.7)
    assert pd.isna(summary.loc["Test acc", "std"])
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
    assert path.exists()


def test_write_summary_reflects_only_given_rows(tmp_path):
    path = tmp_path / "s.csv"
    write_summary([_row(0, 0.1)], path)
    summary = write_summary([_row(0, 0.7), _row(1, 0.71)], path)
    assert summary.loc["Test acc", "mean"] == pytest.approx(0.705)
    on_disk = pd.read_csv(path, index_col=0)
    assert on_disk.loc["Test acc", "mean"] == pytest.approx(0.705)
