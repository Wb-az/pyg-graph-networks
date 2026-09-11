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


# --- resume pre-flight -------------------------------------------------------
import json

from src.node_classification.ogb_run import (build_parser, resume_problem, resumable_config,
                                             validate_args, write_run_config)
from datetime import datetime

FLAGS = "--dataset ogbn-arxiv --model SAGE --num_layers 3 --no-scheduler --no-dataloader"


def _args(flags=FLAGS):
    parser = build_parser()
    return validate_args(parser, parser.parse_args(flags.split()))


def _interrupted(tmp_path, flags=FLAGS, status="running", seeds=(0, 1)):
    config, results = tmp_path / "t_config.json", tmp_path / "t_results.csv"
    write_run_config(config, _args(flags), status=status, started=datetime.now())
    pd.DataFrame([_row(s, 0.7) for s in seeds]).to_csv(results, index=False)
    return config, results


def test_resume_problem_none_for_same_flags(tmp_path):
    config, results = _interrupted(tmp_path)
    assert resume_problem(config, results, _args(FLAGS + " --resume --log_steps 5")) is None


def test_resume_problem_reports_each_differing_flag(tmp_path):
    config, results = _interrupted(tmp_path)
    problem = resume_problem(config, results, _args(FLAGS + " --lr 0.01 --hidden_channels 64"))
    assert "--lr" in problem and "--hidden_channels" in problem
    assert "--dropout" not in problem


def test_resume_problem_finished_run(tmp_path):
    config, results = _interrupted(tmp_path, status="done")
    assert "already finished" in resume_problem(config, results, _args())


def test_resume_problem_missing_files(tmp_path):
    config, results = tmp_path / "t_config.json", tmp_path / "t_results.csv"
    # no finished seed comes first: with a config but no results there is
    # nothing to keep, and main() starts fresh instead of refusing
    assert "no seed finished" in resume_problem(config, results, _args())
    write_run_config(config, _args(), status="running", started=datetime.now())
    assert "no seed finished" in resume_problem(config, results, _args())
    pd.DataFrame([_row(0, 0.7)]).to_csv(results, index=False)
    config.unlink()
    assert "config" in resume_problem(config, results, _args())


def test_resume_problem_config_written_by_older_script(tmp_path):
    config, results = _interrupted(tmp_path)
    payload = json.loads(config.read_text())
    del payload["config"]["weight_power"]           # key the old script did not know
    config.write_text(json.dumps(payload))
    assert "--weight_power" in resume_problem(config, results, _args())


def test_resumable_config_survives_json_roundtrip():
    cfg = resumable_config(_args())
    assert json.loads(json.dumps(cfg)) == cfg


def test_output_tag_h_suffix_only_for_sage_off_256():
    from src.node_classification.ogb_run import output_tag
    def tag(model, hidden):
        return output_tag(_args(f"--dataset ogbn-arxiv --model {model} --num_layers 3 "
                                f"--hidden_channels {hidden} --no-scheduler --no-dataloader "
                                f"--loss cross_entropy"))
    assert tag("SAGE", 256) == "sage_cross_entropy"          # usual width: no suffix
    assert tag("SAGE", 128) == "sage_cross_entropy_h128"
    assert tag("SAGEBN", 128) == "sagebn_cross_entropy_h128"
    assert tag("GATV2", 32) == "gatv2_cross_entropy"          # GATV2 width never suffixed
