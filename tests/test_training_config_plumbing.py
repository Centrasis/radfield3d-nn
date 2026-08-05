"""Training-config plumbing: YAML dataset path, MLflow tracking target, cache min-age guard.

CPU-only, no dataset needed — exercises the config surface of run_network_task.py and the
MLflow URI handling of loggers/mlflow.py.
"""
import os
import subprocess
import sys
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── collect_cacheable_files: min-age guard ────────────────────────────────────

def _touch(path, age_s, now):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x")
    os.utime(path, (now - age_s, now - age_s))


def test_cache_min_age_skips_files_under_simulation(tmp_path):
    sys.path.insert(0, REPO)
    from run_network_task import collect_cacheable_files
    ds = str(tmp_path / "ds")
    now = time.time()
    _touch(os.path.join(ds, "fields", "old.rf3"), age_s=3600, now=now)
    _touch(os.path.join(ds, "fields", "fresh.rf3"), age_s=5, now=now)   # simulation still writing
    _touch(os.path.join(ds, "statistics.json"), age_s=3600, now=now)

    files, skipped = collect_cacheable_files(ds, min_age_s=60, now=now)
    assert skipped == 1
    assert sorted(files) == [os.path.join("fields", "old.rf3"), "statistics.json"]

    # 0 disables the guard entirely
    files, skipped = collect_cacheable_files(ds, min_age_s=0, now=now)
    assert skipped == 0 and len(files) == 3


# ── logger sub-section resolution ─────────────────────────────────────────────

def _resolve(train_cfg):
    sys.path.insert(0, REPO)
    from run_network_task import resolve_logger_config
    return resolve_logger_config(train_cfg)


def test_logger_section_is_read():
    got = _resolve({"logger": {"type": "MLflow", "mlflow_tracking_uri": "/data/mlflow",
                               "project_name": "proj", "run_name": "run-1", "offline": True}})
    assert got == {"type": "mlflow", "mlflow_tracking_uri": "/data/mlflow",
                   "project_name": "proj", "run_name": "run-1", "offline": True}


def test_logger_section_defaults():
    got = _resolve({"logger": {"type": "mlflow"}})
    assert got["project_name"] == "radiation-field-estimator"
    assert got["run_name"] is None and got["offline"] is False
    assert got["mlflow_tracking_uri"] is None


def test_legacy_flat_logger_form_still_works():
    # `training: logger: wandb` + sibling keys — the pre-section layout.
    got = _resolve({"logger": "wandb", "project_name": "old-proj", "run_name": "old-run",
                    "offline": True, "mlflow_tracking_uri": "/old/mlflow"})
    assert got == {"type": "wandb", "project_name": "old-proj", "run_name": "old-run",
                   "offline": True, "mlflow_tracking_uri": "/old/mlflow"}


def test_section_wins_over_flat_keys_but_flat_fills_gaps():
    got = _resolve({"logger": {"type": "mlflow", "project_name": "new"},
                    "project_name": "old", "run_name": "flat-run"})
    assert got["project_name"] == "new"      # section wins
    assert got["run_name"] == "flat-run"     # not in the section -> flat key fills in


def test_no_logger_block_defaults_to_wandb():
    assert _resolve({})["type"] == "wandb"


# ── MLflow tracking target ────────────────────────────────────────────────────

mlflow = pytest.importorskip("mlflow")


def _mk(logs_dir, project="proj"):
    sys.path.insert(0, REPO)
    from loggers.mlflow import MLFlowLogger
    return MLFlowLogger(project_name=project, logs_dir=logs_dir)


def test_mlflow_local_dir_becomes_file_uri_with_project_subdir(tmp_path):
    lg = _mk(str(tmp_path / "mlflow"))
    assert lg.tracking_uri == f"file://{tmp_path / 'mlflow'}/proj"


def test_mlflow_server_url_stays_untouched():
    # A tracking SERVER groups by experiment; a /project path suffix would 404.
    lg = _mk("http://mlflow.example.org:5000")
    assert lg.tracking_uri == "http://mlflow.example.org:5000"


def test_mlflow_explicit_file_uri_keeps_project_subdir(tmp_path):
    lg = _mk(f"file://{tmp_path}/store")
    assert lg.tracking_uri == f"file://{tmp_path}/store/proj"


# ── run_network_task.py: dataset path from YAML vs CLI ────────────────────────

@pytest.mark.slow
def test_dataset_path_required_from_yaml_or_cli(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("training: {}\ndataset: {}\n")
    r = subprocess.run([sys.executable, os.path.join(REPO, "run_network_task.py"), str(cfg),
                        "--logs_path", str(tmp_path / "logs")],
                       capture_output=True, text=True, cwd=REPO, timeout=600)
    assert r.returncode != 0
    assert "no dataset given" in (r.stdout + r.stderr)


@pytest.mark.slow
def test_dataset_path_accepted_from_yaml(tmp_path):
    # With `dataset: path:` set the dataset check passes; the run then fails LATER on the
    # (intentionally missing) model_config — proving the YAML path was consumed.
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"training: {{}}\ndataset: {{path: {tmp_path / 'ds'}}}\n")
    (tmp_path / "ds").mkdir()
    r = subprocess.run([sys.executable, os.path.join(REPO, "run_network_task.py"), str(cfg),
                        "--logs_path", str(tmp_path / "logs")],
                       capture_output=True, text=True, cwd=REPO, timeout=600)
    assert r.returncode != 0
    out = r.stdout + r.stderr
    assert "no dataset given" not in out
    assert "model_config" in out
