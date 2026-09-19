import importlib.util
import pathlib

import numpy as np
import pytest

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "capture_qpu_reads.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("capture_qpu_reads", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture_qpu_reads = _load_module()


def test_atomic_savez_compressed_leaves_no_partial_file_at_final_path(tmp_path):
    path = tmp_path / "nonce.npz"
    capture_qpu_reads.atomic_savez_compressed(
        str(path), spins=np.ones((2, 3), dtype=np.int8), access_us=np.int64(5)
    )
    assert path.exists()
    loaded = np.load(path)
    assert loaded["spins"].tolist() == [[1, 1, 1], [1, 1, 1]]
    assert int(loaded["access_us"]) == 5
    # No stray temp files left behind that resume could mistake for a capture.
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_savez_compressed_cleans_up_temp_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "nonce.npz"

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(capture_qpu_reads.np, "savez_compressed", _boom)
    with pytest.raises(RuntimeError):
        capture_qpu_reads.atomic_savez_compressed(str(path), spins=np.ones((1, 1)))
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_check_budget_allows_estimate_under_the_cap():
    capture_qpu_reads.check_budget(1, max_qpu_seconds=30.0)


def test_check_budget_refuses_estimate_over_the_cap():
    with pytest.raises(SystemExit):
        capture_qpu_reads.check_budget(1000, max_qpu_seconds=30.0)


def test_dry_run_never_constructs_or_connects_a_sampler(tmp_path, monkeypatch):
    heavy = tmp_path / "heavy.csv"
    heavy.write_text("set,nonce\nlow-energy,ab\n", encoding="utf-8")

    class ExplodingSampler:
        def __init__(self, *args, **kwargs):
            raise AssertionError("OceanSampler must not be constructed under --dry-run")

    monkeypatch.setattr(capture_qpu_reads, "OceanSampler", ExplodingSampler)
    monkeypatch.setattr(
        capture_qpu_reads.replay, "load_spec", lambda _path: object()
    )
    monkeypatch.setattr(
        capture_qpu_reads.sys, "argv",
        [
            "capture_qpu_reads.py",
            "--spec", "spec.json",
            "--heavy", str(heavy),
            "--out-dir", str(tmp_path / "out"),
            "--dry-run",
        ],
    )
    assert capture_qpu_reads.main() == 0
