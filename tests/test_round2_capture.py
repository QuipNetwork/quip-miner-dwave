"""The audited capture runner of Round 2: manifest, approval, dry run, capture.

Fakes only -- no live sampler is ever constructed here, and no real QPU
problem is ever submitted. ``--mock`` uses the recording double below; the
"real" path (``--mock`` absent) still runs against the same double, injected
through ``round2_capture.connect`` exactly as ``test_regime_capture.py`` does
for Round 1's ``regime_capture.py``, so the accounting and guard logic is
exercised without a network call.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from quip_miner_dwave import regime_io, round2_io
from quip_miner_dwave.ocean import SampleResult

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "round2_capture.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("round2_capture", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations via sys.modules[cls.__module__]; the
    # module must be registered there before exec_module runs the decorator.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


round2_capture = _load_module()

PROPERTIES = {
    "chip_id": "mock-chip",
    "problem_timing_data": {
        "version": "1.0.0",
        "typical_programming_time": 33631.59,
        "reverse_annealing_with_reinit_prog_time_delta": 0.0,
        "reverse_annealing_without_reinit_prog_time_delta": 31.33,
        "default_programming_thermalization": 1000.0,
        "default_annealing_time": 20.0,
        "readout_time_model": "pwl_log_log",
        "readout_time_model_parameters": [
            0.0, 0.7118156236873552, 1.622919864592155, 2.5460233863267376, 3.6610550848533783,
            1.3562242666114566, 1.6328679675785214, 1.7857781469507206, 1.99055000334358, 1.9905608299940198,
        ],
        "qpu_delay_time_per_sample": 60.57,
        "reverse_annealing_with_reinit_delay_time_delta": -44.5,
        "reverse_annealing_without_reinit_delay_time_delta": -41.5,
        "default_readout_thermalization": 0.0,
        "decorrelation_max_nominal_anneal_time": 2000.0,
        "decorrelation_time_range": [0.0, 0.0],
    },
}

VARIABLE_LABELS = [100, 101, 102]
BUNDLE_EDGES = np.array([[0, 1], [1, 2]], dtype=np.int64)


def _nonce(i: int) -> str:
    return format(i, "064x")


def _write_bundle(bundles_root, cell, nonce, h, j, offset=0.0):
    manifest = {
        "cell": cell,
        "nonce": nonce,
        "identity": {"topology_hash": "topo-fixture", "model_order_hash": "order-fixture"},
        "variable_labels": list(VARIABLE_LABELS),
        "offset": offset,
    }
    return round2_io.write_bundle(
        bundles_root / cell / nonce, manifest, {"h": h, "edges": BUNDLE_EDGES, "j": j}
    )


def _write_range_audit(range_audit_dir, cell, nonce, scales, ceiling=2.0):
    """``scales``: a list of ``(requested, submitted_scale, h_scaled, j_scaled)``."""
    model_dir = range_audit_dir / cell / nonce
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cell": cell, "nonce": nonce, "status": "ok",
        "j_range_used": "j_range", "ceiling": ceiling, "all_zero": False,
        "limiting_constraint": {"kind": "h", "label": 100, "value": 0.3, "bound": "upper"},
        "coefficient_stats": {}, "group_sums": {},
        "chain_strength": None, "chain_strength_ratio_to_logical": None,
        "scales": [{"requested": r, "submitted_scale": s} for r, s, _, _ in scales],
    }
    regime_io.atomic_write_json(model_dir / "manifest.json", manifest)
    arrays = {}
    for requested, _submitted, h_scaled, j_scaled in scales:
        suffix = round2_capture._scale_suffix(requested)
        arrays[f"h_{suffix}"] = h_scaled
        arrays[f"j_{suffix}"] = j_scaled
    regime_io.atomic_savez(model_dir / "arrays.npz", **arrays)


def _fixture(tmp_path, nonces=("aa",), cell="native-pm1", scale=0.5, submitted_scale=1.0):
    """One cell with the given nonces, each bundled and range-audited at ``scale``."""
    h = np.array([0.1, -0.2, 0.3])
    j = np.array([0.4, -0.5])
    for nonce in nonces:
        _write_bundle(tmp_path / "bundles", cell, nonce, h, j)
        _write_range_audit(
            tmp_path / "range-audit", cell, nonce,
            [(scale, submitted_scale, h * submitted_scale, j * submitted_scale)],
        )
    (tmp_path / "properties.json").write_text(json.dumps(PROPERTIES), encoding="utf-8")
    return tmp_path


def _job_spec(tmp_path, nonces=("aa",), cell="native-pm1", scale=0.5, anneal_us=(80,), reads=8,
              order_seed=1, margin=1.5):
    return {
        "cells": [cell],
        "nonces": {cell: list(nonces)},
        "requested_scales": [scale],
        "anneal_us": list(anneal_us),
        "reads": reads,
        "order_seed": order_seed,
        "bundles_root": str(tmp_path / "bundles"),
        "range_audit_dir": str(tmp_path / "range-audit"),
        "properties_path": str(tmp_path / "properties.json"),
        "spec_path": None,
        "embeddings_dir": None,
        "reservation_margin": margin,
    }


def _write_job_spec(tmp_path, **kwargs) -> pathlib.Path:
    path = tmp_path / "job-spec.json"
    path.write_text(json.dumps(_job_spec(tmp_path, **kwargs)), encoding="utf-8")
    return path


class _Sampler:
    """A double of ``OceanSampler``: answers each job with +1 spins at a fixed access time."""

    def __init__(self, access_us=40_000, live_nodes=VARIABLE_LABELS, live_edges=((100, 101), (101, 102)),
                 rows=None, during_job=None):
        self.access_us = access_us
        self._live_nodes = list(live_nodes)
        self._live_edges = list(live_edges)
        self.rows = rows
        self.during_job = during_job
        self.calls = []
        self.closed = False

    @property
    def live_nodes(self):
        return list(self._live_nodes)

    @property
    def live_edges(self):
        return list(self._live_edges)

    def ensure_connected(self):
        pass

    def set_session_topology(self, nodes, edges):
        pass

    def sample(self, nodes, h, edges, j, *, num_reads, anneal_time_us, nonce_seed, label, auto_scale=None):
        assert auto_scale is False, "Round 2 must always submit with auto_scale=False"
        self.calls.append({"nodes": list(nodes), "nonce_seed": nonce_seed.hex(), "anneal_time_us": anneal_time_us})
        if self.during_job is not None:
            self.during_job()
        rows = num_reads if self.rows is None else self.rows
        return SampleResult(
            spins=np.ones((rows, len(nodes)), dtype=np.int8),
            variables=[int(q) for q in nodes],
            energies=[0.0] * rows,
            device_access_time_us=self.access_us,
            num_reads=rows,
        )

    def close(self):
        self.closed = True


def _run(argv):
    return round2_capture.main(argv)


# ------------------------------------------------------------- pure helpers


def test_estimate_access_us_matches_the_sdks_own_formula():
    from dwave.cloud.solver import StructuredSolver

    class _PropertyHolder:
        def __init__(self, properties):
            self.properties = properties

    stand_in = _PropertyHolder(PROPERTIES)
    expected = StructuredSolver.estimate_qpu_access_time(
        stand_in,  # type: ignore[arg-type]  # duck-typed stand-in, offline on purpose
        num_qubits=3, num_reads=8, annealing_time=80.0,
    )

    assert round2_capture.estimate_access_us(PROPERTIES, 3, 8, 80) == pytest.approx(expected)


def test_estimate_access_us_reports_the_limit_before_approval_when_it_cannot_bound_one():
    with pytest.raises(ValueError, match="cannot bound QPU access time"):
        round2_capture.estimate_access_us({}, 3, 8, 80)


def test_coefficient_hash_changes_with_either_array():
    h, j = np.array([1.0, 2.0]), np.array([0.5])
    base = round2_capture.coefficient_hash(h, j)
    assert round2_capture.coefficient_hash(h * 2, j) != base
    assert round2_capture.coefficient_hash(h, j * 2) != base


# ---------------------------------------------------------------- manifest


def test_build_manifest_writes_every_job_with_a_seeded_order_and_its_own_hash(tmp_path):
    _fixture(tmp_path, nonces=(_nonce(1), _nonce(2)))
    job_spec = _job_spec(tmp_path, nonces=(_nonce(1), _nonce(2)), anneal_us=(80, 400))

    manifest = round2_capture.build_manifest(job_spec)

    assert len(manifest["jobs"]) == 4  # 2 nonces x 2 anneal times x 1 scale
    assert sorted(manifest["job_order"]) == sorted(job["capture_key"] for job in manifest["jobs"])
    assert manifest["job_order_hash"] == round2_capture.hashlib.sha256(
        json.dumps(manifest["job_order"]).encode()
    ).hexdigest()
    job = manifest["jobs"][0]
    assert job["embedding_hash"] is None  # native-pm1 has no embedding
    assert job["requested_scale"] == 0.5
    assert job["submitted_scale"] == 1.0
    assert job["reservation_us"] == pytest.approx(job["estimate_us"] * job_spec["reservation_margin"])


def test_the_manifest_file_hash_is_its_own_raw_bytes(tmp_path):
    _fixture(tmp_path)
    job_spec = _write_job_spec(tmp_path)
    out = tmp_path / "manifest.json"

    assert _run(["manifest", "--job-spec", str(job_spec), "--out", str(out)]) == 0
    assert round2_capture.hashlib.sha256(out.read_bytes()).hexdigest()  # just must not raise


def test_two_jobs_that_collide_on_capture_key_refuse_to_build_a_manifest(tmp_path, monkeypatch):
    _fixture(tmp_path, nonces=(_nonce(1),))
    job_spec = _job_spec(tmp_path, nonces=(_nonce(1),))
    job_spec["anneal_us"] = [80, 400]

    # Force every job to carry the same capture key regardless of its fields,
    # without disturbing build_job's own hash-checked reads.
    real_build_job = round2_capture.build_job
    monkeypatch.setattr(
        round2_capture, "build_job",
        lambda *a, **k: {**real_build_job(*a, **k), "capture_key": "same-key"},
    )
    with pytest.raises(ValueError, match="same capture key"):
        round2_capture.build_manifest(job_spec)


# ----------------------------------------------------------------- approval


def _manifest_and_hash(tmp_path):
    _fixture(tmp_path)
    job_spec = _write_job_spec(tmp_path)
    out = tmp_path / "manifest.json"
    _run(["manifest", "--job-spec", str(job_spec), "--out", str(out)])
    return out, round2_capture.hashlib.sha256(out.read_bytes()).hexdigest()


def test_approve_refuses_a_manifest_hash_that_does_not_match_the_file(tmp_path):
    manifest_path, _real_hash = _manifest_and_hash(tmp_path)
    with pytest.raises(SystemExit, match="does not match"):
        _run([
            "approve", "--manifest", str(manifest_path), "--manifest-hash", "0" * 64,
            "--cap-seconds", "10", "--approver", "alice", "--out", str(tmp_path / "approval.json"),
        ])


@pytest.mark.parametrize("cap", ["nan", "inf", "0", "-5"])
def test_approve_refuses_a_cap_that_is_not_a_finite_positive_number(tmp_path, cap):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    with pytest.raises(SystemExit, match="finite positive"):
        _run([
            "approve", "--manifest", str(manifest_path), "--manifest-hash", manifest_hash,
            "--cap-seconds", cap, "--approver", "alice", "--out", str(tmp_path / "approval.json"),
        ])


def test_approve_requires_manifest_hash_and_cap_with_no_default(tmp_path, capsys):
    manifest_path, _hash = _manifest_and_hash(tmp_path)
    with pytest.raises(SystemExit):
        _run(["approve", "--manifest", str(manifest_path), "--approver", "alice", "--out", str(tmp_path / "a.json")])
    assert "required" in capsys.readouterr().err


def _approve(tmp_path, manifest_path, manifest_hash, cap_seconds=10.0, approver="alice"):
    approval_path = tmp_path / "approval.json"
    _run([
        "approve", "--manifest", str(manifest_path), "--manifest-hash", manifest_hash,
        "--cap-seconds", str(cap_seconds), "--approver", approver, "--out", str(approval_path),
    ])
    return approval_path


# ------------------------------------------------------------------ dry-run


def test_dry_run_reports_counts_and_reservations_without_submitting(tmp_path, capsys):
    manifest_path, _hash = _manifest_and_hash(tmp_path)
    round_dir = tmp_path / "round"

    assert _run(["dry-run", "--manifest", str(manifest_path), "--round-dir", str(round_dir)]) == 0
    out = capsys.readouterr().out
    assert "native-pm1: 1 jobs" in out
    assert "not a guaranteed hard bound" in out
    assert regime_io.read_spend(round_dir) == (0, 0)


# ------------------------------------------------------------------ capture


def _capture_argv(tmp_path, manifest_path, approval_path, round_dir, *extra):
    return [
        "capture", "--manifest", str(manifest_path), "--approval", str(approval_path),
        "--round-dir", str(round_dir), *extra,
    ]


def test_capture_refuses_without_an_approval_file(tmp_path, monkeypatch):
    manifest_path, _hash = _manifest_and_hash(tmp_path)
    monkeypatch.setattr(round2_capture, "connect", lambda mock: _Sampler())
    with pytest.raises(SystemExit, match="no approval"):
        _run(_capture_argv(tmp_path, manifest_path, tmp_path / "missing.json", tmp_path / "round", "--mock"))


def test_capture_refuses_an_approval_naming_a_different_manifest_hash(tmp_path, monkeypatch):
    manifest_path, _hash = _manifest_and_hash(tmp_path)
    # approve validates the hash against the manifest file itself, so a
    # mismatched approval has to be written directly.
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps({"manifest_hash": "f" * 64, "cap_seconds": 10.0}), encoding="utf-8")
    monkeypatch.setattr(round2_capture, "connect", lambda mock: _Sampler())
    with pytest.raises(SystemExit, match="different manifest"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, tmp_path / "round", "--mock"))


def test_capture_refuses_an_approval_with_no_cap(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps({"manifest_hash": manifest_hash}), encoding="utf-8")
    monkeypatch.setattr(round2_capture, "connect", lambda mock: _Sampler())
    with pytest.raises(SystemExit, match="names no cap"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, tmp_path / "round", "--mock"))


def test_a_mock_capture_runs_the_full_path_and_saves_every_field(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    sampler = _Sampler()
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)
    round_dir = tmp_path / "round"

    assert _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--mock")) == 0
    assert sampler.closed
    assert regime_io.read_spend(round_dir) == (0, 0)  # mock spends nothing

    files = list((round_dir / "native-pm1").glob("scale-*/qpu-*/*.npz"))
    assert len(files) == 1
    with np.load(files[0]) as capture:
        assert bool(capture["mock"]) is True
        assert bool(capture["auto_scale"]) is False
        assert float(capture["requested_scale"]) == 0.5
        assert float(capture["submitted_scale"]) == 1.0
        assert capture["spins"].shape == (8, 3)
        assert str(capture["coefficient_hash"])
        assert str(capture["request_bytes_hash"])
    mode_marker = json.loads((round_dir / round2_capture.MODE_MARKER_FILE).read_text())
    assert mode_marker == {"mock": True}


def test_a_mock_capture_is_idempotent_and_skips_a_finished_job(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    monkeypatch.setattr(round2_capture, "connect", lambda mock: _Sampler())
    assert _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--mock")) == 0

    second = _Sampler()
    monkeypatch.setattr(round2_capture, "connect", lambda mock: second)
    assert _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--mock")) == 0
    assert second.calls == []


def test_a_real_run_without_yes_submits_nothing(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    sampler = _Sampler()
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)
    round_dir = tmp_path / "round"

    assert _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir)) == 2
    assert sampler.calls == []
    assert regime_io.read_spend(round_dir) == (0, 0)


def test_a_capture_stops_before_the_job_that_would_pass_the_approved_cap(tmp_path, monkeypatch):
    nonces = (_nonce(1), _nonce(2), _nonce(3))
    _fixture(tmp_path, nonces=nonces)
    job_spec = _job_spec(tmp_path, nonces=nonces)
    manifest = round2_capture.build_manifest(job_spec)
    manifest_path = tmp_path / "manifest.json"
    regime_io.atomic_write_json(manifest_path, manifest)
    manifest_hash = round2_capture.hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    reservation_us = manifest["jobs"][0]["reservation_us"]
    # Two jobs charge 40,000 us each (below their reservation); the third
    # job's reservation alone would push the round over a cap set just above
    # what two charges plus one more reservation requires.
    cap_seconds = (2 * 40_000 + reservation_us) / 1e6 - 0.001
    approval_path = _approve(tmp_path, manifest_path, manifest_hash, cap_seconds=cap_seconds)
    sampler = _Sampler(access_us=40_000)
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)
    round_dir = tmp_path / "round"

    assert _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes")) == 3
    assert len(sampler.calls) == 2
    assert sampler.closed


def test_a_failed_submit_is_recorded_and_never_retried_automatically(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"

    def dropped():
        raise RuntimeError("the connection dropped")

    sampler = _Sampler(during_job=dropped)
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)
    with pytest.raises(RuntimeError, match="the connection dropped"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))

    failed = round2_capture.load_failed(round_dir)
    assert len(failed) == 1
    reason = next(iter(failed.values()))["reason"]
    assert "the connection dropped" in reason
    # An open reservation with no charge counts at its estimate.
    _spent, open_jobs = regime_io.read_spend(round_dir)
    assert open_jobs == 1

    # A rerun refuses the same capture key rather than resubmitting it.
    sampler2 = _Sampler()
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler2)
    with pytest.raises(SystemExit, match="never retries an uncertain submission"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))
    assert sampler2.calls == []


def test_a_short_answer_stops_the_run_after_the_charge_is_recorded(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    sampler = _Sampler(rows=2)  # the job asks for 8 reads
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)

    with pytest.raises(SystemExit, match="holds 2 rows"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))

    spent, open_jobs = regime_io.read_spend(round_dir)
    assert spent == sampler.access_us and open_jobs == 0  # the charge is recorded before the decode fails
    assert list((round_dir / "native-pm1").glob("scale-*/qpu-*/*.npz")) == []


def test_a_decode_failure_still_leaves_the_charge_recorded(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    sampler = _Sampler()
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)

    def _broken_decode(model, spins, variables):
        raise ValueError("the reads lack qubits of the model")

    monkeypatch.setattr(round2_capture.regimes, "decode_reads", _broken_decode)
    with pytest.raises(ValueError, match="the reads lack qubits"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))

    spent, open_jobs = regime_io.read_spend(round_dir)
    assert spent == sampler.access_us and open_jobs == 0
    assert list((round_dir / "native-pm1").glob("scale-*/qpu-*/*.npz")) == []


def test_a_changed_bundle_is_refused_before_any_reservation(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"

    # Replace the bundle after the manifest was built, as a rebuilt export
    # would (write_bundle itself refuses to mutate one in place).
    import shutil

    shutil.rmtree(tmp_path / "bundles" / "native-pm1" / "aa")
    _write_bundle(tmp_path / "bundles", "native-pm1", "aa", np.array([9.0, 9.0, 9.0]), np.array([9.0, 9.0]))

    sampler = _Sampler()
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)
    with pytest.raises(SystemExit, match="bundle has changed"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))
    assert sampler.calls == []
    assert regime_io.read_spend(round_dir) == (0, 0)


def test_preflight_stops_before_any_reservation_when_the_live_chip_lacks_a_coupler(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    sampler = _Sampler(live_edges=((100, 101),))  # missing (101, 102)
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)

    with pytest.raises(SystemExit, match="lacks 0 of the qubits and 1 of the couplers"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))
    assert sampler.calls == []
    assert regime_io.read_spend(round_dir) == (0, 0)
    assert sampler.closed


def test_a_mock_and_a_real_capture_must_not_share_a_round_directory(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    monkeypatch.setattr(round2_capture, "connect", lambda mock: _Sampler())
    assert _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--mock")) == 0

    monkeypatch.setattr(round2_capture, "connect", lambda mock: _Sampler())
    with pytest.raises(SystemExit, match="mock is True"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))


def test_a_capture_refuses_a_round_locked_by_another_process(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    round_dir.mkdir()

    code = (
        "import fcntl,sys; "
        'f=open(sys.argv[1], "a"); fcntl.flock(f, fcntl.LOCK_EX); '
        'print("locked", flush=True); sys.stdin.readline()'
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", code, str(round_dir / ".capture.lock")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "locked"
        monkeypatch.setattr(round2_capture, "connect", lambda mock: _Sampler())
        with pytest.raises(SystemExit, match="another capture"):
            _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--mock"))
    finally:
        holder.communicate("\n", timeout=10)


def test_the_output_records_the_scale_never_promises_returned_scaling_metadata(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    monkeypatch.setattr(round2_capture, "connect", lambda mock: _Sampler())
    assert _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--mock")) == 0

    [path] = list((round_dir / "native-pm1").glob("scale-*/qpu-*/*.npz"))
    with np.load(path) as capture:
        assert bool(capture["returned_scaling_present"]) is False
        assert "auto_scale=False" in str(capture["returned_scaling_reason"])
