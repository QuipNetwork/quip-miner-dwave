"""End-to-end conformance: miner vs the quip-solver-drive binary.

Spawns ``quip-solver-drive`` (the standalone CLI from the published
``quip-solver-conformance`` crate) against a small shell wrapper that launches
``python -m quip_miner_dwave`` with mock mode.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]  # worktree root
PYTHON = Path(sys.executable)
DRIVE_BIN = REPO / "rust" / "target" / "debug" / "quip-solver-drive"


def _ensure_drive_bin() -> Path:
    if DRIVE_BIN.is_file():
        return DRIVE_BIN
    # This end-to-end test drives the miner against quip-solver-drive, built
    # from the published quip-solver-conformance crate. That crate is not
    # vendored into this standalone Python repo, so skip (conformance against
    # it is exercised by the Rust miner repos that already depend on it).
    if not (REPO / "rust").is_dir():
        pytest.skip("quip-solver-conformance rust checkout not present (standalone repo)")
    # Build if missing
    cargo = shutil.which("cargo")
    if not cargo:
        pytest.skip("cargo not available to build quip-solver-drive")
    r = subprocess.run(
        [cargo, "build", "-p", "quip-solver-conformance", "--bin", "quip-solver-drive"],
        cwd=str(REPO / "rust"),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if r.returncode != 0:
        pytest.fail(f"build quip-solver-drive failed:\n{r.stderr}")
    assert DRIVE_BIN.is_file()
    return DRIVE_BIN


def _write_miner_wrapper(tmpdir: Path) -> Path:
    """Shell script quip-solver-drive can exec as a solver binary."""
    wrapper = tmpdir / "quip-dwave-qa"
    # Ensure python package path + mock mode
    script = f"""#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="{REPO / 'python'}${{PYTHONPATH:+:$PYTHONPATH}}"
export QUIP_DWAVE_MOCK=1
exec "{PYTHON}" -m quip_miner_dwave --mock "$@"
"""
    wrapper.write_text(script)
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return wrapper


def test_conformance_against_quip_solver_drive():
    drive = _ensure_drive_bin()
    with tempfile.TemporaryDirectory(prefix="quip-dwave-conf-") as td:
        tdp = Path(td)
        miner = _write_miner_wrapper(tdp)
        # quip-solver-drive CLI: quip-solver-drive <solver-bin> <unix://socket>
        env = os.environ.copy()
        env["QUIP_SESSION_TOKEN"] = "test-token"
        env["QUIP_DWAVE_MOCK"] = "1"
        env["PYTHONPATH"] = str(REPO / "python") + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        # quip-solver-drive prints report.summary() and exits 0 iff conformant.
        sock = tdp / "conf.sock"
        uri = f"unix://{sock}"
        proc = subprocess.run(
            [str(drive), str(miner), uri],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            cwd=str(REPO),
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            pytest.fail(
                f"quip-solver-drive returned {proc.returncode}\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        # Soft assertions on log noise if any
        assert "handshake" not in out.lower() or "ok" in out.lower() or proc.returncode == 0
