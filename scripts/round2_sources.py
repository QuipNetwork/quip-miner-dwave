#!/usr/bin/env python3
"""Round 2 source inventory and historical-provenance check.

Task 1 of docs/superpowers/plans/2026-09-22-regime-search-round2.md: before
any later task touches a sampler, this module answers one question about
every candidate data source: which of the twelve fields a historical replica
needs are actually on file, versus assumed from a current default.

``historical_gaps`` is the pure check the plan specifies verbatim. The rest
of the module inventories files on disk (path, size, hash, mtime) without
opening anything that looks like a credential, and assembles that inventory
plus known fields into one record per source with an explicit provenance tag
(``historical``, ``prospective``, or ``synthetic-test``).

Running this file directly regenerates round2/sources.json and
round2/environment.json under the data root (default
``/home/carback1/quip-data/regimes/round2``) from the live state of the D,
C, and P checkouts and the local market-archive inventory. Nothing here
substitutes current data for a historical value that record-keeping never
captured; a missing field stays missing.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROVENANCE_KINDS = ("historical", "prospective", "synthetic-test")

REQUIRED_FIELDS = (
    "market_snapshot_sha256",
    "market_as_of",
    "basket",
    "sliders",
    "deployment_commit",
    "runtime_config",
    "effective_beta",
    "reads",
    "sa_sweeps",
    "qpu_anneal_us",
    "qpu_solver",
    "tie_rules",
)

# Substrings checked case-insensitively against a path's components. A file
# whose name or containing directory matches one of these is never opened or
# hashed by inventory_paths.
CREDENTIAL_MARKERS = ("secret", "token", "keyring", "pass", ".env")


def historical_gaps(record: dict) -> list:
    """Required historical fields a record does not have on file.

    Verbatim from the Task 1 brief: a key counts as missing when it is
    absent or explicitly ``None``. A record built from current defaults
    (for example just a ``runtime_config``) must not read as complete.
    """
    required = (
        "market_snapshot_sha256", "market_as_of", "basket", "sliders",
        "deployment_commit", "runtime_config", "effective_beta",
        "reads", "sa_sweeps", "qpu_anneal_us", "qpu_solver", "tie_rules",
    )
    return [key for key in required if key not in record or record[key] is None]


def is_credential_like(path: Path) -> bool:
    """True if any path component looks like a credential file or store."""
    lowered = [part.lower() for part in path.parts]
    for part in lowered:
        for marker in CREDENTIAL_MARKERS:
            if marker in part:
                return True
    return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso_mtime(path: Path) -> str:
    stamp = path.stat().st_mtime
    return datetime.datetime.fromtimestamp(stamp, tz=datetime.timezone.utc).isoformat()


def inventory_paths(root: Path) -> List[Dict[str, Any]]:
    """Path, size, sha256, and mtime of every non-credential file under root.

    Returns an empty list when root does not exist. Never opens a path that
    ``is_credential_like`` flags; its presence is not even recorded, so a
    credential file's name cannot leak into round2/sources.json.
    """
    root = Path(root)
    if not root.exists():
        return []
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if is_credential_like(path):
            continue
        entries.append(
            {
                "path": path,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "mtime": _iso_mtime(path),
            }
        )
    return entries


def build_source_record(
    *,
    name: str,
    provenance: str,
    root: Path,
    known_fields: Dict[str, Any],
) -> Dict[str, Any]:
    """One round2/sources.json entry: known fields, file inventory, gaps.

    ``known_fields`` may supply any subset of REQUIRED_FIELDS plus free-form
    extras (for example a note on why a field is unrecoverable); fields the
    caller does not supply are recorded as explicit ``None`` rather than
    omitted, so historical_gaps sees them and so does anyone reading the
    JSON directly.
    """
    if provenance not in PROVENANCE_KINDS:
        raise ValueError(f"unknown provenance {provenance!r}; expected one of {PROVENANCE_KINDS}")
    record: Dict[str, Any] = {"name": name, "provenance": provenance}
    for field in REQUIRED_FIELDS:
        record[field] = known_fields.get(field)
    for key, value in known_fields.items():
        if key not in REQUIRED_FIELDS:
            record[key] = value
    record["inventory"] = inventory_paths(root)
    record["gaps"] = historical_gaps(record)
    return record


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {value!r}")


# --------------------------------------------------------------- environment
def _run(cmd: List[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _git_facts(repo: Path) -> Dict[str, Any]:
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    status = _run(["git", "status", "--porcelain"], cwd=repo)
    branch = _run(["git", "branch", "--show-current"], cwd=repo)
    return {
        "path": str(repo),
        "head": head,
        "branch": branch,
        "dirty": bool(status),
        "dirty_files": status.splitlines() if status else [],
    }


def _python_facts(python: Path, imports: List[str]) -> Dict[str, Any]:
    version = _run([str(python), "--version"])
    modules: Dict[str, Optional[str]] = {}
    for module in imports:
        code = f"import {module}; print({module}.__file__)"
        out = _run([str(python), "-c", code])
        modules[module] = out if out else None
    return {"interpreter": str(python), "version": version, "modules": modules}


def collect_environment(
    *,
    d_repo: Path,
    c_repo: Path,
    p_repo: Path,
    plan_d_commit: str,
    plan_c_commit: str,
    plan_p_commit: str,
) -> Dict[str, Any]:
    """Environment facts for D, C, and P: HEAD, dirty state, interpreters, imports, host."""
    d_git = _git_facts(d_repo)
    c_git = _git_facts(c_repo)
    p_git = _git_facts(p_repo)
    d_git["plan_commit"] = plan_d_commit
    c_git["plan_commit"] = plan_c_commit
    p_git["plan_commit"] = plan_p_commit

    d_python = _python_facts(d_repo / ".venv" / "bin" / "python", ["quip_miner_dwave", "quip_msa"])
    p_python = _python_facts(
        p_repo / ".venv" / "bin" / "python",
        ["qpo", "qpo.encoders", "qpo.solvers.sampling", "qpo.solvers.feasibility", "neal", "scipy"],
    )
    d_lock = d_repo / "uv.lock"
    p_lock = p_repo / "uv.lock"

    return {
        "generated_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "host": {
            "cpu_model": _cpu_model(),
            "platform": platform.platform(),
        },
        "compilers": {
            "rustc": _run(["rustc", "--version"]),
            "cargo": _run(["cargo", "--version"]),
        },
        "repositories": {
            "D": {**d_git, "lockfile_sha256": sha256_file(d_lock) if d_lock.exists() else None},
            "C": c_git,
            "P": {**p_git, "lockfile_sha256": sha256_file(p_lock) if p_lock.exists() else None},
        },
        "environments": {
            "D": d_python,
            "P": p_python,
        },
    }


def _cpu_model() -> Optional[str]:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


# --------------------------------------------------------------------- CLI
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/carback1/quip-data/regimes/round2"),
        help="Directory holding sources.json, environment.json, and reference/.",
    )
    parser.add_argument("--d-repo", type=Path, default=Path("/home/carback1/quip-miner-dwave"))
    parser.add_argument("--c-repo", type=Path, default=Path("/home/carback1/quip-miner-cpu"))
    parser.add_argument(
        "--p-repo",
        type=Path,
        default=Path("/home/carback1/quip-data/regimes/round2/reference/qportfolio"),
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("/home/carback1/quip-data"),
        help="Root to search for market archives and reference configuration.",
    )
    args = parser.parse_args(argv)

    sources = {
        "generated_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "records": [
            build_source_record(
                name="portfolio-gutcheck",
                provenance="synthetic-test",
                root=args.archive_root / "regimes" / "portfolio-gutcheck",
                known_fields={
                    "reads": 64,
                    "sa_sweeps": 500,
                    "note": (
                        "Classical-only probe (2026-09-20) against qpo.build_problem's "
                        "shipped default basket, not a captured historical market "
                        "snapshot. No QPU fields recorded; the script never ran on hardware."
                    ),
                },
            ),
            build_source_record(
                name="round1-regime-manifests",
                provenance="historical",
                root=args.archive_root / "regimes" / "round1",
                known_fields={
                    "reads": 64,
                    "qpu_anneal_us": [80, 400],
                    "qpu_solver": "Advantage2_system1",
                    "note": (
                        "QPU regime captures for logical-topology benchmarks "
                        "(native-125, clique-portfolio, native-pm1, diamond-pm1, "
                        "cubic-dimer-pm1), not the portfolio problem instances. No "
                        "market snapshot, basket, sliders, deployment commit, "
                        "effective beta, sa_sweeps, or tie_rules recorded per manifest."
                    ),
                },
            ),
            build_source_record(
                name="analysis-tables-2026-09-21",
                provenance="historical",
                root=args.archive_root / "regimes" / "analysis-tables-2026-09-21",
                known_fields={
                    "note": (
                        "Notion-exported analysis and figures for the round1 capture "
                        "above; text and SVG/PNG outputs, not raw market or run data."
                    ),
                },
            ),
            build_source_record(
                name="qportfolio-reference-checkout",
                provenance="prospective",
                root=args.p_repo,
                known_fields={
                    "deployment_commit": "9136be4c4b57abdab7f0c46120c341ba94fae014",
                    "note": (
                        "Detached checkout of qtbench/Q-Portfolio-Optimization-Benchmark "
                        "at the plan-pinned commit; reference code, not a data capture. "
                        "Inventory below excludes .git and .venv (large, not provenance-bearing)."
                    ),
                },
            ),
        ],
    }
    # The P checkout's .git and .venv trees are large and not provenance-bearing;
    # drop them from the last record's inventory rather than walking them.
    sources["records"][-1]["inventory"] = [
        entry
        for entry in sources["records"][-1]["inventory"]
        if ".git" not in entry["path"].parts and ".venv" not in entry["path"].parts
    ]

    environment = collect_environment(
        d_repo=args.d_repo,
        c_repo=args.c_repo,
        p_repo=args.p_repo,
        plan_d_commit="c6831057daeb720ed16175e5d8ce04f39520a904",
        plan_c_commit="281c697d0785b5a14df199263b4a9af291325dc6",
        plan_p_commit="9136be4c4b57abdab7f0c46120c341ba94fae014",
    )

    args.data_root.mkdir(parents=True, exist_ok=True)
    (args.data_root / "sources.json").write_text(
        json.dumps(sources, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    (args.data_root / "environment.json").write_text(
        json.dumps(environment, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
