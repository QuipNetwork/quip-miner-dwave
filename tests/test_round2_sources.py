"""The provenance check, credential filter, and file inventory of round2_sources.py.

The provenance check runs before any sampler integration, per Task 1 of
docs/superpowers/plans/2026-09-22-regime-search-round2.md: current defaults
never stand in for a historical record, and a missing field is explicit
rather than silently substituted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import pathlib

import pytest

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "round2_sources.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("round2_sources", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


round2_sources = _load_module()

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


def _complete_record() -> dict:
    return {
        "market_snapshot_sha256": "a" * 64,
        "market_as_of": "2026-09-20",
        "basket": ["A0", "A1"],
        "sliders": {"hold_count": 5},
        "deployment_commit": "9136be4c4b57abdab7f0c46120c341ba94fae014",
        "runtime_config": {"beta": 0.4},
        "effective_beta": 0.4,
        "reads": 64,
        "sa_sweeps": 500,
        "qpu_anneal_us": [80, 400],
        "qpu_solver": "Advantage2_system1",
        "tie_rules": "strict",
    }


# ------------------------------------------------------------- historical_gaps
def test_current_defaults_do_not_prove_historical_configuration():
    gaps = round2_sources.historical_gaps({"runtime_config": {"beta": 0.4}})
    assert "market_snapshot_sha256" in gaps
    assert "effective_beta" in gaps
    assert "qpu_anneal_us" in gaps


def test_historical_gaps_empty_when_every_field_present():
    assert round2_sources.historical_gaps(_complete_record()) == []


def test_historical_gaps_treats_none_as_missing():
    record = _complete_record()
    record["market_as_of"] = None
    gaps = round2_sources.historical_gaps(record)
    assert gaps == ["market_as_of"]


def test_historical_gaps_reports_every_required_key_on_empty_record():
    assert round2_sources.historical_gaps({}) == list(REQUIRED_FIELDS)


# ------------------------------------------------------------- credential filter
@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".env.local",
        "api_secret.json",
        "SECRETS.yaml",
        "access_token.txt",
        "oauth-token",
        "keyring.db",
        ".pass-store/entry.gpg",
    ],
)
def test_is_credential_like_flags_secret_shaped_names(name):
    assert round2_sources.is_credential_like(pathlib.Path("/x/y") / name)


@pytest.mark.parametrize("name", ["manifest.json", "ladder.csv", "run-info.json", "qpu.csv"])
def test_is_credential_like_leaves_ordinary_files_alone(name):
    assert not round2_sources.is_credential_like(pathlib.Path("/x/y") / name)


# ------------------------------------------------------------- hashing / inventory
def test_sha256_file_matches_hashlib(tmp_path):
    target = tmp_path / "data.bin"
    target.write_bytes(b"round 2 provenance fixture")
    expected = hashlib.sha256(b"round 2 provenance fixture").hexdigest()
    assert round2_sources.sha256_file(target) == expected


def test_inventory_paths_records_size_hash_and_mtime_and_skips_credentials(tmp_path):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "ladder.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    entries = round2_sources.inventory_paths(tmp_path)
    names = {entry["path"].name for entry in entries}

    assert names == {"manifest.json", "ladder.csv"}
    for entry in entries:
        assert entry["size_bytes"] == entry["path"].stat().st_size
        assert entry["sha256"] == round2_sources.sha256_file(entry["path"])
        assert entry["mtime"].endswith("+00:00") or entry["mtime"].endswith("Z")


def test_inventory_paths_on_missing_root_returns_empty(tmp_path):
    assert round2_sources.inventory_paths(tmp_path / "does-not-exist") == []


# ------------------------------------------------------------- source record assembly
def test_build_source_record_keeps_known_fields_and_computes_gaps(tmp_path):
    (tmp_path / "qpo_real.json").write_text("[]", encoding="utf-8")

    record = round2_sources.build_source_record(
        name="portfolio-gutcheck",
        provenance="synthetic-test",
        root=tmp_path,
        known_fields={"reads": 64, "sa_sweeps": 500},
    )

    assert record["name"] == "portfolio-gutcheck"
    assert record["provenance"] == "synthetic-test"
    assert record["reads"] == 64
    assert record["sa_sweeps"] == 500
    assert record["market_snapshot_sha256"] is None
    assert "market_snapshot_sha256" in record["gaps"]
    assert "reads" not in record["gaps"]
    assert len(record["inventory"]) == 1
    assert record["inventory"][0]["path"].name == "qpo_real.json"


def test_build_source_record_rejects_unknown_provenance(tmp_path):
    with pytest.raises(ValueError):
        round2_sources.build_source_record(
            name="x", provenance="current", root=tmp_path, known_fields={}
        )
