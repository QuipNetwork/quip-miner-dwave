import csv
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "seeded_sweep.py"


def _load_module():
    pytest.importorskip("quip_msa")
    spec = importlib.util.spec_from_file_location("seeded_sweep", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seeded_sweep = _load_module()

ROW = [
    "nonce", "arm", "start_fraction", "sweeps", "best_milli",
    "qpu_lanes", "qpu_lane_best_milli", "fill_lane_best_milli", "wall_ms",
    "fill_wall_ms",
]


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(seeded_sweep.HEADER)
        for row in rows:
            writer.writerow(row)


def test_done_rows_drops_a_malformed_last_line(tmp_path):
    path = tmp_path / "seeded.csv"
    _write_csv(path, [["n1", "cold", "", 1024, -100, 0, "", "", "5", "0"]])
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("n2,cold,,")  # truncated tail, no trailing newline
    done = seeded_sweep.done_rows(str(path))
    assert done == {("n1", "cold", "", 1024)}


def test_ensure_trailing_newline_appends_one_when_missing(tmp_path):
    path = tmp_path / "seeded.csv"
    path.write_bytes(b"n1,cold,,1024,-100,0,,,5,0")
    seeded_sweep.ensure_trailing_newline(str(path))
    assert path.read_bytes().endswith(b"\n")


def test_ensure_trailing_newline_is_a_no_op_when_already_present(tmp_path):
    path = tmp_path / "seeded.csv"
    path.write_bytes(b"n1,cold,,1024,-100,0,,,5,0\n")
    seeded_sweep.ensure_trailing_newline(str(path))
    assert path.read_bytes() == b"n1,cold,,1024,-100,0,,,5,0\n"


def test_summarise_skips_a_missing_cell_instead_of_raising(tmp_path, capsys):
    out = tmp_path / "seeded.csv"
    reads_dir = tmp_path / "reads"
    reads_dir.mkdir()
    # n1 has both cold rows and a seeded row; n2 has only a cold row (its
    # seeded row is missing, e.g. an interrupted run).
    _write_csv(out, [
        ["n1", "cold", "", 1024, -100, 0, "", "", "5", "0"],
        ["n1", "seeded", "0.50", 1024, -110, 4, -110, -90, "6", "20"],
        ["n2", "cold", "", 1024, -80, 0, "", "", "5", "0"],
    ])
    for nonce, best in (("n1", -110), ("n2", -80)):
        np.savez(reads_dir / f"{nonce}.npz", energies_milli=np.array([best]))

    args = seeded_sweep.argparse.Namespace(out=str(out), reads_dir=str(reads_dir), ceiling_fraction="0.50")
    rc = seeded_sweep.summarise(args)
    text = capsys.readouterr().out

    assert rc == 0
    assert "skipped" in text
    assert text.count("skipped") == 1


def test_ceiling_fraction_is_normalised_like_the_stored_key(tmp_path, capsys, monkeypatch):
    out = tmp_path / "seeded.csv"
    reads_dir = tmp_path / "reads"
    reads_dir.mkdir()
    _write_csv(out, [
        ["n1", "cold", "", 1024, -100, 0, "", "", "5", "0"],
        ["n1", "seeded", "0.60", 1024, -110, 4, -110, -90, "6", "20"],
    ])
    np.savez(reads_dir / "n1.npz", energies_milli=np.array([-110]))

    argv = [
        "seeded_sweep.py", "summarise",
        "--reads-dir", str(reads_dir), "--out", str(out),
        "--ceiling-fraction", "0.6",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = seeded_sweep.main()
    text = capsys.readouterr().out

    assert rc == 0
    assert "skipped" not in text
