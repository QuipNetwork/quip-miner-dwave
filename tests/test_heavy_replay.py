import csv
import importlib.util
import pathlib

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "heavy_replay.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("heavy_replay", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


heavy_replay = _load_module()


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(heavy_replay.HEADER)
        for row in rows:
            writer.writerow(row)


def test_read_energies_ignores_sentinel_rows(tmp_path):
    path = tmp_path / "heavy.csv"
    _write_csv(
        path,
        [
            ["n1", "low-energy", -14600000, 128, 131072, 10.0],
            ["n2", "low-energy", heavy_replay.NO_SOLUTION, 128, 131072, 10.0],
            ["n3", "false-positive", -14000000, 128, 131072, 10.0],
        ],
    )
    by_set, ignored = heavy_replay.read_energies(str(path))
    assert ignored == 1
    assert by_set["low-energy"] == [-14600000]
    assert by_set["false-positive"] == [-14000000]


def test_read_energies_with_no_sentinel_rows(tmp_path):
    path = tmp_path / "heavy.csv"
    _write_csv(
        path,
        [
            ["n1", "low-energy", -14600000, 128, 131072, 10.0],
            ["n3", "false-positive", -14000000, 128, 131072, 10.0],
        ],
    )
    by_set, ignored = heavy_replay.read_energies(str(path))
    assert ignored == 0
    assert by_set == {"low-energy": [-14600000], "false-positive": [-14000000]}


def test_compare_with_an_empty_set_prints_n_zero_and_skips_gap(tmp_path, capsys):
    path = tmp_path / "heavy.csv"
    _write_csv(path, [["n1", "low-energy", -14600000, 128, 131072, 10.0]])
    args = heavy_replay.argparse.Namespace(out=str(path))
    rc = heavy_replay.compare(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "false-positive" in out
    assert "skipping mean gap and KS test" in out
