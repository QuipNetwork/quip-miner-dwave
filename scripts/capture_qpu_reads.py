#!/usr/bin/env python3
"""Submit recorded models to the QPU again and keep every read: step 7 input.

The attempts log holds no spins, so the seed test needs fresh reads. Each
model goes through the miner's own ``OceanSampler``, so the defect reduction,
the anneal schedule, and the reconstruction of clamped qubits are the
production code. One ``.npz`` file per model holds the reads in spec node
order, with energies recomputed on the full model in milli units.

``--dry-run`` prints the model count and the estimated spend and submits
nothing. A real run needs ``--yes``. The spend does not pass through the
miner's usage ledger, so it is printed at the end for the operator to record.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from typing import List

import numpy as np

from quip_miner_dwave import replay
from quip_miner_dwave.ocean import OceanSampler

EST_ACCESS_US = 46_100


def pick(heavy_csv: str, per_set: int, seed: int) -> List[str]:
    by_set: dict = {}
    with open(heavy_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_set.setdefault(row["set"], []).append(row["nonce"])
    rng = random.Random(seed)
    chosen: List[str] = []
    for name in sorted(by_set):
        nonces = sorted(by_set[name])
        chosen += rng.sample(nonces, min(per_set, len(nonces)))
    return chosen


def energies_milli(spins: np.ndarray, dense_edges: np.ndarray, h: np.ndarray, j: np.ndarray) -> np.ndarray:
    s = spins.astype(np.int64)
    field = s @ np.rint(h * 1000).astype(np.int64)
    coupling = (s[:, dense_edges[:, 0]] * s[:, dense_edges[:, 1]]) @ np.rint(j * 1000).astype(np.int64)
    return field + coupling


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--spec", required=True)
    parser.add_argument("--heavy", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--per-set", type=int, default=250)
    parser.add_argument("--reads", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1387)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="confirm the QPU spend")
    args = parser.parse_args()

    spec = replay.load_spec(args.spec)
    os.makedirs(args.out_dir, exist_ok=True)
    nonces = [n for n in pick(args.heavy, args.per_set, args.seed)
              if not os.path.exists(f"{args.out_dir}/{n}.npz")]
    print(f"{len(nonces)} models to capture, about {len(nonces) * EST_ACCESS_US / 1e6:.1f} s of QPU access")
    if args.dry_run:
        return 0
    if not args.yes:
        print("pass --yes to spend it")
        return 2

    sampler = OceanSampler(mock=False)
    spent_us = 0
    try:
        for count, nonce in enumerate(nonces, 1):
            h, j = replay.model_from_nonce(spec, nonce)
            result = sampler.sample(
                spec.nodes, h, spec.edges, j,
                num_reads=args.reads, nonce_seed=bytes.fromhex(nonce), label="quip-seed-capture",
            )
            spins = replay.spec_order(result.spins, result.variables, spec)
            np.savez_compressed(
                f"{args.out_dir}/{nonce}.npz",
                spins=spins,
                energies_milli=energies_milli(spins, spec.dense_edges, h, j),
                access_us=np.int64(result.device_access_time_us),
            )
            spent_us += int(result.device_access_time_us)
            if count % 50 == 0:
                print(f"{count} of {len(nonces)}  spent {spent_us / 1e6:.1f} s")
    finally:
        close = getattr(sampler, "close", None)
        if close:
            close()
    print(f"captured {len(nonces)} models, QPU access spent {spent_us / 1e6:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
