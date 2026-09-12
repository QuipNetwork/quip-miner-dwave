# quip-miner-dwave

D-Wave (Ocean) QPU Ising miner for the [quip.network](https://gitlab.com/quip.network)
v0.3 mining protocol. Ships as a pip-installable Python package with a
`quip-dwave-qa` console entry point.

Unlike the classical miners (`quip-miner-cpu`/`-cuda`/`-metal`), there is no
SA/Gibbs binary split: this backend samples on a D-Wave QPU via Ocean
(`dwave.system.DWaveSampler`), and every job goes to the QPU.

For tests and offline runs, `QUIP_DWAVE_MOCK=1` or `--mock` swaps in a dimod
sampler. The default is `ExactSolver`, which enumerates every state and so
handles only the smallest problems. Set `QUIP_DWAVE_MOCK_BACKEND=sa` for
`SimulatedAnnealingSampler`, which scales to realistic topology sizes. You
select this path explicitly; the miner never switches to it on its own.

## Install

### Released executable (macOS, Apple Silicon)

Each release publishes `quip-dwave-qa-darwin-arm64`, a self-contained
executable that bundles Python, the Ocean SDK, and the compiled
`quip_solver_core` extension. Drop it in and run it. The host needs only the
file.

```sh
TAG=v0.3.0-rc3   # or a later release
BASE=https://gitlab.com/api/v4/projects/84792347/packages/generic/quip-miner-dwave
curl -fLO "$BASE/$TAG/quip-dwave-qa-darwin-arm64"
chmod +x quip-dwave-qa-darwin-arm64
mv quip-dwave-qa-darwin-arm64 /usr/local/bin/quip-dwave-qa
```

Rename it, because the coordinator spawns whatever `[dwave] binary` names and
its default is `quip-dwave-qa`. The CPU, CUDA, and Metal miners install the
same way.

### From source

```sh
pip install "quip-miner-dwave @ git+https://gitlab.com/quip.network/quip-miner-dwave.git"
```

The `quip_solver_core` SDK dependency installs from a published PyPI wheel on
Linux (amd64 and aarch64) and macOS (Apple Silicon). Elsewhere pip falls back
to building it from source through maturin, which needs **Rust** on the
machine.

Reaching real QPU hardware takes a D-Wave Leap token, set through
`DWAVE_API_TOKEN`. `DWAVE_API_KEY`, the name the v0.2 stack used, still works:
the miner copies it to `DWAVE_API_TOKEN` at startup and logs a deprecation
warning. Offline and classical sampling take no token.

### Building the executable yourself

```sh
pip install ".[dev]" pyinstaller
pyinstaller --clean --noconfirm pyinstaller/quip-dwave-qa.spec
QUIP_DWAVE_MOCK=1 ./dist/quip-dwave-qa --check
```

`pyinstaller/quip-dwave-qa.spec` names each collected package and gives the
reason. Ocean keeps compiled modules inside PEP 420 namespace packages that
PyInstaller does not find on its own, and PyInstaller reports a successful
build for a binary that cannot import `dimod`. Run the result before trusting
it.

## Running

**Connect to a coordinator** (production):

```sh
quip-dwave-qa --quip-coordinator unix:///run/quip/coord.sock
# or: python -m quip_miner_dwave --quip-coordinator ...
```

**Driver / fixed-input (run in isolation, no chain).** Use the coordinator's
`drive` harness pointed at the `quip-dwave-qa` entry point — `--source random`
for golden-drawn problems, `--source list <jsonl>` for a fixed replay.

**Introspection:**

```sh
quip-dwave-qa --capabilities
quip-dwave-qa --check
```

## Pipeline depth

The miner keeps up to `queue_depth` submissions on the QPU at once. A
cloud-attached QPU spends most of each job's wall time on the round trip, so a
shallow pipeline leaves the device idle between jobs.

The default is 96. It comes from Little's Law:

    depth = chip throughput * round trip

Measured on `Advantage2_system1` with a production-sized problem (4577 nodes,
41514 couplers, `num_reads=48`): 43.2 ms of access time per job gives a chip
ceiling of 23.2 jobs/s, and an uncontended round trip is 1.57 s. That needs 36
in flight. A round trip of 3.05 s, which session logs show under load, needs
71. The default covers a round trip of 4.14 s, so a connectivity blip reduces
throughput instead of stalling the QPU.

Set `queue_depth` in the coordinator's `[dwave]` backend config to override
it:

```toml
queue_depth = 64
```

The operator value wins. If unset, the miner uses the depth the coordinator
sent. If that is also unset, the miner uses 96.

Depth does not raise jobs per hour under a budget. The budget paces spend
across the quota period, and it funds fewer jobs per second than even a
shallow pipeline delivers. Depth sets the burst rate and the duty cycle.

Depth does set how much work a reseed can strand. When the coordinator cancels
a generation, the miner asks D-Wave to drop every job of that generation still
on the QPU. D-Wave refunds only the jobs it has not started annealing, so the
quota pays for the rest. Each session log reports the measured hit rate:

```
[QPU] cancel gen<=339: asked D-Wave to drop 3 in-flight job(s), 3 still
running | session: 12 cancelled, 4 annealed anyway (33% missed)
```

The same config holds other `[dwave]` keys: `budget`, `budget_reset_day`,
`usage_db`, `num_reads`, and `anneal_time_us`. This README does not document
them yet.

## History and the profile report

The miner keeps three tables in the usage database, beside the budget
ledger. One tracks throughput per UTC hour, covering job completions, QPU
busy time, round trips, and the D-Wave service time from SAPI's
`submitted_on` and `solved_on`. Another holds one row per qblock round. The
last holds a histogram of each job's best energy relative to the round's
target. The round strategy reads them. Nothing about mining depends on
them. A database that fails to open disables the history and logs one
warning.

At start the miner seeds past rounds from the coordinator's attempts files,
`<data_dir>/<qblock_id>/attempts.jsonl`. The default location is the
`attempts` directory beside the usage database, which is where the node
manager renders it. `--attempts-dir PATH` overrides it.

`quip-dwave-qa --profile [--usage-db PATH]` prints the hour-of-week grid of
jobs per second, the grid of D-Wave queue wait, the win summary for
weekdays and weekends, and the last 20 rounds with the strategy's predicted
win probability beside what happened. It needs no QPU and no token.

## Tests

```sh
pip install -e ".[dev]"
pytest quip_miner_dwave/tests -v
```

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
