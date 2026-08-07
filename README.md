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

```sh
pip install \
  --extra-index-url https://gitlab.com/api/v4/projects/71492472/packages/pypi/simple \
  "quip-miner-dwave @ git+https://gitlab.com/quip.network/quip-miner-dwave.git"
```

The extra index is quip-miner's package registry, which is where the `quip_proto`
SDK lives. That project is public, so the index needs no credentials.

On **macOS/arm64** the registry carries a prebuilt `cp311-abi3` wheel covering
every CPython 3.11 and later. pip unpacks that wheel as it comes, so the machine
needs only Python. On every other platform pip falls back to the sdist and
builds the PyO3 extension, which does need **Rust** and **protoc**. Adding
`--only-binary quip_proto` turns that fallback into a hard failure instead of a
silent compile.

A D-Wave Leap token is needed to reach real QPU hardware (set
`DWAVE_API_TOKEN`); offline/classical sampling needs no token.

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

## Tests

```sh
pip install -e ".[dev]"
pytest quip_miner_dwave/tests -v
```

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
