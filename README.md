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
executable that bundles Python, the Ocean SDK, and the compiled `quip_proto`
extension. Drop it in and run it. The host needs only the file.

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

The `quip_proto` SDK dependency builds from source through maturin, so this path
needs **Rust** and **protoc** on the machine.

Reaching real QPU hardware takes a D-Wave Leap token, set through
`DWAVE_API_TOKEN`. Offline and classical sampling take no token.

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

## Tests

```sh
pip install -e ".[dev]"
pytest quip_miner_dwave/tests -v
```

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
