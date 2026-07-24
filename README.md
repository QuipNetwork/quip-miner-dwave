# quip-miner-dwave

D-Wave (Ocean) QPU Ising miner for the [quip.network](https://gitlab.com/quip.network)
v0.3 mining protocol. Ships as a pip-installable Python package with a
`quip-miner-dwave` console entry point.

Unlike the classical miners (`quip-miner-cpu`/`-cuda`/`-metal`), there is no
SA/Gibbs binary split — this backend samples on a D-Wave QPU via Ocean
(`dwave.system.DWaveSampler`), with a classical fallback for tiny problems.

## Install

```sh
pip install "quip-miner-dwave @ git+https://gitlab.com/quip.network/quip-miner-dwave.git"
```

The `quip_proto` SDK dependency is built from source (maturin), so installing
needs **Rust** and **protoc** on the machine until `quip_proto` is published to
PyPI. A D-Wave Leap token is needed to reach real QPU hardware (set
`DWAVE_API_TOKEN`); offline/classical sampling needs no token.

## Running

**Connect to a coordinator** (production):

```sh
quip-dwave-qa --quip-coordinator unix:///run/quip/coord.sock
# or: python -m quip_miner_dwave --quip-coordinator ...
```

**Driver / fixed-input (run in isolation, no chain).** Use the coordinator's
`drive` harness pointed at the `quip-miner-dwave` entry point — `--source random`
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
