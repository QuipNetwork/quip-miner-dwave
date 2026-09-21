"""Problem regimes of the regime search: one cell maps a nonce to one model.

A cell has two encodings of the same problem. The QPU encoding uses physical
qubit labels and couplings in energy units. The kernel encoding uses dense
spin numbers from 0. Every energy in a report comes from
:func:`energy_milli` or :func:`energy`, computed from spins and the true
couplings, never from a kernel or a sampler.

Four cells draw their couplings from the protocol stream
(``quip_msa.draw_ising``) and score in integer milli units. The clique cell
draws a portfolio model with real coefficients and scores in float64.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from quip_miner_dwave import lattices
from quip_miner_dwave.embeddings import Placement, embed_clique, resolve_chains
from quip_miner_dwave.msa import MsaUnavailable
from quip_miner_dwave.replay import TopologySpec

CPU_MSA = "cpu-msa"
CPU_SA = "cpu-sa"

PM1_ALPHABET = [-1000, 1000]
# 125 magnitudes in steps of 8 milli, both signs: 250 values, no zero, gcd 8.
LEVELS_125_ALPHABET = [sign * 8 * k for k in range(1, 126) for sign in (-1, 1)]

# The unit bond of a ``cpu-msa`` cell in milli. A z bond of the cubic dimer lattice is two halves of 500.
UNIT_MILLI = {"native-pm1": 1000, "diamond-pm1": 1000, "cubic-dimer-pm1": 500}

CLIQUE_HOLDINGS = 9
CLIQUE_RISK_AVERSION = 5.0
CLIQUE_PENALTY_FACTOR = 0.02

CELL_NAMES = ("native-pm1", "native-125", "diamond-pm1", "cubic-dimer-pm1", "clique-portfolio")


@dataclass(frozen=True)
class Cell:
    """One regime.

    ``draw_edges`` are the bonds that receive a drawn coupling, in draw order
    and in the full numbering of the lattice or topology. ``draw_nodes`` is the
    node count of the draw. A lattice cell with a ``placement`` drops its
    vacancies after the draw, so ``n_spins`` and ``physical_nodes`` describe
    the problem that both arms solve.
    """

    name: str
    kernel_label: str
    alphabet_milli: List[int]
    draw_nodes: int
    draw_edges: np.ndarray
    n_spins: int
    physical_nodes: np.ndarray
    dims: Optional[lattices.Dims] = None
    placement: Optional[Placement] = None


def native_cell(name: str, spec: TopologySpec) -> Cell:
    alphabet = {"native-pm1": PM1_ALPHABET, "native-125": LEVELS_125_ALPHABET}[name]
    label = CPU_MSA if name == "native-pm1" else CPU_SA
    return Cell(name, label, alphabet, len(spec.nodes), spec.dense_edges, len(spec.nodes), spec.nodes)


def _lattice_cell(name: str, dims: lattices.Dims, full_spins: int, draw_edges: np.ndarray,
                  placement: Optional[Placement]) -> Cell:
    if placement is None:
        nodes = np.arange(full_spins, dtype=np.int64)
    else:
        nodes = placement.labels[placement.alive]
    return Cell(name, CPU_MSA, PM1_ALPHABET, full_spins, draw_edges, len(nodes), nodes, dims, placement)


def diamond_cell(dims: lattices.Dims, placement: Optional[Placement] = None) -> Cell:
    """The diamond cell. Without a placement it is the ideal lattice, with spin numbers as labels."""
    sites = lattices.diamond_site_count(dims)
    return _lattice_cell("diamond-pm1", dims, sites, lattices.diamond_edges(dims), placement)


def cubic_dimer_cell(dims: lattices.Dims, placement: Optional[Placement] = None) -> Cell:
    bonds, _ = lattices.cubic_logical_bonds(dims)
    return _lattice_cell("cubic-dimer-pm1", dims, 2 * dims[0] * dims[1] * dims[2], bonds, placement)


def make_nonces(count: int, rng_seed: int) -> List[str]:
    """``count`` nonces of 32 bytes, as hex, from one seeded generator."""
    rng = np.random.default_rng(rng_seed)
    return [rng.bytes(32).hex() for _ in range(count)]


def draw_couplings(cell: Cell, nonce_hex: str) -> np.ndarray:
    """The drawn couplings of this nonce, int64 milli, one for each draw edge.

    The protocol stream draws one ``h`` for each node before the couplings, so
    the node count is part of the draw. ``native-pm1`` with the production
    alphabet is the production model.
    """
    try:
        import quip_msa  # pyright: ignore[reportMissingModuleSource]
    except ImportError as exc:
        raise MsaUnavailable(
            "quip_msa is not installed; build it from quip-miner-cpu/py with `maturin develop --release`"
        ) from exc
    nonce = bytes.fromhex(nonce_hex.removeprefix("0x"))
    _, j = quip_msa.draw_ising(nonce, cell.draw_nodes, len(cell.draw_edges), [0], cell.alphabet_milli)
    return np.rint(np.asarray(j) * 1000).astype(np.int64)


def true_problem(cell: Cell, drawn_milli: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(edges, j_milli)`` of the true problem in kernel numbering.

    The draw covers the full lattice. A placement then drops the edges that
    cannot run, and the spins that remain take dense numbers.
    """
    if cell.name == "cubic-dimer-pm1":
        assert cell.dims is not None
        edges, j_milli, _ = lattices.cubic_dimer_edges(cell.dims, drawn_milli)
    else:
        edges, j_milli = cell.draw_edges, np.asarray(drawn_milli, dtype=np.int64)
    if cell.placement is None:
        return edges, j_milli
    dense = np.cumsum(cell.placement.alive) - 1
    keep = cell.placement.keep
    return dense[edges[keep]], j_milli[keep]


def kernel_problem(cell: Cell, edges: np.ndarray, j_milli: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(h, edges, j)`` as the kernel takes them.

    A ``cpu-msa`` cell becomes unit bonds: an edge with coupling ``J`` appears
    ``|J| / unit`` times with coupling +1.0 or -1.0, where ``unit`` is the unit
    bond of the cell. The cell sets the unit, so every model of a cell has the
    same encoding. The kernel treats a repeated edge as one bond of that
    weight. A ``cpu-sa`` cell passes its couplings as floats in energy units.
    """
    h = np.zeros(cell.n_spins, dtype=np.float64)
    couplings = np.asarray(j_milli, dtype=np.int64)
    if cell.kernel_label == CPU_SA:
        return h, edges, couplings / 1000.0
    unit = UNIT_MILLI[cell.name]
    if (np.abs(couplings) % unit).any():
        raise ValueError(f"{cell.name} has a coupling that is not a multiple of {unit} milli")
    repeats = np.abs(couplings) // unit
    return h, np.repeat(edges, repeats, axis=0), np.repeat(np.sign(couplings).astype(np.float64), repeats)


def qpu_problem(
    edges: np.ndarray, j_milli: np.ndarray, physical_nodes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(nodes, h, edges, j)`` on physical labels. ``physical_nodes[i]`` hosts spin ``i``."""
    nodes = np.asarray(physical_nodes, dtype=np.int64)
    return nodes, np.zeros(len(nodes), dtype=np.float64), nodes[edges], np.asarray(j_milli) / 1000.0


def energy_milli(spins: np.ndarray, edges: np.ndarray, j_milli: np.ndarray) -> np.ndarray:
    """Energy of each read in integer milli units. ``spins`` is ``(reads, n_spins)`` of +1/-1."""
    s = np.asarray(spins, dtype=np.int64)
    return (s[:, edges[:, 0]] * s[:, edges[:, 1]]) @ np.asarray(j_milli, dtype=np.int64)


def observed_kernel_label(kernel: Any, h: np.ndarray, edges: np.ndarray, j: np.ndarray) -> str:
    """Which kernel serves this problem.

    A cold job with couplings that the multi-spin path does not take falls back
    to the scalar kernel without a signal. A seeded job returns an error
    instead, so one seeded call of 1 sweep tells the two apart.
    """
    start = np.ones((1, len(h)), dtype=np.int8)
    try:
        kernel.sample(h, edges, j, num_sweeps=1, num_reads=1, seed=0, initial_spins=start)
    except ValueError:
        return CPU_SA
    return CPU_MSA


def max_bonds_per_spin(n_spins: int, edges: np.ndarray) -> int:
    return int(np.bincount(edges.ravel(), minlength=n_spins).max())


# --------------------------------------------------------------- clique cell


def clique_portfolio(nonce_hex: str, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(sigma, returns)`` of the portfolio of this nonce: the covariance and the expected returns of ``n`` assets.

    The loadings have mixed signs, which is what makes the family hard.
    """
    nonce = bytes.fromhex(nonce_hex.removeprefix("0x"))
    rng = np.random.default_rng(int.from_bytes(nonce[:8], "big"))
    loadings = rng.uniform(-1.6, 1.6, n)
    diagonal = rng.uniform(1e-4, 9e-4, n)
    returns = rng.normal(0.0, 1e-3, n)
    return 0.01 * np.outer(loadings, loadings) + np.diag(diagonal), returns


def clique_model(nonce_hex: str, n: int, k: int = CLIQUE_HOLDINGS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(h, edges, j)`` of the portfolio model of this nonce, normalized to a largest coefficient of 1.

    Choose ``k`` of ``n`` assets at equal weight. The cardinality rule sits
    inside the problem as a penalty, so the energy alone scores a read.
    """
    sigma, returns = clique_portfolio(nonce_hex, n)

    c = CLIQUE_RISK_AVERSION / (2.0 * k * k)
    q = c * sigma
    q[np.diag_indices(n)] = -returns / k + c * np.diagonal(sigma)

    off = q - np.diag(np.diagonal(q))
    a = CLIQUE_PENALTY_FACTOR * (np.abs(np.diagonal(q)).max() + 2.0 * k * np.abs(off).max())
    q = q + a * (np.ones((n, n)) - np.eye(n))
    q[np.diag_indices(n)] += a * (1.0 - 2.0 * k)

    upper = np.triu_indices(n, 1)
    j = q[upper] / 2.0
    h = np.diagonal(q) / 2.0 + (q - np.diag(np.diagonal(q))).sum(axis=1) / 2.0
    scale = max(np.abs(h).max(), np.abs(j).max())
    return h / scale, np.column_stack(upper).astype(np.int64), j / scale


def energy(spins: np.ndarray, h: np.ndarray, edges: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Float energy of each read: ``sum(h s) + sum(J s s)``."""
    s = np.asarray(spins, dtype=np.float64)
    return s @ h + (s[:, edges[:, 0]] * s[:, edges[:, 1]]) @ j


# ------------------------------------------------------- one model, both arms


@dataclass(frozen=True)
class Model:
    """One model in the two encodings, with what it takes to score a read.

    ``score_h`` is None for the cells that score in whole milli units. The
    clique cell scores in float64 and carries its chains.
    """

    kernel_h: np.ndarray
    kernel_edges: np.ndarray
    kernel_j: np.ndarray
    qpu_nodes: np.ndarray
    qpu_h: np.ndarray
    qpu_edges: np.ndarray
    qpu_j: np.ndarray
    score_edges: np.ndarray
    score_j: np.ndarray
    score_h: Optional[np.ndarray] = None
    chains: Optional[List[List[int]]] = None
    chain_strength: float = 0.0


def build_model(cell: Cell, nonce_hex: str) -> Model:
    """The model of a nonce in a cell that draws from the protocol stream."""
    edges, j_milli = true_problem(cell, draw_couplings(cell, nonce_hex))
    kernel_h, kernel_edges, kernel_j = kernel_problem(cell, edges, j_milli)
    qpu_nodes, qpu_h, qpu_edges, qpu_j = qpu_problem(edges, j_milli, cell.physical_nodes)
    return Model(kernel_h, kernel_edges, kernel_j, qpu_nodes, qpu_h, qpu_edges, qpu_j, edges, j_milli)


def build_clique_model(nonce_hex: str, chains: List[List[int]], working_edges: Set[Tuple[int, int]],
                       k: int = CLIQUE_HOLDINGS) -> Model:
    """The clique model of a nonce: the logical problem for the kernel, its chains for the QPU.

    The model has one asset for each chain, so the embedding sets the size of the problem.
    """
    h, edges, j = clique_model(nonce_hex, len(chains), k)
    nodes, phys_h, phys_edges, phys_j, strength = embed_clique(h, edges, j, chains, working_edges)
    return Model(h, edges, j, nodes, phys_h, phys_edges, phys_j, edges, j, h, [list(c) for c in chains], strength)


def score(model: Model, spins: np.ndarray) -> np.ndarray:
    """Energy of each read in kernel numbering. The only source of energies in a report."""
    if model.score_h is None:
        return energy_milli(spins, model.score_edges, model.score_j)
    return energy(spins, model.score_h, model.score_edges, model.score_j)


def coupling_extremes(model: Model) -> Dict[str, float]:
    """The extremes of the QPU encoding that the solver ranges limit, and the energy scale of the couplings.

    ``sum_min`` and ``sum_max`` are the lowest and the highest sum of the
    couplings on one qubit. The solver API scales a problem down when a value
    is outside ``extended_j_range`` or ``per_qubit_coupling_range``.
    ``j_rms`` is the root mean square coupling, and ``j_abs_min`` is the
    smallest coupling size, which the analog precision of the chip limits.
    """
    _, inverse = np.unique(model.qpu_edges, return_inverse=True)
    sums = np.bincount(inverse.reshape(-1), weights=np.repeat(model.qpu_j, 2))
    return {
        "j_min": float(model.qpu_j.min()),
        "j_max": float(model.qpu_j.max()),
        "sum_min": float(sums.min()),
        "sum_max": float(sums.max()),
        "h_abs_max": float(np.abs(model.qpu_h).max()),
        "j_rms": float(np.sqrt(np.mean(model.qpu_j ** 2))),
        "j_abs_min": float(np.abs(model.qpu_j).min()),
    }


def decode_reads(model: Model, spins: np.ndarray, variables: Sequence[int]) -> Tuple[np.ndarray, float]:
    """QPU reads in kernel numbering, and the share of broken chains (0.0 without chains)."""
    if model.chains is not None:
        return resolve_chains(spins, variables, model.chains)
    column = {int(label): i for i, label in enumerate(variables)}
    missing = [int(q) for q in model.qpu_nodes if int(q) not in column]
    if missing:
        raise ValueError(f"the reads lack qubits of the model: {missing[:5]}")
    take = [column[int(q)] for q in model.qpu_nodes]
    return np.ascontiguousarray(np.asarray(spins)[:, take], dtype=np.int8), 0.0
