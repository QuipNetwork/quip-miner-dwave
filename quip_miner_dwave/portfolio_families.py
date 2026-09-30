"""Random portfolio models on a rigid topology, with and without real market structure.

Every model of basket size N shares one graph, the complete graph on N
selection variables, and differs only in its weights. The weights come from
hourly returns, and the returns come from one of two families:

- Family A resamples real returns. :func:`sample_window` takes a random
  contiguous window and :func:`block_bootstrap` resamples it in whole-row
  blocks, so co-movement, subgroups, heavy tails, volatility clustering, and
  closed-market hours all survive.
- Family C removes the structure but keeps the marginals.
  :func:`null_iid_normal` and :func:`null_shuffle` act on the returns and keep
  every asset's missing hours. :func:`null_rotate_correlation` acts on the
  covariance and keeps each asset's variance and roughly the eigenvalue
  spectrum, but not which assets group together.

The diagnostics measure the structure instead of assuming it:
:func:`significant_pairs`, :func:`best_modularity`,
:func:`near_degenerate_count`, and :func:`second_best_gap`. They use each
pair's own overlap, because stock pairs share far fewer hours than crypto pairs.

Only numpy and scipy are needed, so the production backend's environment can
import this module.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

#: Largest exact enumeration :func:`near_degenerate_count` attempts.
ENUMERATION_LIMIT = 10_000_000
_CHUNK = 100_000
#: Fisher z above which a pair's correlation counts as significant.
SIGNIFICANCE_Z = 3.0


def sample_window(returns: np.ndarray, rng: np.random.Generator, rows: int) -> tuple[int, np.ndarray]:
    """``(start, window)``: a random contiguous window of ``rows`` return rows and its first row."""
    if rows > returns.shape[0]:
        raise ValueError(f"a {rows}-row window does not fit a {returns.shape[0]}-row history")
    start = int(rng.integers(0, returns.shape[0] - rows + 1))
    return start, returns[start : start + rows]


def block_bootstrap(returns: np.ndarray, rng: np.random.Generator, block: int) -> np.ndarray:
    """Circular block bootstrap of whole rows, the same length as ``returns``.

    Each row keeps every asset's value for that hour, so cross-asset structure
    and the missing-hour pattern move together.
    """
    rows = returns.shape[0]
    starts = rng.integers(0, rows, size=math.ceil(rows / block))
    index = (starts[:, None] + np.arange(block)[None, :]).ravel()[:rows] % rows
    return returns[index]


def null_iid_normal(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independent normal draws with each asset's observed mean and volatility."""
    result = np.full_like(returns, np.nan)
    for column in range(returns.shape[1]):
        observed = ~np.isnan(returns[:, column])
        values = returns[observed, column]
        result[observed, column] = rng.normal(values.mean(), values.std(), observed.sum())
    return result


def null_shuffle(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Each asset's observed values permuted independently among its observed hours."""
    result = returns.copy()
    for column in range(returns.shape[1]):
        observed = np.flatnonzero(~np.isnan(returns[:, column]))
        result[observed, column] = returns[rng.permutation(observed), column]
    return result


def correlation(sigma: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.diagonal(sigma))
    return sigma / np.outer(scale, scale)


def null_rotate_correlation(sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A covariance with the same variances and a Haar-rotated correlation spectrum.

    Rescaling the rotated matrix back to a unit diagonal moves the eigenvalues
    slightly, so the spectrum is kept approximately, not exactly.
    """
    values, _ = np.linalg.eigh(correlation(sigma))
    gaussian = rng.standard_normal(sigma.shape)
    q, r = np.linalg.qr(gaussian)
    q = q * np.sign(np.diagonal(r))
    rotated = correlation((q * values) @ q.T)
    scale = np.sqrt(np.diagonal(sigma))
    return rotated * np.outer(scale, scale)


def pair_overlaps(returns: np.ndarray) -> np.ndarray:
    """Hours in which both assets of each pair have a return; the diagonal is each asset's count."""
    observed = (~np.isnan(returns)).astype(np.int64)
    return observed.T @ observed


def significant_pairs(corr: np.ndarray, overlaps: np.ndarray, z: float = SIGNIFICANCE_Z) -> np.ndarray:
    """Pairs whose correlation is significant over that pair's own overlap (Fisher z).

    The backend estimates each correlation over the hours both assets traded,
    so a stock pair rests on far fewer hours than a crypto pair and needs a
    larger correlation to count. The diagonal is never significant.
    """
    clipped = np.clip(corr, -0.999999, 0.999999)
    score = np.abs(np.arctanh(clipped)) * np.sqrt(np.clip(overlaps - 3, 0, None))
    mask = score > z
    np.fill_diagonal(mask, False)
    return mask


def best_modularity(corr: np.ndarray, significant: np.ndarray) -> float:
    """Largest Newman modularity over the cuts of an average-linkage clustering.

    The graph weight is the positive correlation of each significant pair. A
    matrix with no significant positive pair has modularity 0.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    weights = np.where(significant & (corr > 0.0), corr, 0.0)
    total = weights.sum()
    n = corr.shape[0]
    if total <= 0.0 or n < 2:
        return 0.0
    degree = weights.sum(axis=1)
    distance = np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="average")
    best = 0.0
    for groups in range(2, n + 1):
        labels = fcluster(tree, groups, criterion="maxclust")
        same = labels[:, None] == labels[None, :]
        best = max(best, float(((weights - np.outer(degree, degree) / total) * same).sum() / total))
    return best


def _selection_values(q: np.ndarray, k: int) -> np.ndarray | None:
    """QUBO value of every K-selection, or None beyond :data:`ENUMERATION_LIMIT`."""
    n = q.shape[0]
    if math.comb(n, k) > ENUMERATION_LIMIT:
        return None
    symmetric = (q + q.T) / 2.0
    values = []
    combinations = itertools.combinations(range(n), k)
    while True:
        chunk = np.fromiter(itertools.chain.from_iterable(itertools.islice(combinations, _CHUNK)), dtype=np.int64)
        if chunk.size == 0:
            break
        subsets = chunk.reshape(-1, k)
        values.append(symmetric[subsets[:, :, None], subsets[:, None, :]].sum(axis=(1, 2)))
    return np.concatenate(values)


def near_degenerate_count(q: np.ndarray, k: int, tolerance: float) -> int | None:
    """Exact count of K-selections whose QUBO value lies within ``tolerance`` of the best.

    ``tolerance`` is relative to the best value's magnitude. Returns None when
    the enumeration exceeds :data:`ENUMERATION_LIMIT`.
    """
    values = _selection_values(q, k)
    if values is None:
        return None
    best = values.min()
    return int((np.abs(values - best) <= tolerance * abs(best)).sum())


def second_best_gap(q: np.ndarray, k: int) -> float | None:
    """Relative gap between the best and second-best K-selection, or None beyond the limit."""
    values = _selection_values(q, k)
    if values is None or values.size < 2:
        return None
    first, second = np.partition(values, 1)[:2]
    return float((second - first) / abs(first))
