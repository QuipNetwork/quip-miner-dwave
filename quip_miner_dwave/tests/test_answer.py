"""Read an answer without building a dimod.SampleSet.

OceanSampler needs five things off a finished job: spins, variable labels,
energies, occurrence counts and timing. A cloud Future carries all five. Going
through .sampleset instead costs 28 ms per job at production size, because it
turns the decoded numpy arrays into Python lists, walks them with a nested
comprehension, and hands them to dimod to turn back into numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

from quip_miner_dwave.answer import AnswerView, answer_view


class FakeFuture:
    """The five attributes answer_view reads off a cloud Future."""

    def __init__(self, samples, variables, energies, num_occurrences, timing):
        self.samples = samples
        self.variables = variables
        self.energies = energies
        self.num_occurrences = num_occurrences
        self.timing = timing
        self.sampleset_touched = False

    @property
    def sampleset(self):
        self.sampleset_touched = True
        raise AssertionError("answer_view must not build a SampleSet")


def _future(**kw):
    base = dict(
        samples=np.array([[1, -1], [-1, -1]], dtype=np.int8),
        variables=[10, 20],
        energies=[-3.0, -1.0],
        num_occurrences=[3, 5],
        timing={"qpu_programming_time": 33_000, "qpu_sampling_time": 10_200},
    )
    base.update(kw)
    return FakeFuture(**base)


def test_a_cloud_future_is_read_without_building_a_sampleset():
    view = answer_view(_future())

    assert isinstance(view, AnswerView)
    assert view.spins.tolist() == [[1, -1], [-1, -1]]
    assert view.variables == [10, 20]
    assert view.energies == [-3.0, -1.0]


def test_reads_count_anneals_performed_not_distinct_solutions():
    # The cloud client folds identical reads into one row carrying
    # num_occurrences. Reporting the row count would under-report the anneals
    # the QPU actually ran, which is what SamplerMeta.reads means.
    view = answer_view(_future(num_occurrences=[3, 5]))

    assert view.reads == 8


def test_access_time_is_programming_plus_sampling():
    view = answer_view(_future())

    assert view.access_time_us == 43_200


def test_missing_timing_bills_nothing_rather_than_guessing():
    view = answer_view(_future(timing={}))

    assert view.access_time_us == 0


def test_spins_are_normalised_to_plus_or_minus_one():
    # Offline samplers can hand back 0/1. The wire format is one signed byte
    # per spin, so a 0 would reach the coordinator as a third value.
    view = answer_view(_future(samples=np.array([[0, 1]], dtype=np.int8)))

    assert view.spins.tolist() == [[-1, 1]]
    assert view.spins.dtype == np.int8


def test_a_dimod_sampleset_reads_the_same_way():
    # The mock and injected-sampler paths still return a SampleSet.
    dimod = pytest.importorskip("dimod")
    ss = dimod.SampleSet.from_samples(
        ([[1, -1], [-1, -1]], [10, 20]), vartype="SPIN", energy=[-3.0, -1.0]
    )
    ss = ss.copy()
    ss.info.update(
        timing={"qpu_programming_time": 33_000, "qpu_sampling_time": 10_200}
    )

    view = answer_view(ss)

    assert view.spins.tolist() == [[1, -1], [-1, -1]]
    assert view.variables == [10, 20]
    assert view.energies == [-3.0, -1.0]
    assert view.access_time_us == 43_200


def test_an_empty_answer_does_not_crash():
    view = answer_view(
        _future(
            samples=np.zeros((0, 2), dtype=np.int8),
            energies=[],
            num_occurrences=[],
        )
    )

    assert view.spins.shape == (0, 2)
    assert view.reads == 0
