"""What reaches SAPI: an anneal schedule on every job, and never a time.

SAPI refuses a problem that carries both ``annealing_time`` and
``anneal_schedule``, and a reverse anneal can only be written as a schedule.
These tests pin the submitted parameters on the real (non-mock) submit path,
with only the hand-off to the cloud client stubbed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from quip_miner_dwave.ocean import OceanSampler
from quip_miner_dwave.schedule import ScheduleError

_PARAMETERS = {
    "num_reads": None,
    "anneal_schedule": None,
    "initial_state": None,
    "reinitialize_state": None,
    "label": None,
}


class _Done:
    """A finished cloud problem, shaped like dwave.cloud.computation.Future."""

    samples = np.array([[1, -1]], dtype=np.int8)
    energies = [-1.5]
    num_occurrences = [1]
    variables = [0, 1]
    timing = {"qpu_programming_time": 100, "qpu_sampling_time": 50}

    def done(self):
        return True


def _cloud_sampler(properties=None):
    """A sampler on the real submit path whose solver publishes ``properties``."""
    s = OceanSampler(mock=False)
    s._connected = True
    s._is_mock = False

    published = (
        {
            "default_annealing_time": 20.0,
            "annealing_time_range": [0.5, 2000.0],
            # Labels have holes on a real chip: more physical qubits than
            # the two this solver exposes.
            "num_qubits": 4,
        }
        if properties is None
        else properties
    )

    class _Identity:
        def dict(self):
            return {"name": "FakeSolver", "version": {"graph_id": "x"}}

    class _Solver:
        _encoding_qubits = [0, 1]
        _encoding_couplers = [(0, 1)]
        _params: dict = {}
        parameters = dict(_PARAMETERS)
        properties = published
        identity = _Identity()

        def _format_params(self, type_, params):
            pass

    class _Sampler:
        solver = _Solver()

    s.sampler = _Sampler()
    bodies = []

    def _submit_encoded(solver, body, cancel_key):
        bodies.append(json.loads(body))
        return _Done()

    s._submit_encoded = _submit_encoded  # type: ignore[method-assign]
    return s, bodies


def _submit(s, **kwargs):
    return s.sample(
        np.array([0, 1]),
        np.array([1.0, -1.0]),
        np.array([(0, 1)]),
        np.array([0.5]),
        num_reads=4,
        **kwargs,
    )


def test_a_job_with_no_anneal_time_sends_the_solver_default_as_a_schedule():
    s, bodies = _cloud_sampler()
    _submit(s)

    params = bodies[0]["params"]
    assert params["anneal_schedule"] == [[0.0, 0.0], [20.0, 1.0]]
    assert "annealing_time" not in params
    s.close()


def test_an_anneal_time_override_becomes_the_end_of_the_ramp():
    s, bodies = _cloud_sampler()
    _submit(s, anneal_time_us=250)

    params = bodies[0]["params"]
    assert params["anneal_schedule"] == [[0.0, 0.0], [250.0, 1.0]]
    assert "annealing_time" not in params
    s.close()


def test_a_cold_job_sends_no_reverse_anneal_parameters():
    s, bodies = _cloud_sampler()
    _submit(s)

    assert "initial_state" not in bodies[0]["params"]
    assert "reinitialize_state" not in bodies[0]["params"]
    s.close()


def test_a_solver_that_publishes_no_default_gets_no_schedule_at_all():
    # Writing a guessed default onto the wire would change the anneal. With
    # nothing sent, the solver's own default applies, as it did before.
    s, bodies = _cloud_sampler(properties={})
    _submit(s)

    assert "anneal_schedule" not in bodies[0]["params"]
    assert "annealing_time" not in bodies[0]["params"]
    s.close()


def test_an_anneal_the_chip_cannot_run_raises_before_anything_is_submitted():
    s, bodies = _cloud_sampler()
    with pytest.raises(ScheduleError):
        _submit(s, anneal_time_us=5000)

    assert bodies == []
    s.close()
