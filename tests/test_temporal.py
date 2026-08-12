import numpy as np
import pytest

from pea_pgnn import audit_trajectory, candidate_time_laws, convex_time_evolution


def test_candidate_laws_are_bounded_and_monotonic() -> None:
    time = np.linspace(0.0, 2000.0, 500)
    laws = candidate_time_laws(time, timescale=80.0, alpha=0.5)
    assert laws.shape == (500, 4)
    assert np.all(laws >= 0.0)
    assert np.all(laws <= 1.0)
    assert np.all(np.diff(laws, axis=0) >= -1e-12)


def test_convex_mixture_and_constraint_audit() -> None:
    time = np.linspace(0.0, 1000.0, 200)
    evolution = convex_time_evolution(time, 70.0, [0.1, 0.2, 0.3, 0.4])
    prediction = 600.0 * evolution
    report = audit_trajectory(prediction, magnitude=600.0)
    assert report.passed
    assert prediction[0] == pytest.approx(0.0)


def test_invalid_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        convex_time_evolution([1.0, 2.0], 10.0, [0.1, 0.2, 0.3, 0.3])

