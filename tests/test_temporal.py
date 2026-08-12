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


@pytest.mark.parametrize(
    ("time", "timescale", "alpha", "message"),
    [
        ([-1.0], 10.0, 0.5, "time"),
        ([1.0], 0.0, 0.5, "timescale"),
        ([1.0], 10.0, 0.0, "alpha"),
    ],
)
def test_invalid_candidate_law_inputs_are_rejected(
    time, timescale, alpha, message
) -> None:
    with pytest.raises(ValueError, match=message):
        candidate_time_laws(time, timescale, alpha)


def test_numpy_and_torch_candidate_laws_agree() -> None:
    torch = pytest.importorskip("torch")
    from pea_pgnn import ModelConfig, PriorAnchoredTemporalModel

    time = np.array([0.0, 7.0, 28.0, 365.0])
    timescale = np.full_like(time, 80.0)
    alpha = np.full_like(time, 0.45)
    numpy_laws = candidate_time_laws(time, timescale, alpha)
    model = PriorAnchoredTemporalModel(1, ModelConfig(hidden_dims=(4,), dropout=0.0))
    torch_laws = model.candidate_laws(
        torch.tensor(time), torch.tensor(timescale), torch.tensor(alpha)
    ).numpy()
    np.testing.assert_allclose(numpy_laws, torch_laws, rtol=1e-12, atol=1e-12)

