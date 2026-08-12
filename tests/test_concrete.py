import numpy as np
import pytest

from pea_pgnn.concrete import (
    aci209_shrinkage,
    b3_shrinkage,
    concrete_prior_anchors,
    ec2_shrinkage,
    gl2000_shrinkage,
)


def test_concrete_priors_are_vectorized_and_positive() -> None:
    priors = concrete_prior_anchors(
        loading_age=np.array([7.0, 14.0]),
        relative_humidity=60.0,
        volume_surface_ratio=np.array([40.0, 60.0]),
        water_content=180.0,
        compressive_strength=40.0,
    )
    assert priors["magnitude"].shape == (2,)
    assert np.all(priors["magnitude"] > 0.0)
    assert np.all(priors["timescale"] > 0.0)


def test_concrete_prior_reference_case_is_stable() -> None:
    """Lock the public implementation, not independent code conformance."""

    priors = concrete_prior_anchors(
        loading_age=7.0,
        relative_humidity=60.0,
        volume_surface_ratio=50.0,
        water_content=180.0,
        compressive_strength=40.0,
    )
    expected = {
        "magnitude": 584.8934682777909,
        "timescale": 289.2651304724738,
        "b3_magnitude": 502.4736688803325,
        "gl2000_magnitude": 660.2272837510428,
        "aci209_magnitude": 591.9794522019972,
    }
    assert set(priors) == set(expected)
    for name, value in expected.items():
        assert float(priors[name]) == pytest.approx(value, rel=1e-12)


@pytest.mark.parametrize(
    "function", [b3_shrinkage, gl2000_shrinkage, aci209_shrinkage, ec2_shrinkage]
)
def test_empirical_trajectories_are_monotonic(function) -> None:
    time = np.linspace(0.0, 1000.0, 100)
    common = {
        "time": time,
        "relative_humidity": 60.0,
        "volume_surface_ratio": 50.0,
    }
    if function is b3_shrinkage:
        values = function(
            **common,
            loading_age=7.0,
            water_content=180.0,
            compressive_strength=40.0,
        )
    elif function is gl2000_shrinkage:
        values = function(**common, compressive_strength=40.0)
    elif function is aci209_shrinkage:
        values = function(**common)
    else:
        values = function(
            time=time,
            relative_humidity=60.0,
            notional_size=100.0,
            compressive_strength=40.0,
        )
    assert values[0] == pytest.approx(0.0)
    assert np.all(np.diff(values) >= -1e-10)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"loading_age": 0.0}, "loading_age"),
        ({"relative_humidity": 101.0}, "relative_humidity"),
        ({"volume_surface_ratio": 0.0}, "volume_surface_ratio"),
        ({"water_content": -1.0}, "water_content"),
        ({"compressive_strength": 0.0}, "compressive_strength"),
    ],
)
def test_invalid_concrete_prior_inputs_are_rejected(kwargs, message) -> None:
    inputs = {
        "loading_age": 7.0,
        "relative_humidity": 60.0,
        "volume_surface_ratio": 50.0,
        "water_content": 180.0,
        "compressive_strength": 40.0,
    }
    inputs.update(kwargs)
    with pytest.raises(ValueError, match=message):
        concrete_prior_anchors(**inputs)
