import numpy as np
import pytest

from pea_pgnn.concrete import (
    aci209_shrinkage,
    b3_shrinkage,
    concrete_prior_anchors,
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


@pytest.mark.parametrize("function", [b3_shrinkage, gl2000_shrinkage, aci209_shrinkage])
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
    else:
        values = function(**common)
    assert values[0] == pytest.approx(0.0)
    assert np.all(np.diff(values) >= -1e-10)

