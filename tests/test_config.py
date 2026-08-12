import pytest

from pea_pgnn import ModelConfig


def test_concrete_strength_preset_uses_mpa_and_day_scale_bounds() -> None:
    config = ModelConfig.for_concrete_strength()

    assert config.hidden_dims == (64, 32)
    assert config.magnitude_bounds == (1.0, 200.0)
    assert config.timescale_bounds == (0.25, 1500.0)
    assert config.additive_magnitude_scale == 15.0
    assert config.alpha_bounds == (0.1, 1.5)


def test_concrete_strength_preset_allows_valid_overrides() -> None:
    config = ModelConfig.for_concrete_strength(
        hidden_dims=(32, 16),
        magnitude_bounds=(2.0, 120.0),
    )

    assert config.hidden_dims == (32, 16)
    assert config.magnitude_bounds == (2.0, 120.0)
    assert config.timescale_bounds == (0.25, 1500.0)


def test_concrete_strength_preset_still_validates_overrides() -> None:
    with pytest.raises(ValueError, match="magnitude_bounds"):
        ModelConfig.for_concrete_strength(magnitude_bounds=(200.0, 1.0))
