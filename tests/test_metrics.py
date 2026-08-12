import math

import pytest

from pea_pgnn import regression_metrics


def test_regression_metrics_known_values_and_finite_filtering() -> None:
    result = regression_metrics(
        target=[1.0, 2.0, 3.0, math.nan],
        prediction=[1.0, 3.0, 2.0, 99.0],
    )
    assert result["n"] == 3
    assert result["r2"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(math.sqrt(2.0 / 3.0))
    assert result["mae"] == pytest.approx(2.0 / 3.0)
    assert result["mape"] == pytest.approx((0.0 + 0.5 + 1.0 / 3.0) / 3.0 * 100.0)


def test_regression_metrics_rejects_unusable_arrays() -> None:
    with pytest.raises(ValueError, match="equal shape"):
        regression_metrics([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="no jointly finite"):
        regression_metrics([math.nan], [math.nan])


def test_constant_target_has_undefined_r_squared() -> None:
    result = regression_metrics([2.0, 2.0], [2.0, 3.0])
    assert math.isnan(result["r2"])

