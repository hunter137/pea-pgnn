"""Minimal synthetic demonstration of training and constraint auditing."""

import numpy as np

from pea_pgnn import (
    ModelConfig,
    PriorAnchoredRegressor,
    TrainingConfig,
    audit_trajectory,
    regression_metrics,
)


def main() -> None:
    rng = np.random.default_rng(7)
    n_conditions = 24
    times = np.array([7.0, 28.0, 90.0, 180.0, 365.0])
    condition_context = rng.uniform(-1.0, 1.0, size=(n_conditions, 3))
    magnitude = 550.0 + 80.0 * condition_context[:, 0]
    timescale = 70.0 + 15.0 * condition_context[:, 1]

    context = np.repeat(condition_context, len(times), axis=0)
    time = np.tile(times, n_conditions)
    magnitude_prior = np.repeat(magnitude * 0.92, len(times))
    timescale_prior = np.repeat(timescale * 1.10, len(times))
    truth = np.repeat(magnitude, len(times)) * np.tanh(
        np.sqrt(time / np.repeat(timescale, len(times)))
    )
    target = truth + rng.normal(0.0, 6.0, size=truth.shape)

    regressor = PriorAnchoredRegressor(
        model_config=ModelConfig(hidden_dims=(32, 16), dropout=0.0),
        training_config=TrainingConfig(
            epochs=120,
            batch_size=64,
            patience=25,
            learning_rate=1e-3,
            seed=7,
            verbose=True,
        ),
    )
    regressor.fit(context, time, target, magnitude_prior, timescale_prior)
    prediction = regressor.predict(context, time, magnitude_prior, timescale_prior)
    print(regression_metrics(target, prediction))

    grid = np.linspace(0.0, 800.0, 200)
    repeated_context = np.repeat(condition_context[:1], len(grid), axis=0)
    grid_prediction = regressor.predict(
        repeated_context,
        grid,
        magnitude_prior[0],
        timescale_prior[0],
    )
    details = regressor.predict_details(
        repeated_context,
        grid,
        magnitude_prior[0],
        timescale_prior[0],
    )
    print(audit_trajectory(grid_prediction, magnitude=details["magnitude"]))


if __name__ == "__main__":
    main()

