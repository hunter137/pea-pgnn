"""Synthetic, leakage-resistant concrete strength-development demonstration."""

import numpy as np

from pea_pgnn import (
    ModelConfig,
    PriorAnchoredRegressor,
    TrainingConfig,
    audit_trajectory,
    grouped_train_validation_test_split,
    regression_metrics,
)


def main() -> None:
    rng = np.random.default_rng(17)
    n_mixtures = 30
    ages = np.array([1.0, 3.0, 7.0, 14.0, 28.0, 90.0])

    # Three time-invariant descriptors: water--binder ratio, supplementary
    # cementitious-material fraction, and curing temperature in degrees Celsius.
    water_binder_ratio = rng.uniform(0.30, 0.58, n_mixtures)
    scm_fraction = rng.uniform(0.0, 0.35, n_mixtures)
    curing_temperature = rng.uniform(15.0, 28.0, n_mixtures)
    mixture_context = np.column_stack(
        [water_binder_ratio, scm_fraction, curing_temperature]
    )

    true_magnitude = (
        108.0
        - 105.0 * water_binder_ratio
        + 11.0 * scm_fraction
        + 0.30 * (curing_temperature - 20.0)
    )
    true_timescale = (
        9.0
        + 35.0 * water_binder_ratio
        + 35.0 * scm_fraction
        - 0.22 * (curing_temperature - 20.0)
    )

    context = np.repeat(mixture_context, len(ages), axis=0)
    time = np.tile(ages, n_mixtures)
    groups = np.repeat([f"mix-{index:03d}" for index in range(n_mixtures)], len(ages))
    magnitude = np.repeat(true_magnitude, len(ages))
    timescale = np.repeat(true_timescale, len(ages))

    clean_strength = magnitude * np.tanh(np.sqrt(time / timescale))
    target = np.maximum(clean_strength + rng.normal(0.0, 0.8, len(time)), 0.0)

    # Synthetic imperfect priors stand in for a documented empirical strength
    # model. Real applications must construct them without using test targets.
    magnitude_prior = magnitude * (0.92 + 0.02 * np.repeat(scm_fraction, len(ages)))
    timescale_prior = timescale * (
        1.12 - 0.02 * np.repeat(water_binder_ratio, len(ages))
    )

    split = grouped_train_validation_test_split(groups, seed=17)

    def arrays(indices: np.ndarray) -> tuple[np.ndarray, ...]:
        return (
            context[indices],
            time[indices],
            target[indices],
            magnitude_prior[indices],
            timescale_prior[indices],
        )

    regressor = PriorAnchoredRegressor(
        model_config=ModelConfig.for_concrete_strength(hidden_dims=(32, 16)),
        training_config=TrainingConfig(
            epochs=100,
            batch_size=64,
            learning_rate=1e-3,
            patience=20,
            seed=17,
            device="cpu",
        ),
    )
    regressor.fit(
        *arrays(split.train_indices),
        validation_data=arrays(split.validation_indices),
    )

    test_prediction = regressor.predict(
        context[split.test_indices],
        time[split.test_indices],
        magnitude_prior[split.test_indices],
        timescale_prior[split.test_indices],
    )
    print("test metrics:", regression_metrics(target[split.test_indices], test_prediction))

    mixture_index = split.test_indices[0] // len(ages)
    age_grid = np.linspace(0.0, 365.0, 366)
    curve_context = np.repeat(
        mixture_context[mixture_index : mixture_index + 1],
        len(age_grid),
        axis=0,
    )
    curve_details = regressor.predict_details(
        curve_context,
        age_grid,
        true_magnitude[mixture_index] * 0.92,
        true_timescale[mixture_index]
        * (1.12 - 0.02 * water_binder_ratio[mixture_index]),
    )
    report = audit_trajectory(
        curve_details["prediction"],
        magnitude=curve_details["magnitude"],
    )
    print("trajectory audit:", report)
    print(
        "fitted curve parameters:",
        {
            "magnitude_mpa": float(curve_details["magnitude"][0]),
            "timescale_days": float(curve_details["timescale"][0]),
            "alpha": float(curve_details["alpha"][0]),
            "weights": curve_details["weights"][0].round(4).tolist(),
        },
    )


if __name__ == "__main__":
    main()
