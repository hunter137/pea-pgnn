# Concrete compressive-strength development

This document describes how to use PEA-PGNN for an age-dependent compressive-
strength curve. It is an adaptation recipe, not a validated strength model or
an implementation of a particular design standard.

## Appropriate and inappropriate tasks

The current architecture is suitable when the scientific target is the latent
mean strength-development trajectory of a fixed mixture and curing condition:

- strength is expressed in a non-negative unit such as MPa;
- age or justified equivalent age is non-negative;
- mean strength should not decrease over the modeled domain; and
- a finite long-term strength scale is meaningful.

It is not the right architecture for a dataset containing only one common age,
such as a pure 28-day tabular prediction problem. It also does not represent
freeze--thaw, fire, chemical attack, fatigue, or other deterioration processes
whose response can decline. Those tasks require a static model or different
candidate laws and guarantees.

## Recommended long-table schema

Store one observation per row. Repeated ages for the same mixture use the same
`mix_id`, context, and prior anchors.

| Column | Role | Example unit |
| --- | --- | --- |
| `mix_id` | grouping only; not a numeric feature | identifier |
| `age_days` | `time` | day |
| `strength_mpa` | `target` | MPa |
| `magnitude_prior_mpa` | `magnitude_prior` | MPa |
| `timescale_prior_days` | `timescale_prior` | day |
| mixture proportions | `context` | kg/m3 or documented ratio |
| curing descriptors | `context` if time-invariant | documented unit/category |

Possible context columns include water--binder ratio, water, cement, fly ash,
slag, silica fume, fine and coarse aggregate, admixture dosage, cement strength
class, specimen geometry, and time-invariant curing descriptors. Encode
categorical values before fitting and preserve the exact column order for
inference.

Do not place `age_days` in context. If temperature changes with time, a single
time-invariant temperature field can be inadequate. Derive and justify an
equivalent age or extend the model for time-varying covariates rather than
hiding a time series inside a static vector.

## Anchor strategies

The preferred approach starts from a documented empirical or mechanistic
strength-development curve

\[
f_c^p(t)=A^p g^p(t;\tau^p),
\]

where `A^p` becomes `magnitude_prior_mpa` and `tau^p` becomes
`timescale_prior_days`. State whether the magnitude represents an asymptotic
value or another long-term scale; it should not be casually equated with the
28-day design strength.

Early-age measurements can also calibrate the two anchors for a later-age
forecast. The prediction task must then be named by its information cutoff,
for example, "predicting 28--365-day strength from observations through day
7." Never use measurements beyond that cutoff to build the anchors.

If a separate machine-learning model creates the anchors, fit it only on the
training portion or generate out-of-fold anchors. Always evaluate the anchor
alone. A PEA-PGNN that cannot improve on its prior has not demonstrated the
value of its learned correction.

## Strength-specific starting configuration

Use the explicit preset rather than the drying-shrinkage defaults:

```python
from pea_pgnn import ModelConfig

config = ModelConfig.for_concrete_strength()
```

It starts with MPa- and day-scale assumptions:

```text
magnitude_bounds               = (1.0, 200.0) MPa
timescale_bounds               = (0.25, 1500.0) day
magnitude_relative_bounds      = (-0.4, 0.6)
timescale_relative_bounds      = (-0.8, 2.0)
additive_magnitude_scale       = 15.0 MPa
alpha_bounds                   = (0.1, 1.5)
hidden_dims                    = (64, 32)
```

These are broad starting values, not certified concrete limits. Override them
from the declared database domain:

```python
config = ModelConfig.for_concrete_strength(
    hidden_dims=(32, 16),
    magnitude_bounds=(2.0, 120.0),
    timescale_bounds=(0.5, 730.0),
)
```

Inspect fitted values near the bounds and repeat the study with reasonable
alternatives. Model size should reflect the number of independent mixtures;
ten ages from one mixture do not create ten independent material conditions.

## Group-disjoint training workflow

```python
from pea_pgnn import (
    PriorAnchoredRegressor,
    TrainingConfig,
    grouped_train_validation_test_split,
    regression_metrics,
)

split = grouped_train_validation_test_split(df["mix_id"], seed=42)

feature_columns = [
    "water_binder_ratio",
    "water_kg_m3",
    "cement_kg_m3",
    "fly_ash_kg_m3",
    "slag_kg_m3",
    "fine_aggregate_kg_m3",
    "coarse_aggregate_kg_m3",
    "admixture_kg_m3",
    "curing_temperature_c",
]

context = df[feature_columns].to_numpy(float)
time = df["age_days"].to_numpy(float)
target = df["strength_mpa"].to_numpy(float)
magnitude_prior = df["magnitude_prior_mpa"].to_numpy(float)
timescale_prior = df["timescale_prior_days"].to_numpy(float)

def arrays(index):
    return (
        context[index],
        time[index],
        target[index],
        magnitude_prior[index],
        timescale_prior[index],
    )

regressor = PriorAnchoredRegressor(
    model_config=ModelConfig.for_concrete_strength(),
    training_config=TrainingConfig(seed=42),
)
regressor.fit(
    *arrays(split.train_indices),
    validation_data=arrays(split.validation_indices),
)

prediction = regressor.predict(
    context[split.test_indices],
    time[split.test_indices],
    magnitude_prior[split.test_indices],
    timescale_prior[split.test_indices],
)
print(regression_metrics(target[split.test_indices], prediction))
```

The package scales numeric context using training data only. The application
must handle missing values, category encoding, unit conversion, plausibility
checks, duplicate resolution, and versioned feature ordering before this call.

## Predict and audit one complete curve

For a new mixture, repeat its unchanged context and two unchanged anchors along
a dense age grid:

```python
import numpy as np

from pea_pgnn import audit_trajectory

age_grid = np.linspace(0.0, 365.0, 366)
curve_context = np.repeat(new_mix_context[None, :], len(age_grid), axis=0)

details = regressor.predict_details(
    context=curve_context,
    time=age_grid,
    magnitude_prior=new_magnitude_prior_mpa,
    timescale_prior=new_timescale_prior_days,
)
report = audit_trajectory(
    details["prediction"],
    magnitude=details["magnitude"],
)
assert report.passed
```

The learned `magnitude`, `timescale`, `alpha`, and mixture `weights` are useful
model quantities, not independently measured material properties.

## Minimum scientific comparisons

Compare against the empirical prior, a single empirical curve refitted on the
training data, an unconstrained neural or tree model, and PEA-PGNN ablations.
Use at least group-disjoint unseen-mixture evaluation and a declared temporal-
extrapolation protocol. External laboratory or database validation is needed
before making transfer claims.

The runnable synthetic demonstration is
[`examples/concrete_strength_development.py`](../examples/concrete_strength_development.py).
It verifies software wiring and constraints only; its metrics are not evidence
of real-concrete accuracy.
