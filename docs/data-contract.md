# Data contract

## Generic model inputs

`PriorAnchoredRegressor` expects five aligned arrays:

| Argument | Shape | Meaning |
| --- | --- | --- |
| `context` | `(n, p)` | finite, time-invariant condition descriptors |
| `time` | `(n,)` | non-negative query time |
| `target` | `(n,)` | non-negative observed response |
| `magnitude_prior` | `(n,)` | positive empirical response-scale anchor |
| `timescale_prior` | `(n,)` | positive empirical characteristic-time anchor |

Do not include query time in `context`. Categorical variables must be encoded
before fitting. The wrapper applies a `RobustScaler` to context only; it does not
impute missing values or infer units.

For a repeated condition trajectory, all context values and prior anchors must
remain identical while `time` changes.

## Concrete implementation units

The empirical utilities in `pea_pgnn.concrete` use the units below:

| Quantity | Unit |
| --- | --- |
| drying/query time | day |
| loading/start-of-drying age | day |
| relative humidity | percent, from 0 to 100 |
| volume-to-surface ratio | millimetre |
| notional size | millimetre |
| water content | kg/m^3 |
| compressive strength | MPa |
| returned drying shrinkage | microstrain |

The paper implementation used time-invariant descriptors such as loading age,
relative humidity, temperature, geometry, mixture proportions, strength,
cement-type indicators, derived humidity/strength/geometry terms, and empirical
prior estimates. Users are responsible for defining a consistent schema for
their own data and for preventing information leakage during split construction.

## Evaluation warning

A random row split is usually inappropriate for temporal extrapolation when
multiple observations belong to the same physical condition. Build the scientific
train/test protocol outside the estimator, keep condition groups intact, fit
preprocessing on training data only, and pass an explicit validation set to
`fit` when its grouping or temporal boundary matters.

