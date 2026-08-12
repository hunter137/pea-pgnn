# PEA-PGNN

PEA-PGNN is a research-oriented Python package for **prior-anchored,
structure-preserving prediction of time-dependent engineering responses**. It
turns three types of incomplete engineering knowledge into explicit
computational roles:

1. quantitative empirical estimates become correctable parameter anchors;
2. alternative temporal laws form a context-conditioned convex mixture; and
3. non-negativity, monotonicity, and boundedness are inherited from the
   forward construction.

The first application bundled with the package is long-term concrete drying
shrinkage prediction. The package contains the method and empirical-prior
utilities, but **does not contain the paper's database, trained weights, or
submission files**.

> Status: `0.1.0` alpha. This repository is a clean, reusable implementation
> extracted from research code. It is not yet the exact reproduction archive
> for every experiment reported in the manuscript.

## Installation

After cloning the repository:

```bash
python -m pip install -e .
```

For development and testing:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Once a later release is published to PyPI, the intended command will be:

```bash
python -m pip install pea-pgnn
```

The distribution name contains a hyphen, while the Python import name uses an
underscore:

```python
import pea_pgnn
```

## Quick start: concrete empirical anchors

Inputs may be scalars or broadcast-compatible NumPy arrays. Shrinkage is
returned in microstrain.

```python
from pea_pgnn.concrete import concrete_prior_anchors

priors = concrete_prior_anchors(
    loading_age=7.0,          # d
    relative_humidity=60.0,  # %
    volume_surface_ratio=50.0,  # mm
    water_content=180.0,     # kg/m^3
    compressive_strength=40.0,  # MPa
)

print(priors["magnitude"])
print(priors["timescale"])
```

## Quick start: structured temporal evolution

The four normalized candidate laws can also be used without training a neural
network:

```python
import numpy as np

from pea_pgnn import convex_time_evolution

time = np.array([0.0, 7.0, 28.0, 90.0, 365.0])
evolution = convex_time_evolution(
    time=time,
    timescale=80.0,
    alpha=0.5,
    weights=[0.25, 0.25, 0.25, 0.25],
)

prediction = 650.0 * evolution
```

## Quick start: train a prior-anchored model

`PriorAnchoredRegressor` offers a compact NumPy-style interface. Context must
contain only time-invariant condition descriptors; query time is passed
separately so that mixture weights remain fixed along one condition trajectory.

```python
from pea_pgnn import PriorAnchoredRegressor, TrainingConfig

regressor = PriorAnchoredRegressor(
    training_config=TrainingConfig(epochs=300, patience=40, seed=42)
)

regressor.fit(
    context=X_train,
    time=t_train,
    target=y_train,
    magnitude_prior=A_prior_train,
    timescale_prior=tau_prior_train,
)

y_pred = regressor.predict(
    context=X_test,
    time=t_test,
    magnitude_prior=A_prior_test,
    timescale_prior=tau_prior_test,
)
```

A complete runnable synthetic example is provided in
[`examples/synthetic_demo.py`](examples/synthetic_demo.py).

## Public API

- `PriorAnchoredTemporalModel`: low-level PyTorch module.
- `PriorAnchoredRegressor`: fitted preprocessing, training, prediction, and
  checkpoint wrapper.
- `ModelConfig` and `TrainingConfig`: explicit model and optimization settings.
- `candidate_time_laws` and `convex_time_evolution`: NumPy implementations of
  the structured temporal basis.
- `audit_trajectory`: numerical audit of non-negativity, monotonicity, and
  optional upper boundedness.
- `regression_metrics`: R-squared, RMSE, MAE, and MAPE.
- `pea_pgnn.concrete`: B3-, GL2000-, ACI209-, and EC2-inspired empirical
  shrinkage utilities used by the concrete implementation.

## Method boundary

The construction guarantees point-prediction properties only when its input
contract is respected:

- query time is non-negative and supplied separately from condition context;
- candidate weights are constant with time for a fixed context;
- candidate laws are non-negative, monotone, and bounded on the implemented
  temporal domain; and
- the learned response magnitude is positive and bounded.

These properties do not automatically extend to prediction-interval endpoints.
They also do not establish accuracy, transferability, or validity outside the
training domain. See [`docs/method.md`](docs/method.md) and
[`docs/data-contract.md`](docs/data-contract.md).

The relationship between the cleaned package and the working manuscript code is
listed in [`docs/research-code-map.md`](docs/research-code-map.md).

## Reproducibility scope

This repository intentionally separates reusable method code from research
assets. To reproduce the manuscript experiments exactly, a separate archival
release should later pin the evaluated dataset version, split identifiers,
hyperparameters, trained checkpoints, and figure/table scripts, subject to data
licensing and manuscript-publication constraints.

## Citation

The associated manuscript is still being prepared. A final citation and
`CITATION.cff` should be added after the title, author list, DOI, and software
archive DOI are confirmed.

## License

PEA-PGNN is open-source software released under the
[`MIT License`](LICENSE). It may be used, copied, modified, distributed,
sublicensed, and sold under the terms of that license. The software is provided
without warranty.

---

## 中文说明

PEA-PGNN 将论文中的知识体系整理为可复用代码：把经验量作为可修正锚点，把多种时间演化规律组成凸组合，并由前向结构保证点预测的非负、单调和有界性质。

当前仓库是首个干净的软件包版本，包含核心模型、混凝土经验先验、训练封装、约束检查、测试和示例；不包含论文数据库、训练权重、论文正文、审稿材料和实验输出。本项目采用 MIT 开源许可证。正式公开时仍应确认作者署名、数据权利以及专利或投稿时机。
