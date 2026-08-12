# PEA-PGNN

[![License: MIT](https://img.shields.io/github/license/hunter137/pea-pgnn?style=flat-square)](https://github.com/hunter137/pea-pgnn/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.9-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/pea-pgnn?style=flat-square)](https://pypi.org/project/pea-pgnn/)
[![Version](https://img.shields.io/badge/version-0.1.1-2ea44f?style=flat-square)](https://github.com/hunter137/pea-pgnn/blob/main/CHANGELOG.md)
[![Tests](https://github.com/hunter137/pea-pgnn/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/hunter137/pea-pgnn/actions/workflows/tests.yml)
[![Ruff](https://img.shields.io/badge/lint-Ruff-D7FF64?style=flat-square&logo=ruff&logoColor=261230)](https://docs.astral.sh/ruff/)
[![GitHub release](https://img.shields.io/github/v/release/hunter137/pea-pgnn?style=flat-square)](https://github.com/hunter137/pea-pgnn/releases)
[![GitHub stars](https://img.shields.io/github/stars/hunter137/pea-pgnn?style=flat-square)](https://github.com/hunter137/pea-pgnn/stargazers)

[Documentation](https://github.com/hunter137/pea-pgnn/tree/main/docs) ·
[Examples](https://github.com/hunter137/pea-pgnn/tree/main/examples) ·
[Public API](#public-api) ·
[Model card](https://github.com/hunter137/pea-pgnn/blob/main/MODEL_CARD.md) ·
[Contributing](https://github.com/hunter137/pea-pgnn/blob/main/CONTRIBUTING.md) ·
[Issues](https://github.com/hunter137/pea-pgnn/issues) ·
[Releases](https://github.com/hunter137/pea-pgnn/releases)

**Prior-anchored, structure-preserving neural prediction for time-dependent
engineering responses.**

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

> Status: `0.1.1` alpha. This repository is a clean, reusable implementation
> extracted from research code. It is not yet the exact reproduction archive
> for every experiment reported in the manuscript.

## Highlights

- **Prior-anchored learning:** empirical estimates enter the model as
  correctable magnitude and timescale anchors rather than fixed answers.
- **Structure-preserving evolution:** convex mixtures of normalized temporal
  laws inherit non-negativity, monotonicity, and boundedness by construction.
- **Two levels of use:** work with the low-level PyTorch module or the fitted
  NumPy-style regressor interface.
- **Concrete application:** use B3-, GL2000-, ACI209-, and EC2-inspired
  utilities to construct drying-shrinkage priors.
- **Auditable behavior:** check predicted trajectories and standard regression
  metrics with reusable evaluation utilities.
- **Research-oriented documentation:** explicit input contracts, method
  boundaries, reproducibility scope, tests, and a runnable synthetic example.

## Installation

### Requirements

- Python 3.9 or later;
- NumPy 1.23 or later;
- scikit-learn 1.2 or later; and
- PyTorch 2.0 or later.

The runtime dependencies are installed automatically. A GPU is optional; for a
CUDA- or ROCm-specific PyTorch build, follow the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/)
before installing PEA-PGNN.

### Install from PyPI

Install the released package with:

```bash
python -m pip install pea-pgnn
```

### Install from GitHub

The latest development version can be installed directly from GitHub:

```bash
python -m pip install "git+https://github.com/hunter137/pea-pgnn.git"
```

Alternatively, clone the repository and install it locally:

```bash
git clone https://github.com/hunter137/pea-pgnn.git
cd pea-pgnn
python -m pip install -e .
```

### Development install

For development and testing:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

### Package naming

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

`concrete_prior_anchors` returns `magnitude`, `timescale`, and the three
component estimates `b3_magnitude`, `gl2000_magnitude`, and
`aci209_magnitude`. Their formulation lineage, units, simplifications, and
applicability limits are documented in
[`docs/empirical-priors.md`](https://github.com/hunter137/pea-pgnn/blob/main/docs/empirical-priors.md).

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
[`examples/synthetic_demo.py`](https://github.com/hunter137/pea-pgnn/blob/main/examples/synthetic_demo.py).
Run it after a development install with:

```bash
python examples/synthetic_demo.py
```

It prints epoch losses, regression metrics, and a `ConstraintReport` whose
`passed=True` value verifies the sampled point trajectory. The example is a
software demonstration on synthetic data, not a benchmark claim.

> **Evaluation warning:** when several rows belong to one physical condition,
> the estimator's fallback random validation split is only a convenience for
> optimization. It is generally not a defensible temporal-extrapolation
> protocol. Keep condition groups intact, construct the cutoff externally, and
> pass `validation_data=` explicitly for scientific evaluation.

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

`PriorAnchoredRegressor.predict_details` returns `prediction`, corrected
`magnitude` and `timescale`, `alpha`, normalized `weights`, the four
`candidate_laws`, their mixed `evolution`, and the three learned correction
terms. These outputs are interpretable model quantities, not automatically
identifiable material properties.

## Documentation and support

- [Method](https://github.com/hunter137/pea-pgnn/blob/main/docs/method.md):
  computational formulation and inherited structural properties.
- [Empirical-prior provenance](https://github.com/hunter137/pea-pgnn/blob/main/docs/empirical-priors.md):
  source formulations, units, compact implementation choices, and applicability
  limits.
- [Data contract](https://github.com/hunter137/pea-pgnn/blob/main/docs/data-contract.md):
  required inputs, shapes, units, and evaluation cautions.
- [Model card](https://github.com/hunter137/pea-pgnn/blob/main/MODEL_CARD.md):
  intended uses, out-of-scope uses, outputs, evaluation guidance, and risks.
- [Research-code map](https://github.com/hunter137/pea-pgnn/blob/main/docs/research-code-map.md):
  relationship between this package and the working manuscript code.
- [Synthetic example](https://github.com/hunter137/pea-pgnn/blob/main/examples/synthetic_demo.py):
  complete training, prediction, and trajectory-audit workflow.

For bugs, unexpected behavior, or feature requests, open a
[GitHub issue](https://github.com/hunter137/pea-pgnn/issues) and include a
minimal reproducible example, your Python version, and your PEA-PGNN version.
Contributions are welcome; see the
[contribution guide](https://github.com/hunter137/pea-pgnn/blob/main/CONTRIBUTING.md)
before submitting a pull request.
Security-sensitive reports should follow the private route in the
[security policy](https://github.com/hunter137/pea-pgnn/blob/main/SECURITY.md),
not a public issue.

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
training domain. See the
[method documentation](https://github.com/hunter137/pea-pgnn/blob/main/docs/method.md)
and
[data contract](https://github.com/hunter137/pea-pgnn/blob/main/docs/data-contract.md).

The relationship between the cleaned package and the working manuscript code is
listed in the
[research-code map](https://github.com/hunter137/pea-pgnn/blob/main/docs/research-code-map.md).

## Reproducibility scope

This repository intentionally separates reusable method code from research
assets. To reproduce the manuscript experiments exactly, a separate archival
release should later pin the evaluated dataset version, split identifiers,
hyperparameters, trained checkpoints, and figure/table scripts, subject to data
licensing and manuscript-publication constraints.

## Citation

If this software is useful in your research, please cite the software release
using the repository's **Cite this repository** menu. The metadata are stored in
[`CITATION.cff`](https://github.com/hunter137/pea-pgnn/blob/main/CITATION.cff).
The associated manuscript is still being prepared; its final bibliographic
citation and DOI will be added when available.

## Authors

- **Deyu Liang** — School of Transportation and Surveying Engineering,
  Shenyang Jianzhu University, Shenyang, China
- **Jinlong Liu** — School of Civil Engineering, Southeast University, Nanjing,
  China
- **Lei Xu** — Laboratory of Construction Materials, École Polytechnique
  Fédérale de Lausanne, Lausanne, Switzerland

## Acknowledgements

This work was supported by the National Key R&D Program of China (2024YFC38098,
2024YFC3809803); the Liaoning Xingliao Talents Program for Science and
Technology Innovation Team of China (No. XLYC2404005); the Technology Research
and Development Program of Shenyang Science and Technology Bureau (Grant No.
24-213-3-33).

## License

PEA-PGNN is open-source software released under the
[`MIT License`](https://github.com/hunter137/pea-pgnn/blob/main/LICENSE). It may
be used, copied, modified, distributed, sublicensed, and sold under the terms of
that license. The software is provided without warranty.

---

## 中文说明

PEA-PGNN 将论文中的知识体系整理为可复用代码：把经验量作为可修正锚点，把多种时间演化规律组成凸组合，并由前向结构保证点预测的非负、单调和有界性质。经验模型来源、单位、简化假设及适用边界见 `docs/empirical-priors.md`；模型用途、风险和评估要求见 `MODEL_CARD.md`。

当前仓库是首个干净的软件包版本，包含核心模型、混凝土经验先验、训练封装、约束检查、测试和示例；不包含论文数据库、训练权重、论文正文、审稿材料和实验输出。本项目采用 MIT 开源许可证，作者为 Deyu Liang、Jinlong Liu 和 Lei Xu，资助信息见上方 Acknowledgements。
