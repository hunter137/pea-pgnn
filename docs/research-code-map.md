# Research-code map and first-release scope

This document records how the reusable package relates to the manuscript's
working research code. The public package was rewritten into import-safe
modules; it is not a byte-for-byte copy of the monolithic analysis scripts.

## Added on the `0.2.0.dev0` development line

The domain-adaptation guide, concrete-strength development walkthrough,
strength-specific configuration preset, grouped split helper, and synthetic
strength example demonstrate how the reusable architecture can be wired to a
second non-negative, monotone, bounded response. They do not add a calibrated
strength database, a certified strength formulation, trained weights, or a
real-concrete performance claim.

## Included in `0.1.1`

| Research concept | Public implementation |
| --- | --- |
| empirical magnitude anchors | `pea_pgnn.concrete` |
| B3-inspired characteristic-time anchor | `pea_pgnn.concrete.b3_timescale` |
| bounded magnitude and timescale correction | `PriorAnchoredTemporalModel` |
| four candidate time-evolution laws | `pea_pgnn.temporal` and the model |
| non-negative normalized context weights | model softmax head |
| structured magnitude-times-evolution prediction | model forward pass |
| robust context scaling and early-stopped fitting | `PriorAnchoredRegressor` |
| point-trajectory property checks | `audit_trajectory` |
| basic prediction metrics | `regression_metrics` |

The correction heads are initialized to exactly zero correction. This fixes an
ambiguity in the working script, where zero raw head output was described as
anchor-centered even though asymmetric sigmoid bounds map zero to the interval
midpoint. The cleaned package computes the logit needed to map each asymmetric
interval to a true zero correction.

Version `0.1.1` also adds an explicit empirical-prior provenance document,
model card, release/security/community files, stable implementation reference
cases, NumPy/PyTorch temporal-law parity checks, and broader metric/input tests.

## Deliberately not included in the reusable core

- the Northwestern University shrinkage database or derived CSV files;
- paper-specific data cleaning, database-ID resolution, profile reconstruction,
  and fixed experimental split files;
- NSGA-II architecture search and TOPSIS selection;
- manuscript-specific ablation suites and baseline orchestration;
- heteroscedastic and conformal uncertainty-quantification experiments;
- SHAP, integrated-gradient, Sobol, Morris, interaction, and response-surface
  analyses;
- plotting scripts, generated figures, tables, checkpoints, and GUI assets;
- manuscript drafts, reviewer audits, reference corpora, and submission files.

These materials have different dependency, licensing, privacy, and
reproducibility requirements. If released, they should be organized as an
explicit reproduction layer or a separately archived companion repository
rather than being imported whenever a user runs `import pea_pgnn`.

## Claim boundary

The reusable estimator's default validation split is a convenience for software
use, not the manuscript's scientific evaluation protocol. Paper-level claims
must come from the documented condition-disjoint or represented-condition
temporal protocols, fold-fitted preprocessing, and paired condition-profile
inference. Users should construct those splits externally and pass an explicit
validation set when fitting.
