# Model card for PEA-PGNN

## Model summary

PEA-PGNN is a prior-anchored, structure-preserving neural regressor for
non-negative, non-decreasing, bounded time-dependent responses. A neural
network maps time-invariant context to bounded corrections of empirical
magnitude and timescale anchors, an exponent, and convex mixture weights over
four normalized temporal laws. Query time enters only through those laws.

This repository contains a reusable model implementation, not a pretrained
model or a complete reproduction archive.

## Intended uses

- research on knowledge-guided temporal extrapolation;
- controlled experiments comparing empirical anchors and candidate temporal
  laws;
- prediction of concrete drying-shrinkage magnitude when users supply and
  validate an appropriate dataset, split protocol, and prior inputs;
- methodological reuse in another non-negative saturating-response problem
  after redefining and validating its priors, laws, bounds, and units.

## Out-of-scope uses

- structural design, code-compliance decisions, or safety-critical assessment
  without independent engineering review;
- treating the bundled empirical utilities as certified implementations of
  their source standards;
- exact reproduction of the associated manuscript from this package alone;
- interpreting learned mixture weights as posterior probabilities or learned
  parameters as independently measured material properties;
- assuming accuracy, calibration, monotonic interval endpoints, or transfer to
  unseen materials and environments merely from the architectural constraints.

## Inputs and outputs

The high-level estimator receives a two-dimensional, finite, time-invariant
context matrix plus aligned vectors for query time, target, magnitude prior,
and timescale prior. Time and targets are non-negative; the two anchors are
positive. Concrete-specific units are documented in
[`docs/data-contract.md`](docs/data-contract.md).

`predict` returns point predictions. `predict_details` additionally returns:

| Key | Interpretation |
| --- | --- |
| `prediction` | point prediction |
| `magnitude` | corrected upper response scale |
| `timescale` | corrected characteristic time |
| `alpha` | rational-power exponent |
| `weights` | four non-negative normalized candidate weights |
| `candidate_laws` | value of each normalized temporal basis |
| `evolution` | convex combination of the bases |
| correction keys | learned bounded departures from the supplied anchors |

These quantities are model internals with a useful computational
interpretation; they are not automatically identifiable physical properties.

## Training and evaluation

`PriorAnchoredRegressor.fit` scales context using training data, optimizes the
structured PyTorch model, and restores the checkpoint with the lowest
validation loss. If no validation data are supplied, a seeded random row split
is used as a software convenience.

For scientific temporal extrapolation, the default split is generally
insufficient. Keep repeated observations from the same physical condition
together, define the temporal cutoff or extrapolation domain externally, fit
preprocessing only on development data, and pass an explicit validation set.
Report group-level as well as record-level behavior where appropriate.

No benchmark accuracy is claimed in this reusable repository. Manuscript-level
metrics require the separately versioned dataset, split IDs, configurations,
seeds, checkpoints, and evaluation scripts.

## Structural guarantees

For a fixed context and valid inputs, each candidate law is non-negative,
non-decreasing, and bounded by one. Convex weights and a positive corrected
magnitude therefore make the point prediction non-negative, non-decreasing,
and bounded by that magnitude. The guarantee is lost if time-varying values,
including query time, are inserted into context.

The guarantee concerns the implemented point predictor. It does not establish
predictive accuracy, causal validity, uncertainty coverage, or equivalent
properties for separately calculated interval endpoints.

## Known limitations and risks

- empirical priors inherit the assumptions and domain limits of their source
  formulations and include research-code simplifications;
- absolute parameter bounds are numerical/modeling choices, not universal
  material limits;
- distribution shift, sparse late-age observations, data leakage, and an
  inappropriate row-wise split can produce misleading performance estimates;
- correlated database records can overstate effective sample size;
- model checkpoints use PyTorch serialization and must be loaded only from
  trusted sources;
- the package does not impute missing data, infer units, encode categorical
  variables, or detect that context changes along a nominal trajectory.

## Reproducibility and provenance

- software version: `0.1.1` development line;
- license: MIT;
- authors and software citation metadata: [`CITATION.cff`](CITATION.cff);
- empirical lineage and implementation boundary:
  [`docs/empirical-priors.md`](docs/empirical-priors.md);
- reusable-code versus manuscript-code scope:
  [`docs/research-code-map.md`](docs/research-code-map.md).

When publishing results, archive the installed package version or commit,
dataset and preprocessing version, group/split identifiers, configuration,
random seeds, hardware and dependency information, checkpoint, and exact
evaluation script.

