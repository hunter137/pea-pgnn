# Adapting PEA-PGNN to a new engineering domain

PEA-PGNN is a temporal model, not a universal tabular regressor. A new
application is a good structural match only when its response is expected to
be non-negative, non-decreasing with time, and bounded for a fixed condition.
The model then represents the response as a corrected magnitude multiplied by
a convex mixture of normalized time laws.

## Suitability test

Before training, answer all of the following questions.

1. Is the target a time-dependent response rather than only one fixed-age
   value?
2. For a fixed condition, should its latent mean trajectory be non-negative
   and non-decreasing over the modeled time domain?
3. Is a finite long-term response scale scientifically meaningful?
4. Can domain knowledge supply a positive response-scale anchor and a positive
   characteristic-time anchor without using forbidden future observations?
5. Can repeated observations from one physical condition be assigned a stable
   group identifier?

If the answer to questions 1--3 is no, the current architecture should not be
forced onto the problem. Static fixed-age prediction needs a tabular model;
growth followed by deterioration needs candidate laws that can represent that
change of direction.

## Map the domain to the five input arrays

| Generic input | Domain interpretation |
| --- | --- |
| `context` | finite, time-invariant descriptors of one condition |
| `time` | non-negative query time or a justified equivalent-time variable |
| `target` | observed non-negative response |
| `magnitude_prior` | positive prior for the long-term response scale |
| `timescale_prior` | positive prior for the response-development timescale |

For every repeated trajectory, `context` and both prior anchors must remain
constant while `time` changes. Do not copy query time into `context`; doing so
allows magnitude, timescale, and mixture weights to change between queries and
invalidates the trajectory-level monotonicity argument.

PEA-PGNN scales numeric context internally. It does not impute missing values,
infer units, or encode categories. Those choices belong in an explicit,
versioned application pipeline.

## Construct anchors without leakage

An anchor can come from a published empirical formulation, a mechanistic
approximation, a design estimate, or information genuinely available at the
prediction cutoff. Record its equation, units, parameter source, calibration
data, domain limits, and any simplification.

Do not construct an anchor using the target that the model is being evaluated
on. For example, a measured 28-day value may be a legitimate input to a task
that predicts 90--365 days, but not to a nominal blind prediction of 28-day
response. A learned anchor model must be fitted inside the training data or
cross-fitted; fitting it once on the complete dataset leaks test information.

An uninformative positive constant can make the software run, but it no longer
supports a strong claim of knowledge-guided prediction. Compare against the
anchor alone so that the neural correction demonstrates measurable value.

## Set domain-specific bounds

The default `ModelConfig` values were selected for the first drying-shrinkage
application and are not unit-free. A new domain must configure:

- `magnitude_bounds` in target units;
- `timescale_bounds` in time units;
- `additive_magnitude_scale` in target units;
- relative correction bounds that reflect the expected reliability of the
  two anchors; and
- network width appropriate to the number of independent physical conditions,
  not merely the number of repeated rows.

Bounds are modeling assumptions. Report sensitivity to reasonable alternatives
and check how frequently fitted parameters approach them.

## Split by condition before fitting

Rows from one physical condition must not be scattered across ordinary train,
validation, and test sets when the goal is generalization to unseen
conditions. Use the public helper:

```python
from pea_pgnn import grouped_train_validation_test_split

split = grouped_train_validation_test_split(
    groups=condition_ids,
    validation_fraction=0.2,
    test_fraction=0.2,
    seed=42,
)
```

The returned indices are mutually group-disjoint. Whole-group assignment means
that realized row fractions can differ from the requested fractions when group
sizes are unequal.

A defensible study often needs more than one protocol:

- unseen-condition testing, with condition IDs disjoint across sets;
- temporal extrapolation, with a declared observation cutoff;
- external validation on another laboratory, site, material source, or
  database; and
- repeated seeds or grouped folds to quantify split sensitivity.

## Minimum evaluation package

Report the empirical prior by itself, an unconstrained data-driven baseline,
the complete PEA-PGNN, and ablations of each knowledge component. Include
record-level metrics only alongside condition-level summaries and trajectory
plots. Audit non-negativity, monotonicity, and boundedness on a dense time grid,
and report the number of violations rather than showing only selected curves.

Structural guarantees do not imply accuracy, calibration, causal validity, or
transfer outside the evaluated domain. Archive the feature schema, unit table,
group identifiers, split indices, anchor implementation, configurations,
seeds, dependency versions, checkpoints, and evaluation code.

The concrete-strength walkthrough applies this workflow to age-dependent
compressive-strength development in
[`docs/concrete-strength-development.md`](concrete-strength-development.md).
