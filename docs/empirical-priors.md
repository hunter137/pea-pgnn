# Concrete empirical priors: provenance and implementation boundary

This document records where the concrete drying-shrinkage utilities in
`pea_pgnn.concrete` come from, what the package actually computes, and where
the implementation deliberately departs from a direct design-code
calculation.

## Short version

The functions are **research-oriented, formulation-inspired utilities**. They
retain characteristic magnitude, humidity, size, strength, and time terms from
established shrinkage formulations, but they also contain database-specific
choices, unit conversions, numerical clipping, and shared-parameter
modifications used by PEA-PGNN. They must not be presented as certified or
clause-complete implementations of ACI 209R-92, Model B3, or GL2000.

Before using these functions for design, compliance, safety assessment, or a
new material domain, consult the controlling standard or paper and validate the
implementation independently.

## Input and output convention

| Quantity | Package convention |
| --- | --- |
| `time` | duration since start of drying, day |
| `loading_age` | age at start of drying, day |
| `relative_humidity` | percent in `[0, 100]` |
| `volume_surface_ratio` | `V/S`, millimetre |
| `water_content` | kg/m³ |
| `compressive_strength` | MPa |
| returned shrinkage | non-negative magnitude, microstrain |

The package predicts the **magnitude** of drying shrinkage. A tensile/compressive
strain sign convention must be applied outside these utilities.

## Implemented utilities

### Model B3-inspired anchor

`b3_timescale` evaluates the characteristic-time expression used in the
research code,

```text
tau = max(0.085 * t0^(-0.08) * fcm^(-0.25) * (2 V/S)^2, 1 day).
```

`b3_ultimate_shrinkage` combines the water-content and strength magnitude term,
the relative-humidity term, and the working-code age correction. The resulting
microstrain magnitude is clipped to `[0, 3000]`. `b3_shrinkage` multiplies that
magnitude by `tanh(sqrt(t / tau))`.

The scientific source is the Model B3 formulation by Bažant and Baweja. The
public implementation uses a unit-consistent subset and the assumptions
recorded above; it is not a complete B3 creep-and-shrinkage implementation.

### GL2000-inspired anchor

`gl2000_ultimate_shrinkage` retains strength and fourth-power humidity terms
from the GL2000 working implementation. `gl2000_shrinkage` applies the
size-dependent development factor

```text
sqrt(t / (t + 0.15 * (V/S)^2)).
```

The scientific source is the Gardner--Lockman design provision for
normal-strength concrete. The function name identifies its lineage, not full
conformance with every scope condition or provision in the source.

### ACI 209-inspired anchor

`aci209_ultimate_shrinkage` applies the working implementation's piecewise
humidity factor and exponential `V/S` correction. `aci209_shrinkage` uses the
development factor

```text
t / (35 days + t).
```

The source is ACI 209R-92. Cement, curing, slump, fine-aggregate, and other
correction factors that may be needed in a complete ACI calculation are not
exposed as a comprehensive design-code interface here.

## Composite PEA-PGNN anchors

`concrete_prior_anchors` returns the following mapping:

| Key | Meaning |
| --- | --- |
| `magnitude` | arithmetic mean of B3-, GL2000-, and ACI209-inspired ultimate estimates |
| `timescale` | B3-inspired characteristic time, clipped to `[1, 5000]` days |
| `b3_magnitude` | B3-inspired component, microstrain |
| `gl2000_magnitude` | GL2000-inspired component, microstrain |
| `aci209_magnitude` | ACI209-inspired component, microstrain |

The equal-weight mean is an implementation choice. It does not claim that the
three estimates are unbiased, statistically independent, or equally valid for
every concrete. In the neural predictor these values are **anchors**, not fixed
answers: bounded, context-dependent corrections are learned during fitting.

## Candidate-law naming

The temporal model uses four normalized bases named `tanh_sqrt`,
`rational_power`, `sqrt_rational`, and `logarithmic`. The first three retain
characteristic patterns motivated by B3, ACI 209, and GL2000, but share a
corrected condition-dependent timescale; the rational-power basis also has an
adaptive exponent. For this reason, the manuscript and package describe them
as B3-, ACI-, and GL2000-**type** laws. The logarithmic law is a bounded
computational candidate, not another empirical design formulation.

## Applicability and validation checklist

Before relying on an empirical utility in a new study:

1. confirm every input unit and the start-of-drying time convention;
2. check that material strength, humidity, geometry, curing, and age lie within
   the source formulation's intended domain;
3. compare several hand-calculated cases against the controlling source;
4. record any National Annex, cement type, curing, shape-factor, or calibration
   choice that the compact API does not represent;
5. validate against data not used to select or calibrate the implementation;
6. preserve the `-inspired` or `-type` wording unless full conformance has been
   independently established.

## Primary sources

- ACI Committee 209. *ACI 209R-92: Prediction of Creep, Shrinkage, and
  Temperature Effects in Concrete Structures* (1992; reapproved 2008).
  [ACI document page](https://www.concrete.org/store/productdetail.aspx?Format=DOWNLOAD&ItemID=20992&Language=English&Units=US_AND_METRIC)
- Z. P. Bažant and S. Baweja. “Creep and shrinkage prediction model for
  analysis and design of concrete structures—Model B3.” *Materials and
  Structures* 28, 357–365 (1995).
  [doi:10.1007/BF02473152](https://doi.org/10.1007/BF02473152)
- N. J. Gardner and M. J. Lockman. “Design Provisions for Drying Shrinkage and
  Creep of Normal-Strength Concrete.” *ACI Materials Journal* 98(2), 159–167
  (2001). [doi:10.14359/10199](https://doi.org/10.14359/10199)
These references establish the formulation lineage. The exact public code is
the executable specification for this package release; any claim of direct
standard conformance requires a separate clause-by-clause verification.

