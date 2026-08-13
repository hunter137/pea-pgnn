# Changelog

All notable changes to this project will be documented here.

## [0.2.0] - Unreleased

### Added

- a domain-adaptation guide with an explicit suitability test, input mapping,
  prior-leakage rules, configuration guidance, and minimum evaluation package;
- a concrete compressive-strength development walkthrough and runnable
  synthetic example;
- `ModelConfig.for_concrete_strength()` as an overrideable MPa- and day-scale
  starting configuration; and
- `GroupedSplit` and `grouped_train_validation_test_split()` for reproducible,
  group-disjoint train, validation, and test partitions.

### Changed

- kept the `main` branch on the `0.2.0.dev0` development line after the stable
  `0.1.2` maintenance release; and
- limited the concrete empirical utilities to the B3-, GL2000-, and
  ACI209-inspired formulations actually used by the implementation.

## [0.1.2] - 2026-08-13

### Removed

- an unused comparison utility that was not part of the paper implementation;
  the concrete prior anchor continues to use only Model B3, ACI 209, and
  GL2000, while the four candidate temporal laws remain unchanged.

## [0.1.1] - 2026-08-12

### Added

- empirical-prior provenance, units, implementation choices, and primary
  source links;
- a model card and security policy;
- bug-report and feature-request forms, a pull-request template, code of
  conduct, and Dependabot configuration;
- a Trusted Publishing release workflow for TestPyPI and PyPI; and
- stable concrete implementation cases, metrics tests,
  invalid-input tests, a coverage threshold, and NumPy/PyTorch parity tests.

### Changed

- expanded the README with feature highlights, requirements, documentation,
  support, author, and funding information;
- made README repository links work from both GitHub and PyPI;
- standardized author metadata for Deyu Liang, Jinlong Liu, and Lei Xu;
- expanded README guidance for empirical outputs, `predict_details`, runnable
  examples, and scientifically defensible temporal validation;
- centralized the package version and aligned development metadata at `0.1.1`;
- updated the GitHub Actions dependencies used by CI and release workflows; and
- updated the public-release checklist to distinguish completed repository work
  from external publication configuration.

## [0.1.0] - 2026-08-12

### Added

- reusable prior-anchored structured PyTorch model;
- NumPy-facing training and checkpoint wrapper;
- four normalized candidate temporal laws and convex-mixture utility;
- concrete drying-shrinkage empirical prior utilities;
- trajectory-constraint auditing and regression metrics;
- tests, a synthetic example, documentation, and GitHub Actions CI;
- MIT open-source license; and
- GitHub project badges and software citation metadata.
