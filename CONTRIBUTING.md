# Contributing

Thank you for helping improve PEA-PGNN.

1. Read and follow the [code of conduct](CODE_OF_CONDUCT.md).
2. Create a focused branch from the default branch.
3. Install development dependencies with `python -m pip install -e ".[dev]"`.
4. Add or update tests for behavioral changes.
5. Run `python -m ruff check .`, `python -m pytest`, `python -m build`, and
   `python -m twine check dist/*` when packaging or README metadata changes.
6. Describe the scientific assumption, source formulation, unit convention,
   applicable domain, and compatibility impact of any change to priors or
   temporal laws.

Do not commit restricted datasets, manuscript review correspondence, private
paths, credentials, or trained artifacts whose redistribution rights are
unclear.
