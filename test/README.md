# Tests

Run from the repository root in the environment configured in the
[main README](../README.md).

```bash
python -m pip install pytest coverage
python -m pytest -q
```

Run a single test file:

```bash
python -m pytest -q test/test_pdb_cache.py
```

Measure branch coverage:

```bash
python -m coverage erase
python -m coverage run --branch --source=src -m pytest -q
python -m coverage report -m
```

Tests use temporary files, mocked APIs and executable fixtures. STRIDE and TmDet
installation tests mock downloads and build commands. Patch dependencies in the
module that consumes them, rather than in the public entry-point module.
