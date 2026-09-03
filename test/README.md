# Tests

Run tests from the repository root after completing the setup in the
[main README](../README.md).

Install the test runner and coverage reporter:

```bash
python -m pip install pytest coverage
```

Run the complete test suite:

```bash
python -m pytest -q
```

Measure branch coverage for the tracked application code (the local
`helpers/` directory is intentionally excluded):

```bash
python -m coverage erase
python -m coverage run --branch --source=src -m pytest -q
python -m coverage report -m
```

Run one test file or one test method:

```bash
python -m pytest -q test/test_pdb_cache.py
python -m pytest -q \
  test/test_pdb_cache.py::SafePDBCacheTests::test_fresh_metadata_avoids_another_remote_request
```

The tests use temporary files and mocked API responses; they do not modify the
datasets or figures in the repository. STRIDE bootstrap tests also mock Git and
GNU Make, so the test suite never downloads or compiles STRIDE.

## License

See the root [`LICENSE`](../LICENSE) file for the applicable terms.
