# Tests

Run tests from the repository root after completing the setup in the
[main README](../README.md).

Install the test runner:

```bash
python -m pip install pytest
```

Run the complete test suite:

```bash
python -m pytest -q
```

Run one test file or one test method:

```bash
python -m pytest -q test/test_pdb_cache.py
python -m pytest -q \
  test/test_pdb_cache.py::SafePDBCacheTests::test_fresh_metadata_avoids_another_remote_request
```

The tests use temporary files and mocked API responses; they do not modify the
datasets or figures in the repository.

## License

See the root [`LICENSE`](../LICENSE) file for the applicable terms.
