# hdf5schema Test Suite

This directory contains the comprehensive test suite for the hdf5schema package.

## Test Structure

The test suite is organized into focused test modules:

### Unit Tests

- [test_validator.py](test_validator.py) (1,129 lines) - Core validator functionality
  - Dataset validation (shapes, dtypes, attributes)
  - Group validation
  - Complex hierarchical structures
  - Pattern matching

- [test_boolean_expressions.py](test_boolean_expressions.py) (707 lines) - Boolean logic operators
  - `anyOf` - at least one schema must match
  - `allOf` - all schemas must match
  - `oneOf` - exactly one schema must match
  - `not` - schema must not match

- [test_group_schema.py](test_group_schema.py) (580 lines) - Group schema validation
  - Member validation
  - Pattern members
  - Required members

- [test_dependent.py](test_dependent.py) (549 lines) - Dependency features
  - `dependentRequired` - conditional required fields
  - `dependentSchemas` - conditional schema application

- [test_dataset_schema.py](test_dataset_schema.py) (506 lines) - Dataset schema validation
  - Dtype validation (simple and compound)
  - Shape validation
  - Attribute validation

- [test_conditonals.py](test_conditonals.py) (427 lines) - Conditional validation
  - `if/then/else` patterns
  - Nested conditionals
  - Complex conditional logic

- [test_format_validation.py](test_format_validation.py) (283 lines) - String format validation
  - Format validators (email, uri, date-time, uuid, etc.)
  - Length constraints
  - Pattern matching

### Integration Tests

- [test_integration.py](test_integration.py) - End-to-end workflow tests
  - Complete validation workflows
  - Schema generation round-trips
  - Pattern matching workflows
  - Conditional validation scenarios
  - Boolean logic workflows
  - Real-world scientific data scenarios

## Running the Tests

### Run All Tests

Using unittest (built-in):
```bash
C:\envs\hdf5schema\python.exe -m unittest discover tests
```

### Run Specific Test Module

```bash
C:\envs\hdf5schema\python.exe -m unittest tests.test_integration
```

### Run Specific Test Class

```bash
C:\envs\hdf5schema\python.exe -m unittest tests.test_integration.TestBasicWorkflow
```

### Run Specific Test Method

```bash
C:\envs\hdf5schema\python.exe -m unittest tests.test_integration.TestBasicWorkflow.test_sensor_data_workflow
```

### Run with Coverage

Install coverage:
```bash
pip install coverage
```

Run tests with coverage:
```bash
coverage run -m unittest discover tests
coverage report
coverage html
```

View HTML coverage report:
```bash
# Opens htmlcov/index.html in your browser
start htmlcov\index.html
```

## Test Data

Tests create temporary HDF5 files during execution:
- Test data is created in `tests/data/` (gitignored)
- Most tests use `tempfile.TemporaryDirectory()` for automatic cleanup
- No persistent test data is committed to the repository

## Writing New Tests

### Test Template

```python
import unittest
import tempfile
from pathlib import Path
import h5py
from hdf5schema.validator import Hdf5Validator


class TestMyFeature(unittest.TestCase):
    """Test description."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmppath = Path(self.tmpdir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.tmpdir.cleanup()

    def test_my_feature(self):
        """Test specific behavior."""
        # Create test HDF5 file
        hdf5_file = self.tmppath / "test.h5"
        with h5py.File(hdf5_file, 'w') as f:
            f.create_dataset('data', data=[1, 2, 3])

        # Create schema
        schema = {
            "type": "group",
            "members": {
                "data": {"type": "dataset", "dtype": "<i8"}
            }
        }

        # Validate
        validator = Hdf5Validator(hdf5_file, schema)
        errors = list(validator.iter_errors())

        # Assert
        self.assertEqual(len(errors), 0)


if __name__ == '__main__':
    unittest.main()
```

### Best Practices

1. **Use descriptive test names** - Test method names should describe what is being tested
2. **One assertion per test** - Keep tests focused on a single behavior
3. **Use setUp/tearDown** - Clean up temporary files properly
4. **Test both success and failure** - Verify both valid and invalid inputs
5. **Document test purpose** - Add docstrings explaining what the test validates

## Test Coverage Goals

- **Core validation**: >95% coverage
- **Schema classes**: >90% coverage
- **CLI interfaces**: >80% coverage
- **Overall package**: >90% coverage

## Continuous Integration

Tests are automatically run on:
- Every push to the repository
- Every pull request
- Nightly builds (if configured)

See `.github/workflows/` for CI/CD configuration.

## Troubleshooting

### Tests Fail on Import

Ensure hdf5schema is installed:
```bash
pip install -e .
```

### Permission Errors

On Windows, ensure HDF5 files are properly closed:
```python
with h5py.File(filepath, 'w') as f:
    # ... create data ...
    pass  # File is automatically closed
```

### Memory Issues with Large Tests

For tests creating large HDF5 files, consider:
- Using smaller test data
- Running tests individually
- Increasing available memory

## Additional Resources

- [unittest documentation](https://docs.python.org/3/library/unittest.html)
- [h5py testing guide](https://docs.h5py.org/en/stable/build.html#testing)
- [coverage.py documentation](https://coverage.readthedocs.io/)
