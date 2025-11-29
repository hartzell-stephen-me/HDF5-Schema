# HDF5 Schema Examples

This directory contains practical examples demonstrating various features of the `hdf5schema` package.

## Running the Examples

All examples are standalone Python scripts that can be run directly:

```bash
# Using the conda environment Python
C:\envs\hdf5schema\python.exe 01_basic_validation.py

# Or if hdf5schema is installed in your current environment
python 01_basic_validation.py
```

## Available Examples

### 01_basic_validation.py
**Difficulty:** Beginner
**Topics:** Basic validation, simple schemas, Python API usage

Demonstrates the fundamentals of HDF5 schema validation:
- Creating HDF5 files with groups, datasets, and attributes
- Defining basic JSON schemas
- Using the `validate()` function
- Using `Hdf5Validator` for detailed error reporting
- Understanding validation errors

**Use this example if you're:** Getting started with hdf5schema

---

### 02_pattern_matching.py
**Difficulty:** Intermediate
**Topics:** Pattern matching, regex, dynamic member names

Shows how to validate HDF5 files with dynamically named members:
- Using regex patterns to match dataset names
- `patternMembers` schema definition
- Validating multiple channels/sensors with consistent structure
- Pattern specificity and matching rules

**Use this example if you're:** Working with files that have variable member names following naming conventions

---

### 03_conditional_validation.py
**Difficulty:** Intermediate
**Topics:** Conditional validation, if/then/else

Demonstrates conditional schema validation:
- Using `if/then/else` constructs
- Applying different validation rules based on attributes
- Validating RGB vs grayscale image data
- Creating flexible schemas for multi-format data

**Use this example if you're:** Validating files with different formats based on metadata

---

### 04_schema_generation.py
**Difficulty:** Beginner
**Topics:** Schema generation, documentation, reverse engineering

Shows how to automatically generate schemas from existing HDF5 files:
- Using `generate_schema()` function
- Generating schemas for specific group paths
- Understanding generated schema limitations
- Refining auto-generated schemas

**Use this example if you're:**
- Creating schemas from existing reference files
- Documenting legacy HDF5 file structures
- Getting started with a schema template

---

### 05_advanced_boolean_logic.py
**Difficulty:** Advanced
**Topics:** Boolean logic, anyOf, allOf, oneOf, not

Demonstrates advanced boolean logic operators:
- `anyOf` - validating when at least one schema matches
- `allOf` - validating when all schemas match
- `oneOf` - validating when exactly one schema matches
- `not` - validating when a schema does NOT match
- Combining operators for complex validation rules

**Use this example if you're:**
- Creating complex validation rules
- Supporting multiple valid file formats
- Enforcing mutually exclusive constraints

---

## Example Dependencies

All examples require:
- `h5py` - HDF5 file I/O
- `numpy` - Array operations
- `hdf5schema` - Schema validation

These are automatically installed when you install `hdf5schema`:

```bash
pip install hdf5schema
```

## Example Output

Each example includes:
- ✓/✗ status indicators for validation results
- Detailed error messages when validation fails
- Explanations of the schema features being demonstrated
- File structure visualization

## Common Patterns

### Creating Test Data

Most examples use this pattern:
```python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    hdf5_file = Path(tmpdir) / "data.h5"
    # Create and test files...
```

### Validation Pattern

```python
from hdf5schema.validator import Hdf5Validator

validator = Hdf5Validator(hdf5_file, schema)
errors = list(validator.iter_errors())

if errors:
    print(f"Found {len(errors)} errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Validation passed!")
```

### Quick Validation

```python
from hdf5schema import validate

try:
    is_valid = validate(hdf5_file, schema)
    print("Valid!")
except ValidationError as e:
    print(f"Invalid: {e}")
```

## Next Steps

After exploring these examples:

1. **Read the main README** - Comprehensive documentation of all features
2. **Check the test suite** - See `tests/` for more advanced examples
3. **Try the CLI tools** - Use `hdf5schema-validate` and `hdf5schema-generate`
4. **Build your own schemas** - Apply these patterns to your data

## Getting Help

- **Documentation:** See the main [README.md](../README.md)
- **Issues:** Report problems at https://github.com/hartzell-stephen-me/hdf5schema/issues
- **Examples not working?** Ensure you have the latest version: `pip install --upgrade hdf5schema`
