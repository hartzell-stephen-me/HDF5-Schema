# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-29

### Added
- Initial release of hdf5schema package
- Core validation engine with comprehensive HDF5 schema validation
- Support for JSON Schema features:
  - Boolean logic (`anyOf`, `allOf`, `oneOf`, `not`)
  - Conditional validation (`if/then/else`)
  - Dependencies (`dependentRequired`, `dependentSchemas`)
  - Data constraints (`enum`, `const`)
  - String validation (format, pattern, length constraints)
- Schema generation from existing HDF5 files
- Command-line interfaces for validation and schema generation
- Comprehensive test suite 
- Full documentation with README
- Support for Python 3.9-3.13
- MIT License

### Schema Features
- Group and dataset validation
- Pattern-based member matching with regex
- Data type validation (simple and compound dtypes)
- Shape validation with variable dimensions
- Attribute validation
- Required/optional members
- JSON Schema `$ref` references with cycle detection

### Documentation
- Comprehensive README with usage examples
- NumPy-style docstrings throughout codebase
- CITATION.cff for academic citation
- CLAUDE.md for development guidelines

[1.0.0]: https://github.com/hartzell-stephen-me/hdf5schema/releases/tag/1.0.0
