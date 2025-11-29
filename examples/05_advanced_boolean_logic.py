"""
Advanced Boolean Logic Example
===============================

This example demonstrates advanced schema validation using boolean logic
operators: anyOf, allOf, oneOf, and not.
"""
import gc
import h5py
import numpy as np
import tempfile
from pathlib import Path

from hdf5schema.validator import Hdf5Validator


def create_measurement_data(filepath: Path, config: str) -> None:
    """
    Create HDF5 file with measurement data.

    Parameters
    ----------
    filepath : Path
        Output HDF5 file path
    config : str
        Configuration: 'basic', 'advanced', or 'expert'
    """
    with h5py.File(filepath, 'w') as f:
        f.attrs['config'] = config
        f.attrs['timestamp'] = '2025-01-29T12:00:00'

        # All configs have raw data
        f.create_dataset('raw_data', data=np.random.randn(1000), dtype='float64')

        if config in ['advanced', 'expert']:
            # Advanced: also has filtered data
            f.create_dataset('filtered_data', data=np.random.randn(1000), dtype='float64')

        if config == 'expert':
            # Expert: also has analysis results
            analysis = f.create_group('analysis')
            analysis.create_dataset('mean', data=0.0, dtype='float64')
            analysis.create_dataset('std', data=1.0, dtype='float64')
            analysis.create_dataset('peaks', data=np.array([100, 250, 500, 750]), dtype='int64')


def create_anyof_schema() -> dict:
    """
    Schema using anyOf: data must match AT LEAST ONE of the schemas.

    Valid if the file has EITHER:
    - A 'raw_data' dataset, OR
    - A 'filtered_data' dataset, OR
    - Both
    """
    return {
        "type": "group",
        "attrs": [
            {"name": "config", "dtype": "U128"},
            {"name": "timestamp", "dtype": "U128"}
        ],
        "anyOf": [
            {
                "members": {
                    "raw_data": {
                        "type": "dataset",
                        "dtype": "<f8"
                    }
                },
                "required": ["raw_data"]
            },
            {
                "members": {
                    "filtered_data": {
                        "type": "dataset",
                        "dtype": "<f8"
                    }
                },
                "required": ["filtered_data"]
            },
            {
                "members": {
                    "raw_data": {
                        "type": "dataset",
                        "dtype": "<f8"
                    },
                    "filtered_data": {
                        "type": "dataset",
                        "dtype": "<f8"
                    }
                },
                "required": ["raw_data", "filtered_data"]
            }
        ]
    }


def create_allof_schema() -> dict:
    """
    Schema using allOf: data must match ALL of the schemas.

    Valid if the file has:
    - A 'config' attribute, AND
    - A 'timestamp' attribute, AND
    - A 'raw_data' dataset
    """
    return {
        "type": "group",
        "allOf": [
            {
                "attrs": [
                    {"name": "config", "dtype": "U128"}
                ]
            },
            {
                "attrs": [
                    {"name": "timestamp", "dtype": "U128"}
                ]
            },
            {
                "members": {
                    "raw_data": {
                        "type": "dataset",
                        "dtype": "<f8"
                    }
                }
            }
        ]
    }


def create_oneof_schema() -> dict:
    """
    Schema using oneOf: data must match EXACTLY ONE of the schemas.

    Valid if the file is EITHER:
    - Basic config (only raw_data), OR
    - Advanced config (raw_data + filtered_data, no analysis), OR
    - Expert config (all data including analysis)

    But NOT a mix that doesn't match any single schema.
    """
    return {
        "type": "group",
        "attrs": [
            {"name": "config", "dtype": "U128"},
            {"name": "timestamp", "dtype": "U128"}
        ],
        "oneOf": [
            {
                # Basic: only raw_data
                "members": {
                    "raw_data": {"type": "dataset", "dtype": "<f8"}
                },
                "required": ["raw_data"],
                "not": {
                    "members": {
                        "filtered_data": {"type": "dataset"}
                    }
                }
            },
            {
                # Advanced: raw_data + filtered_data, no analysis
                "members": {
                    "raw_data": {"type": "dataset", "dtype": "<f8"},
                    "filtered_data": {"type": "dataset", "dtype": "<f8"}
                },
                "required": ["raw_data", "filtered_data"],
                "not": {
                    "members": {
                        "analysis": {"type": "group"}
                    }
                }
            },
            {
                # Expert: everything including analysis
                "members": {
                    "raw_data": {"type": "dataset", "dtype": "<f8"},
                    "filtered_data": {"type": "dataset", "dtype": "<f8"},
                    "analysis": {
                        "type": "group",
                        "members": {
                            "mean": {"type": "dataset", "dtype": "<f8"},
                            "std": {"type": "dataset", "dtype": "<f8"},
                            "peaks": {"type": "dataset", "dtype": "<i8"}
                        },
                        "required": ["mean", "std", "peaks"]
                    }
                },
                "required": ["raw_data", "filtered_data", "analysis"]
            }
        ]
    }


def main():
    """Run the advanced boolean logic example."""
    print("=" * 60)
    print("Advanced Boolean Logic Schema Validation Example")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: anyOf validation
        print("\n1. Testing anyOf (at least one must match)...")
        print("   anyOf allows files with raw_data OR filtered_data OR both")

        anyof_schema = create_anyof_schema()

        # Should pass: has raw_data
        file1 = Path(tmpdir) / "basic.h5"
        create_measurement_data(file1, 'basic')
        validator = Hdf5Validator(file1, anyof_schema)
        errors = list(validator.iter_errors())
        print(f"   Basic config (only raw_data): {'[PASS] PASS' if not errors else '[FAIL] FAIL'}")
        del validator

        # Should pass: has both
        file2 = Path(tmpdir) / "advanced.h5"
        create_measurement_data(file2, 'advanced')
        validator = Hdf5Validator(file2, anyof_schema)
        errors = list(validator.iter_errors())
        print(f"   Advanced config (both datasets): {'[PASS] PASS' if not errors else '[FAIL] FAIL'}")
        del validator

        # Test 2: allOf validation
        print("\n2. Testing allOf (all must match)...")
        print("   allOf requires config attribute AND timestamp AND raw_data")

        allof_schema = create_allof_schema()

        file3 = Path(tmpdir) / "complete.h5"
        create_measurement_data(file3, 'expert')
        validator = Hdf5Validator(file3, allof_schema)
        errors = list(validator.iter_errors())
        print(f"   Expert config (all requirements): {'[PASS] PASS' if not errors else '[FAIL] FAIL'}")
        del validator

        # Test missing requirement
        with h5py.File(Path(tmpdir) / "incomplete.h5", 'w') as f:
            f.attrs['config'] = 'basic'
            # Missing timestamp and raw_data
        validator = Hdf5Validator(Path(tmpdir) / "incomplete.h5", allof_schema)
        errors = list(validator.iter_errors())
        print(f"   Incomplete file (missing timestamp & data): {'[PASS] CORRECTLY FAILED' if errors else '[FAIL] UNEXPECTED PASS'}")
        if errors:
            print(f"     Found {len(errors)} errors (as expected)")
        del validator

        # Test 3: oneOf validation
        print("\n3. Testing oneOf (exactly one must match)...")
        print("   oneOf requires EXACTLY one config type: basic, advanced, or expert")

        oneof_schema = create_oneof_schema()

        # Test each config
        for config in ['basic', 'advanced', 'expert']:
            filepath = Path(tmpdir) / f"{config}_oneof.h5"
            create_measurement_data(filepath, config)
            validator = Hdf5Validator(filepath, oneof_schema)
            errors = list(validator.iter_errors())
            print(f"   {config.capitalize()} config: {'[PASS] PASS' if not errors else '[FAIL] FAIL'}")
            if errors:
                for error in errors[:2]:
                    print(f"     - {error}")
            del validator

        # Test 4: Demonstrate 'not' operator
        print("\n4. Testing 'not' operator...")
        not_schema = {
            "type": "group",
            "not": {
                "members": {
                    "analysis": {"type": "group"}
                }
            }
        }

        file_without_analysis = Path(tmpdir) / "no_analysis.h5"
        create_measurement_data(file_without_analysis, 'basic')
        validator = Hdf5Validator(file_without_analysis, not_schema)
        errors = list(validator.iter_errors())
        print(f"   Basic config (no analysis group): {'[PASS] PASS' if not errors else '[FAIL] FAIL'}")
        del validator

        file_with_analysis = Path(tmpdir) / "with_analysis.h5"
        create_measurement_data(file_with_analysis, 'expert')
        validator = Hdf5Validator(file_with_analysis, not_schema)
        errors = list(validator.iter_errors())
        print(f"   Expert config (has analysis group): {'[PASS] CORRECTLY FAILED' if errors else '[FAIL] UNEXPECTED PASS'}")

        # Summary
        print("\n5. Boolean Logic Summary:")
        print("   - anyOf: Validates if AT LEAST ONE sub-schema matches")
        print("   - allOf: Validates if ALL sub-schemas match")
        print("   - oneOf: Validates if EXACTLY ONE sub-schema matches")
        print("   - not: Validates if the sub-schema does NOT match")

        # Explicitly delete validators to close HDF5 files (Windows compatibility)
        del validator
        # Force garbage collection to ensure HDF5 files are closed
        gc.collect()

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
