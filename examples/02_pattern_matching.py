"""
Pattern Matching Example
=========================

This example demonstrates schema validation using pattern matching for
datasets with dynamic names following specific naming conventions.
"""
import gc
import h5py
import json
import re
import tempfile
from pathlib import Path

from hdf5schema.validator import Hdf5Validator


def create_multi_channel_data(filepath: Path) -> None:
    """Create HDF5 file with multiple channels following a naming pattern."""
    with h5py.File(filepath, 'w') as f:
        f.attrs['recording_id'] = 'REC-2025-001'
        f.attrs['num_channels'] = 8

        # Create channels with pattern: channel_01, channel_02, etc.
        for i in range(1, 9):
            channel_name = f'channel_{i:02d}'
            dataset = f.create_dataset(
                channel_name,
                data=[i * 10 + j for j in range(100)],
                dtype='int32'
            )
            dataset.attrs['channel_number'] = i
            dataset.attrs['gain'] = 1.0 + (i * 0.1)
            dataset.attrs['enabled'] = True if i <= 6 else False


def create_pattern_schema() -> dict:
    """
    Create a schema using pattern matching.

    This schema validates that:
    - Any member matching 'channel_\\d{2}' pattern must be an int32 dataset
    - Each channel has required attributes
    - Channels have exactly 100 samples
    """
    return {
        "type": "group",
        "attrs": [
            {"name": "recording_id", "dtype": "U128"},
            {"name": "num_channels", "dtype": "<i8"}
        ],
        "patternMembers": {
            r"^channel_\d{2}$": {
                "type": "dataset",
                "dtype": "<i4",
                "shape": [100],
                "attrs": [
                    {"name": "channel_number", "dtype": "<i8"},
                    {"name": "gain", "dtype": "<f8"},
                    {"name": "enabled", "dtype": "bool"}
                ]
            }
        }
    }


def create_invalid_data(filepath: Path) -> None:
    """Create HDF5 file that violates the pattern schema."""
    with h5py.File(filepath, 'w') as f:
        f.attrs['recording_id'] = 'REC-2025-002'
        f.attrs['num_channels'] = 2

        # Valid channel
        dataset1 = f.create_dataset('channel_01', data=range(100), dtype='int32')
        dataset1.attrs['channel_number'] = 1
        dataset1.attrs['gain'] = 1.1
        dataset1.attrs['enabled'] = True

        # Invalid: wrong shape (should be 100, but is 50)
        dataset2 = f.create_dataset('channel_02', data=range(50), dtype='int32')
        dataset2.attrs['channel_number'] = 2
        dataset2.attrs['gain'] = 1.2
        dataset2.attrs['enabled'] = True

        # Invalid: wrong dtype (float instead of int32)
        dataset3 = f.create_dataset('channel_03', data=[i * 1.5 for i in range(100)], dtype='float64')
        dataset3.attrs['channel_number'] = 3
        dataset3.attrs['gain'] = 1.3
        dataset3.attrs['enabled'] = False


def main():
    """Run the pattern matching example."""
    print("=" * 60)
    print("Pattern Matching Schema Validation Example")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with valid data
        print("\n1. Validating VALID multi-channel data...")
        valid_file = Path(tmpdir) / "valid_channels.h5"
        create_multi_channel_data(valid_file)

        schema = create_pattern_schema()
        validator = Hdf5Validator(valid_file, schema)
        errors = list(validator.iter_errors())

        if errors:
            print(f"   [FAIL] Found {len(errors)} errors:")
            for error in errors:
                print(f"     - {error}")
        else:
            print("   [PASS] Validation passed!")

        # Show what was validated
        with h5py.File(valid_file, 'r') as f:
            channel_names = [name for name in f.keys() if re.match(r'^channel_\d{2}$', name)]
            print(f"   Validated {len(channel_names)} channels: {', '.join(channel_names)}")

        # Test with invalid data
        print("\n2. Validating INVALID multi-channel data...")
        invalid_file = Path(tmpdir) / "invalid_channels.h5"
        create_invalid_data(invalid_file)

        validator = Hdf5Validator(invalid_file, schema)
        errors = list(validator.iter_errors())

        if errors:
            print(f"   [PASS] Correctly detected {len(errors)} errors:")
            for i, error in enumerate(errors, 1):
                print(f"     {i}. {error}")
        else:
            print("   [FAIL] No errors found (unexpected!)")

        # Demonstrate pattern specificity
        print("\n3. Pattern Matching Details:")
        print("   Schema pattern: r'^channel_\\d{2}$'")
        print("   Matches:")
        print("     [PASS] 'channel_01' - valid format")
        print("     [PASS] 'channel_99' - valid format")
        print("     [FAIL] 'channel_1'  - missing leading zero")
        print("     [FAIL] 'channel_001' - too many digits")
        print("     [FAIL] 'data_01'    - wrong prefix")

        # Explicitly delete validators to close HDF5 files (Windows compatibility)
        del validator
        # Force garbage collection to ensure HDF5 files are closed
        gc.collect()

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
