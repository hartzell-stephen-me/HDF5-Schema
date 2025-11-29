"""
Basic HDF5 Schema Validation Example
=====================================

This example demonstrates basic schema validation for a simple HDF5 file
containing sensor measurement data.
"""
import gc
import h5py
import json
import tempfile
from pathlib import Path

from hdf5schema.validate import validate
from hdf5schema.validator import Hdf5Validator


def create_sensor_data_file(filepath: Path) -> None:
    """Create an HDF5 file with sensor measurement data."""
    with h5py.File(filepath, 'w') as f:
        # Root group attributes
        f.attrs['experiment_name'] = 'Temperature Monitoring'
        f.attrs['version'] = '1.0'

        # Create sensor data group
        sensor_group = f.create_group('sensors')
        sensor_group.attrs['num_sensors'] = 3

        # Temperature sensor data
        temp_data = sensor_group.create_dataset(
            'temperature',
            data=[20.5, 21.0, 20.8, 21.2, 20.9],
            dtype='float64'
        )
        temp_data.attrs['unit'] = 'celsius'
        temp_data.attrs['sensor_id'] = 'TEMP-001'

        # Humidity sensor data
        humid_data = sensor_group.create_dataset(
            'humidity',
            data=[45.2, 46.1, 45.8, 46.5, 45.9],
            dtype='float64'
        )
        humid_data.attrs['unit'] = 'percent'
        humid_data.attrs['sensor_id'] = 'HUM-001'


def create_schema() -> dict:
    """Create a schema for sensor data validation."""
    return {
        "type": "group",
        "attrs": [
            {"name": "experiment_name", "dtype": "U128"},
            {"name": "version", "dtype": "U128"}
        ],
        "members": {
            "sensors": {
                "type": "group",
                "attrs": [
                    {"name": "num_sensors", "dtype": "<i8"}
                ],
                "members": {
                    "temperature": {
                        "type": "dataset",
                        "dtype": "<f8",
                        "shape": [5],
                        "attrs": [
                            {"name": "unit", "dtype": "U128"},
                            {"name": "sensor_id", "dtype": "U128"}
                        ]
                    },
                    "humidity": {
                        "type": "dataset",
                        "dtype": "<f8",
                        "shape": [5],
                        "attrs": [
                            {"name": "unit", "dtype": "U128"},
                            {"name": "sensor_id", "dtype": "U128"}
                        ]
                    }
                },
                "required": ["temperature", "humidity"]
            }
        },
        "required": ["sensors"]
    }


def main():
    """Run the basic validation example."""
    print("=" * 60)
    print("Basic HDF5 Schema Validation Example")
    print("=" * 60)

    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_file = Path(tmpdir) / "sensor_data.h5"
        schema_file = Path(tmpdir) / "sensor_schema.json"

        # Create HDF5 file
        print("\n1. Creating HDF5 file with sensor data...")
        create_sensor_data_file(hdf5_file)
        print(f"   Created: {hdf5_file}")

        # Create schema
        print("\n2. Creating validation schema...")
        schema = create_schema()
        schema_file.write_text(json.dumps(schema, indent=2))
        print(f"   Created: {schema_file}")

        # Validate using Python API
        print("\n3. Validating using Python API...")
        try:
            is_valid = validate(hdf5_file, schema)
            print(f"   [PASS] Validation passed: {is_valid}")
        except Exception as e:
            print(f"   [FAIL] Validation failed: {e}")

        # Validate using validator object for detailed errors
        print("\n4. Checking for validation errors...")
        validator = Hdf5Validator(hdf5_file, schema)
        errors = list(validator.iter_errors())

        if errors:
            print(f"   Found {len(errors)} errors:")
            for i, error in enumerate(errors, 1):
                print(f"   {i}. {error}")
        else:
            print("   [PASS] No validation errors found!")

        # Show file structure
        print("\n5. HDF5 file structure:")
        with h5py.File(hdf5_file, 'r') as f:
            print(f"   Root attributes: {dict(f.attrs)}")
            print(f"   Groups: {list(f.keys())}")
            print(f"   Sensor datasets: {list(f['sensors'].keys())}")

        # Explicitly delete validator to close HDF5 files
        del validator
        # Force garbage collection to ensure HDF5 files are closed
        gc.collect()

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
