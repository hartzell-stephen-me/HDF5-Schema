"""
Integration Tests
==================

End-to-end integration tests for the hdf5schema package.
Tests complete workflows including file creation, schema definition,
validation, and CLI operations.
"""
import gc
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from hdf5schema.generate_schema import generate_schema
from hdf5schema.validate import validate
from hdf5schema.validator import Hdf5Validator


class TestBasicWorkflow(unittest.TestCase):
    """Test basic end-to-end validation workflows."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmppath = Path(self.tmpdir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        # Force garbage collection to close any open HDF5 files
        gc.collect()
        self.tmpdir.cleanup()

    def test_sensor_data_workflow(self):
        """Test complete workflow for sensor data validation."""
        # Create HDF5 file with sensor data
        hdf5_file = self.tmppath / "sensors.h5"
        with h5py.File(hdf5_file, 'w') as f:
            f.attrs['experiment'] = 'Temperature Monitoring'
            f.attrs['version'] = '1.0'

            sensors = f.create_group('sensors')
            temp = sensors.create_dataset('temperature', data=[20.5, 21.0, 20.8], dtype='float64')
            temp.attrs['unit'] = 'celsius'

            humid = sensors.create_dataset('humidity', data=[45.2, 46.1, 45.8], dtype='float64')
            humid.attrs['unit'] = 'percent'

        # Create schema
        schema = {
            "type": "group",
            "attrs": [
                {"name": "experiment", "dtype": "U128"},
                {"name": "version", "dtype": "U128"}
            ],
            "members": {
                "sensors": {
                    "type": "group",
                    "members": {
                        "temperature": {
                            "type": "dataset",
                            "dtype": "<f8",
                            "shape": [3],
                            "attrs": [{"name": "unit", "dtype": "U128"}]
                        },
                        "humidity": {
                            "type": "dataset",
                            "dtype": "<f8",
                            "shape": [3],
                            "attrs": [{"name": "unit", "dtype": "U128"}]
                        }
                    },
                    "required": ["temperature", "humidity"]
                }
            },
            "required": ["sensors"]
        }

        # Validate using Python API
        is_valid = validate(hdf5_file, schema)
        self.assertTrue(is_valid)

        # Validate using validator object
        validator = Hdf5Validator(hdf5_file, schema)
        errors = list(validator.iter_errors())
        self.assertEqual(len(errors), 0)

    def test_invalid_data_workflow(self):
        """Test workflow with invalid data produces expected errors."""
        # Create HDF5 file with missing required member
        hdf5_file = self.tmppath / "invalid.h5"
        with h5py.File(hdf5_file, 'w') as f:
            f.attrs['version'] = '1.0'
            sensors = f.create_group('sensors')
            # Missing 'humidity' dataset - only has temperature
            sensors.create_dataset('temperature', data=[20.5], dtype='float64')

        # Schema requires both temperature and humidity
        schema = {
            "type": "group",
            "members": {
                "sensors": {
                    "type": "group",
                    "members": {
                        "temperature": {"type": "dataset", "dtype": "<f8"},
                        "humidity": {"type": "dataset", "dtype": "<f8"}
                    },
                    "required": ["temperature", "humidity"]
                }
            }
        }

        # Validation should fail due to missing humidity
        validator = Hdf5Validator(hdf5_file, schema)
        errors = list(validator.iter_errors())
        self.assertGreater(len(errors), 0, "Should have validation errors for missing required member")
        # Check that errors mention the missing humidity dataset
        error_messages = [str(e) for e in errors]
        has_humidity_error = any('humidity' in msg for msg in error_messages)
        self.assertTrue(has_humidity_error, f"Expected error about missing 'humidity'. Got: {error_messages}")


class TestSchemaGeneration(unittest.TestCase):
    """Test schema generation and validation round-trip."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmppath = Path(self.tmpdir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        # Force garbage collection to close any open HDF5 files
        gc.collect()
        self.tmpdir.cleanup()

    def test_generate_and_validate_roundtrip(self):
        """Test generating schema from file and validating against it."""
        # Create HDF5 file
        hdf5_file = self.tmppath / "data.h5"
        with h5py.File(hdf5_file, 'w') as f:
            f.attrs['title'] = 'Test Data'
            f.create_dataset('data', data=np.arange(100), dtype='int32')
            f.create_dataset('timestamps', data=np.arange(100), dtype='float64')

        # Generate schema
        schema = generate_schema(hdf5_file)

        # Validate original file against generated schema
        validator = Hdf5Validator(hdf5_file, schema)
        errors = list(validator.iter_errors())
        self.assertEqual(len(errors), 0, f"Generated schema should validate source file. Errors: {errors}")

    def test_generate_schema_from_group(self):
        """Test generating schema from specific group path."""
        # Create HDF5 file with nested groups
        hdf5_file = self.tmppath / "nested.h5"
        with h5py.File(hdf5_file, 'w') as f:
            results = f.create_group('results')
            experiment1 = results.create_group('experiment1')
            experiment1.create_dataset('data', data=[1, 2, 3], dtype='int32')

        # Generate schema for specific group
        schema = generate_schema(hdf5_file, group_path='/results/experiment1')

        # Schema should be for a group containing 'data' dataset
        self.assertEqual(schema['type'], 'group')
        self.assertIn('members', schema)
        self.assertIn('data', schema['members'])

    def test_generate_schema_with_compound_dtype(self):
        """Test schema generation with compound datatypes."""
        # Create HDF5 file with compound dtype
        hdf5_file = self.tmppath / "compound.h5"

        # Use fixed-length ASCII strings (S type) which work better with h5py
        compound_dtype = np.dtype([
            ('id', '<i4'),
            ('name', 'S32'),
            ('value', '<f8')
        ])

        data = np.array([
            (1, b'sensor_a', 3.14),
            (2, b'sensor_b', 2.71)
        ], dtype=compound_dtype)

        with h5py.File(hdf5_file, 'w') as f:
            f.create_dataset('sensors', data=data)

        # Generate schema
        schema = generate_schema(hdf5_file)

        # Validate against generated schema
        validator = Hdf5Validator(hdf5_file, schema)
        errors = list(validator.iter_errors())
        self.assertEqual(len(errors), 0)


class TestPatternMatching(unittest.TestCase):
    """Integration tests for pattern matching features."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmppath = Path(self.tmpdir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        # Force garbage collection to close any open HDF5 files
        gc.collect()
        self.tmpdir.cleanup()

    def test_multi_channel_pattern_workflow(self):
        """Test pattern matching with multiple channels."""
        # Create HDF5 file with multiple channels
        hdf5_file = self.tmppath / "channels.h5"
        with h5py.File(hdf5_file, 'w') as f:
            for i in range(1, 11):
                channel_name = f'ch_{i:02d}'
                dataset = f.create_dataset(channel_name, data=range(100), dtype='int32')
                dataset.attrs['channel_number'] = i

        # Create schema with pattern matching
        schema = {
            "type": "group",
            "patternMembers": {
                r"^ch_\d{2}$": {
                    "type": "dataset",
                    "dtype": "<i4",
                    "shape": [100],
                    "attrs": [
                        {"name": "channel_number", "dtype": "<i8"}
                    ]
                }
            }
        }

        # Validate
        validator = Hdf5Validator(hdf5_file, schema)
        errors = list(validator.iter_errors())
        self.assertEqual(len(errors), 0)


class TestConditionalValidation(unittest.TestCase):
    """Integration tests for conditional validation."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmppath = Path(self.tmpdir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        # Force garbage collection to close any open HDF5 files
        gc.collect()
        self.tmpdir.cleanup()

    def test_conditional_rgb_grayscale_workflow(self):
        """Test conditional validation for RGB vs grayscale images."""
        schema = {
            "type": "group",
            "attrs": [{"name": "image_type", "dtype": "U128"}],
            "members": {
                "image": {"type": "dataset", "dtype": "|u1"}
            },
            "required": ["image"],
            "if": {
                "attrs": [{"name": "image_type", "const": "rgb"}]
            },
            "then": {
                "members": {
                    "image": {"type": "dataset", "dtype": "|u1", "shape": [256, 256, 3]}
                }
            },
            "else": {
                "members": {
                    "image": {"type": "dataset", "dtype": "|u1", "shape": [256, 256]}
                }
            }
        }

        # Test RGB image
        rgb_file = self.tmppath / "rgb.h5"
        with h5py.File(rgb_file, 'w') as f:
            f.attrs['image_type'] = 'rgb'
            f.create_dataset('image', data=np.zeros((256, 256, 3), dtype='uint8'))

        validator = Hdf5Validator(rgb_file, schema)
        errors = list(validator.iter_errors())
        self.assertEqual(len(errors), 0)

        # Test grayscale image
        gray_file = self.tmppath / "gray.h5"
        with h5py.File(gray_file, 'w') as f:
            f.attrs['image_type'] = 'grayscale'
            f.create_dataset('image', data=np.zeros((256, 256), dtype='uint8'))

        validator = Hdf5Validator(gray_file, schema)
        errors = list(validator.iter_errors())
        self.assertEqual(len(errors), 0)


class TestBooleanLogic(unittest.TestCase):
    """Integration tests for boolean logic operators."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmppath = Path(self.tmpdir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        # Force garbage collection to close any open HDF5 files
        gc.collect()
        self.tmpdir.cleanup()

    def test_anyof_workflow(self):
        """Test anyOf validation workflow."""
        # Schema requires at least one of two datasets to be present
        # Based on test_allOf_group_level from test_boolean_expressions.py
        schema = {
            "type": "group",
            "anyOf": [
                {
                    "members": {
                        "data1": {"type": "dataset", "dtype": "<i4", "shape": [-1]}
                    },
                    "required": ["data1"]
                },
                {
                    "members": {
                        "data2": {"type": "dataset", "dtype": "<i4", "shape": [-1]}
                    },
                    "required": ["data2"]
                }
            ]
        }

        # File with data1 should pass
        file_a = self.tmppath / "file_a.h5"
        with h5py.File(file_a, 'w') as f:
            f.create_dataset('data1', data=np.array([1, 2, 3], dtype=np.int32))

        validator = Hdf5Validator(file_a, schema)
        errors = list(validator.iter_errors())
        self.assertEqual(len(errors), 0, f"File with data1 should pass: {errors}")

        # File with data2 should pass
        file_b = self.tmppath / "file_b.h5"
        with h5py.File(file_b, 'w') as f:
            f.create_dataset('data2', data=np.array([4, 5, 6], dtype=np.int32))

        validator = Hdf5Validator(file_b, schema)
        errors = list(validator.iter_errors())
        self.assertEqual(len(errors), 0, f"File with data2 should pass: {errors}")

    def test_oneof_workflow(self):
        """Test oneOf validation workflow."""
        # Schema with oneOf: exactly one alternative must match
        # Data with int32 dtype matches first schema, float32 matches second
        schema = {
            "type": "group",
            "oneOf": [
                {
                    "members": {
                        "data": {"type": "dataset", "dtype": "<i4", "shape": [-1]}
                    },
                    "required": ["data"]
                },
                {
                    "members": {
                        "data": {"type": "dataset", "dtype": "<f4", "shape": [-1]}
                    },
                    "required": ["data"]
                }
            ]
        }

        # File with int32 data should pass (matches first alternative only)
        file_int = self.tmppath / "file_int.h5"
        with h5py.File(file_int, 'w') as f:
            f.create_dataset('data', data=np.array([1, 2, 3], dtype=np.int32))

        validator = Hdf5Validator(file_int, schema)
        errors = list(validator.iter_errors())
        self.assertEqual(len(errors), 0, f"File with int32 should pass: {errors}")

        # File with float32 data should pass (matches second alternative only)
        file_float = self.tmppath / "file_float.h5"
        with h5py.File(file_float, 'w') as f:
            f.create_dataset('data', data=np.array([1.5, 2.5, 3.5], dtype=np.float32))

        validator = Hdf5Validator(file_float, schema)
        errors = list(validator.iter_errors())
        self.assertEqual(len(errors), 0, f"File with float32 should pass: {errors}")


class TestRealWorldScenarios(unittest.TestCase):
    """Integration tests for real-world usage scenarios."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmppath = Path(self.tmpdir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        # Force garbage collection to close any open HDF5 files
        gc.collect()
        self.tmpdir.cleanup()

    def test_scientific_experiment_workflow(self):
        """Test complete workflow for scientific experiment data."""
        # Create realistic scientific data file
        hdf5_file = self.tmppath / "experiment.h5"
        with h5py.File(hdf5_file, 'w') as f:
            # Metadata
            f.attrs['experiment_id'] = 'EXP-2025-001'
            f.attrs['date'] = '2025-01-29'
            f.attrs['researcher'] = 'Jane Doe'

            # Raw measurements
            raw = f.create_group('raw_data')
            raw.create_dataset('temperature', data=np.random.randn(1000) * 5 + 20, dtype='float64')
            raw.create_dataset('pressure', data=np.random.randn(1000) * 2 + 101.3, dtype='float64')
            raw.create_dataset('timestamps', data=np.arange(1000), dtype='int64')

            # Processed results
            processed = f.create_group('processed')
            processed.create_dataset('filtered_temperature', data=np.random.randn(1000) * 4 + 20, dtype='float64')
            processed.attrs['filter_type'] = 'lowpass'
            processed.attrs['cutoff_frequency'] = 0.1

            # Analysis
            analysis = f.create_group('analysis')
            analysis.create_dataset('mean_temp', data=20.0, dtype='float64')
            analysis.create_dataset('std_temp', data=5.0, dtype='float64')
            analysis.create_dataset('correlation', data=np.random.randn(1000, 2), dtype='float64')

        # Generate schema from this file
        generated_schema = generate_schema(hdf5_file)

        # Validate original file
        validator = Hdf5Validator(hdf5_file, generated_schema)
        errors = list(validator.iter_errors())
        self.assertEqual(len(errors), 0)

        # Create modified file that should validate
        hdf5_file2 = self.tmppath / "experiment2.h5"
        with h5py.File(hdf5_file2, 'w') as f:
            f.attrs['experiment_id'] = 'EXP-2025-002'
            f.attrs['date'] = '2025-01-30'
            f.attrs['researcher'] = 'John Smith'

            raw = f.create_group('raw_data')
            raw.create_dataset('temperature', data=np.random.randn(1000) * 5 + 25, dtype='float64')
            raw.create_dataset('pressure', data=np.random.randn(1000) * 2 + 100, dtype='float64')
            raw.create_dataset('timestamps', data=np.arange(1000), dtype='int64')

            processed = f.create_group('processed')
            processed.create_dataset('filtered_temperature', data=np.random.randn(1000) * 4 + 25, dtype='float64')
            processed.attrs['filter_type'] = 'highpass'
            processed.attrs['cutoff_frequency'] = 0.2

            analysis = f.create_group('analysis')
            analysis.create_dataset('mean_temp', data=25.0, dtype='float64')
            analysis.create_dataset('std_temp', data=5.5, dtype='float64')
            analysis.create_dataset('correlation', data=np.random.randn(1000, 2), dtype='float64')

        # Second file should also validate
        validator2 = Hdf5Validator(hdf5_file2, generated_schema)
        errors2 = list(validator2.iter_errors())
        self.assertEqual(len(errors2), 0)


if __name__ == '__main__':
    unittest.main()
