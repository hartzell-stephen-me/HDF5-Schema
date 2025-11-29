import h5py
import numpy as np
import pathlib
import shutil
import unittest
from hdf5schema.schema import GroupSchema
from hdf5schema.validator import Hdf5Validator

THIS_PATH = pathlib.Path(__file__).parent.resolve()
DATA_DIR = THIS_PATH / "data"


class TestValidator(unittest.TestCase):

    def setUp(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.fid = h5py.File(DATA_DIR / "test.h5", "w")

    def tearDown(self):
        self.fid.close()
        if DATA_DIR.exists:
            shutil.rmtree(DATA_DIR)

    def clear_fid(self):
        self.fid.close()
        self.fid = h5py.File(DATA_DIR / "test.h5", "w")

    def test_dataset_1d_no_attrs_check_exists(self):
        self.fid.create_dataset("Test Dataset", data=np.array([1,2,3,4], dtype=np.uint8))
        schema_dict = {
            "type": "group",
            "members": {
                "Test Dataset": {
                    "type": "dataset",
                    "description": "Test Dataset",
                    "dtype": "uint8",
                    "shape": [-1]
                },
                "required": [
                    "Test Dataset"
                ]
            },
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_dataset_1d_no_attrs_check_not_valid(self):
        self.fid.create_dataset("Test Dataset", data=np.array([1,2,3,4], dtype=np.uint8))
        schema_dict = {
            "type": "group",
            "members": {
                "Test Dataset 2": {
                    "type": "dataset",
                    "description": "Test Dataset",
                    "dtype": "uint8",
                    "shape": [-1]
                },
                "required": [
                    "Test Dataset 2"
                ]
            },
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()


    def test_complex_group(self):
        self.fid.create_dataset("attachments", data=np.array([( b"file1.txt", b"title1", b"caption1"),
                                                              ( b"file2.txt", b"title2", b"caption2")],
                                                            dtype=[("file", "S128"), ("title", "S128"), ("caption", "S128")]))
        tensors_grp = self.fid.create_group("tensors")
        tensors_grp.create_dataset("tensor1", data=np.random.rand(10,10), dtype=np.float32)
        tensors_grp.create_dataset("tensor2", data=np.random.rand(5,5,5), dtype=np.float32)

        schema_dict = {
            "type": "group",
            "description": "Root group",
            "members": {
                "attachments": {
                    "type": "dataset",
                    "description": "Attachments",
                    "dtype": [
                        {"name": "file",  "dtype": "S128"},
                        {"name": "title", "dtype": "S128"},
                        {"name": "caption","dtype": "S128"}
                    ]
                },
                "tensors": {
                    "type": "group",
                    "description": "Tenors description",
                    "patternMembers": {
                        "^.*$": {
                            "anyOf": [
                                {"type": "dataset", "dtype": "<f8"},
                                {"type": "dataset", "dtype": "<f4"}
                            ]
                        }
                    }
                }
            }
        }
        schema = GroupSchema(schema=schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_complex_group_failure(self):
        self.fid.create_dataset("attachments", data=np.array([( b"file1.txt", b"title1", b"caption1"),
                                                              ( b"file2.txt", b"title2", b"caption2")],
                                                            dtype=[("file", "S128"), ("title", "S128"), ("caption", "S128")]))
        tensors_grp = self.fid.create_group("tensors")
        tensors_grp.create_dataset("tensor1", data=np.random.rand(10,10), dtype=np.float32)
        tensors_grp.create_dataset("tensor2", data=np.random.rand(5,5,5), dtype=np.float32)
        self.fid.create_dataset("extra", data=np.array([1,2,3], dtype=np.uint8))
        schema_dict = {
            "type": "group",
            "description": "Root group",
            "members": {
                "attachments": {
                    "type": "dataset",
                    "description": "Attachments",
                    "dtype": [
                        {"name": "file",  "dtype": "S128"},
                        {"name": "title", "dtype": "S128"},
                        {"name": "caption","dtype": "S128"}
                    ]
                },
                "tensors": {
                    "type": "group",
                    "description": "Tenors description",
                    "patternMembers": {
                        "^.*$": {
                            "anyOf": [
                                {"type": "dataset", "dtype": "<f8"},
                                {"type": "dataset", "dtype": "<f4"}
                            ]
                        }
                    }
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_dataset_shape_wildcard_valid(self):
        self.fid.create_dataset("d1", data=np.zeros((5, 10), dtype=np.float32))
        schema_dict = {
            "type": "group",
            "members": {
                "d1": {
                    "type": "dataset",
                    "dtype": "float32",
                    "shape": [-1, 10]
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_dataset_shape_wildcard_invalid(self):
        self.fid.create_dataset("d1", data=np.zeros((5, 11), dtype=np.float32))
        schema_dict = {
            "type": "group",
            "members": {
                "d1": {
                    "type": "dataset",
                    "dtype": "float32",
                    "shape": [-1, 10]
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_exact_shape_valid(self):
        self.fid.create_dataset("d1", data=np.zeros((5, 10), dtype=np.float32))
        schema_dict = {
            "type": "group",
            "members": {
                "d1": {
                    "type": "dataset",
                    "dtype": "float32",
                    "shape": [5, 10]
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_required_attribute_present(self):
        dset = self.fid.create_dataset("d1", data=np.zeros(5, dtype=np.uint8))
        dset.attrs["version"] = np.uint8(1)
        schema_dict = {
            "type": "group",
            "members": {
                "d1": {
                    "type": "dataset",
                    "dtype": "uint8",
                    "shape": [-1],
                    "attrs": [
                        {"name": "version", "dtype": "uint8", "required": True}
                    ]
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_required_attribute_missing(self):
        self.fid.create_dataset("d1", data=np.zeros(5, dtype=np.uint8))
        # Intentionally do not set required attributes
        schema_dict = {
            "type": "group",
            "members": {
                "d1": {
                    "type": "dataset",
                    "dtype": "uint8",
                    "shape": [-1],
                    "attrs": [
                        {"name": "version", "dtype": "uint8", "required": True}
                    ]
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_extra_attribute_failure(self):
        dset = self.fid.create_dataset("d1", data=np.zeros(5, dtype=np.uint8))
        dset.attrs["unexpected"] = 5
        schema_dict = {
            "type": "group",
            "members": {
                "d1": {
                    "type": "dataset",
                    "dtype": "uint8",
                    "shape": [-1]
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_compound_dtype_dict(self):
        compound_dtype = np.dtype({
            "names": ["x", "y"],
            "formats": ["<f4", "<f4"]
        })
        data = np.zeros(3, dtype=compound_dtype)
        self.fid.create_dataset("points", data=data)
        schema_dict = {
            "type": "group",
            "members": {
                "points": {
                    "type": "dataset",
                    "dtype": {
                        "formats": [
                            {"name": "x", "format": "<f4"},
                            {"name": "y", "format": "<f4"}
                        ]
                    },
                    "shape": [-1]
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_json_schema_file(self):
        self.fid.create_dataset("d1", data=np.zeros(5, dtype=np.uint8))
        schema_path = THIS_PATH / "data" / "simple_schema.json"
        schema_dict = {
            "type": "group",
            "members": {
                "d1": {
                    "type": "dataset",
                    "dtype": "uint8",
                    "shape": [-1]
                }
            }
        }
        with open(schema_path, "w") as f:
            import json
            json.dump(schema_dict, f)

        validator = Hdf5Validator(self.fid, schema_path)
        self.assertTrue(validator.is_valid())
        schema_path.unlink()
        self.clear_fid()

    def test_offset_compound_dtype_valid(self):
        compound_dtype = np.dtype({
            "names": ["a", "b"],
            "formats": ["<i4", "<f8"],
            "offsets": [0, 8],
            "itemsize": 16
        })
        data = np.zeros(3, dtype=compound_dtype)
        self.fid.create_dataset("data", data=data)
        schema_dict = {
            "type": "group",
            "members": {
                "data": {
                    "type": "dataset",
                    "dtype": {
                        "formats": [
                            {"name": "a", "format": "<i4", "offset": 0},
                            {"name": "b", "format": "<f8", "offset": 8}
                        ],
                        "itemsize": 16
                    },
                    "shape": [-1]
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_offset_compound_dtype_invalid(self):
        compound_dtype = np.dtype({
            "names": ["a", "b"],
            "formats": ["<i4", "<f8"],
            "offsets": [0, 8],
            "itemsize": 16
        })
        data = np.zeros(3, dtype=compound_dtype)
        self.fid.create_dataset("data", data=data)
        schema_dict = {
            "type": "group",
            "members": {
                "data": {
                    "type": "dataset",
                    "dtype": {
                        "formats": [
                            {"name": "a", "format": "<i4", "offset": 0},
                            {"name": "b", "format": "<f8", "offset": 16}  # Incorrect offset
                        ],
                        "itemsize": 24
                    },
                    "shape": [-1]
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_dtype_larger_itemsize(self):
        compound_dtype = np.dtype({
            "names": ["a", "b"],
            "formats": ["<i4", "<f8"],
            "offsets": [0, 8],
            "itemsize": 32  # Larger than needed
        })
        data = np.zeros(3, dtype=compound_dtype)
        self.fid.create_dataset("data", data=data)
        schema_dict = {
            "type": "group",
            "members": {
                "data": {
                    "type": "dataset",
                    "dtype": {
                        "formats": [
                            {"name": "a", "format": "<i4", "offset": 0},
                            {"name": "b", "format": "<f8", "offset": 8}
                        ],
                        "itemsize": 32
                    },
                    "shape": [-1]
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_dtype_smaller_itemsize_invalid(self):
        compound_dtype = np.dtype({
            "names": ["a", "b", "c"],
            "formats": ["<i4", "<f8", "<f8"],
            "offsets": [0, 8, 16],
            "itemsize": 24
        })
        data = np.zeros(3, dtype=compound_dtype)
        self.fid.create_dataset("data", data=data)
        schema_dict = {
            "type": "group",
            "members": {
                "data": {
                    "type": "dataset",
                    "dtype": {
                        "formats": [
                            {"name": "a", "format": "<i4", "offset": 0},
                            {"name": "b", "format": "<f8", "offset": 8}
                        ],
                        "itemsize": 16
                    },
                    "shape": [-1]
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_required_attribute_wrong_dtype(self):
        dset = self.fid.create_dataset("d1", data=np.zeros(5, dtype=np.uint8))
        dset.attrs["version"] = "1"  # Wrong dtype, should be uint8
        schema_dict = {
            "type": "group",
            "members": {
                "d1": {
                    "type": "dataset",
                    "dtype": "uint8",
                    "shape": [-1],
                    "attrs": [
                        {"name": "version", "dtype": "uint8", "required": True}
                    ]
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_iter_errors_functionality(self):
        """Test that iter_errors collects all validation errors instead of raising on first error."""
        self.fid.create_dataset("d1", data=np.zeros(5, dtype=np.uint8))
        self.fid.create_dataset("extra", data=np.zeros(3, dtype=np.int32))  # Extra dataset not in schema

        schema_dict = {
            "type": "group",
            "members": {
                "d1": {
                    "type": "dataset",
                    "dtype": "float32",  # Wrong dtype
                    "shape": [10],  # Wrong shape
                    "attrs": [
                        {"name": "missing_attr", "dtype": "uint8", "required": True}  # Missing required attr
                    ]
                },
                "d2": {
                    "type": "dataset",
                    "dtype": "uint8"
                }
            },
            "required": ["d1", "d2"]  # d2 is missing
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)

        errors = validator.iter_errors()
        self.assertGreater(len(errors), 1)  # Should have multiple errors
        self.clear_fid()

    def test_empty_group_validation(self):
        """Test validation of an empty group."""
        schema_dict = {
            "type": "group",
            "members": {}
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_nested_groups_deep(self):
        """Test validation of deeply nested groups."""
        level1 = self.fid.create_group("level1")
        level2 = level1.create_group("level2")
        level3 = level2.create_group("level3")
        level3.create_dataset("deep_data", data=np.array([1, 2, 3]))

        schema_dict = {
            "type": "group",
            "members": {
                "level1": {
                    "type": "group",
                    "members": {
                        "level2": {
                            "type": "group",
                            "members": {
                                "level3": {
                                    "type": "group",
                                    "members": {
                                        "deep_data": {
                                            "type": "dataset",
                                            "dtype": "int64",
                                            "shape": [-1]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_scalar_dataset_validation(self):
        """Test validation of scalar datasets (0-dimensional)."""
        self.fid.create_dataset("scalar", data=42)

        schema_dict = {
            "type": "group",
            "members": {
                "scalar": {
                    "type": "dataset",
                    "dtype": "int64",
                    "shape": []  # Empty shape for scalar
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_string_dtype_validation(self):
        """Test validation of string dtypes."""
        string_data = np.array([b"hello", b"world"], dtype="S10")
        self.fid.create_dataset("strings", data=string_data)

        schema_dict = {
            "type": "group",
            "members": {
                "strings": {
                    "type": "dataset",
                    "dtype": "S10",
                    "shape": [-1]
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_variable_length_string_dtype_validation(self):
        """Test validation of variable-length string dtypes."""
        # HDF5 supports variable-length strings better than fixed Unicode
        string_dtype = h5py.string_dtype(encoding="utf-8")
        string_data = np.array(["hello", "world"], dtype=string_dtype)
        self.fid.create_dataset("var_strings", data=string_data)

        schema_dict = {
            "type": "group",
            "members": {
                "var_strings": {
                    "type": "dataset",
                    "dtype": string_dtype.str,  # Use the actual string representation
                    "shape": [-1]
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_multidimensional_shape_validation(self):
        """Test validation of multidimensional arrays with specific shapes."""
        data = np.random.rand(3, 4, 5)
        self.fid.create_dataset("multi_dim", data=data)

        schema_dict = {
            "type": "group",
            "members": {
                "multi_dim": {
                    "type": "dataset",
                    "dtype": "float64",
                    "shape": [3, 4, 5]
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_mixed_wildcard_shape(self):
        """Test validation with mixed wildcard and fixed dimensions."""
        data = np.random.rand(7, 4, 10)
        self.fid.create_dataset("mixed_shape", data=data)

        schema_dict = {
            "type": "group",
            "members": {
                "mixed_shape": {
                    "type": "dataset",
                    "dtype": "float64",
                    "shape": [-1, 4, -1]  # First and third dimensions are wildcards
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_attribute_shape_validation(self):
        """Test validation of attribute shapes."""
        dset = self.fid.create_dataset("data", data=np.zeros(5))
        dset.attrs["matrix_attr"] = np.array([[1, 2], [3, 4]])

        schema_dict = {
            "type": "group",
            "members": {
                "data": {
                    "type": "dataset",
                    "dtype": "float64",
                    "shape": [-1],
                    "attrs": [
                        {"name": "matrix_attr", "dtype": "int64", "shape": [2, 2], "required": True}
                    ]
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_invalid_file_path(self):
        """Test validation with invalid file path."""
        with self.assertRaises(FileNotFoundError):
            Hdf5Validator("/nonexistent/path.h5", {})

    def test_schema_from_dict(self):
        """Test creating validator directly from schema dict instead of GroupSchema object."""
        self.fid.create_dataset("test", data=np.array([1, 2, 3]))

        schema_dict = {
            "type": "group",
            "members": {
                "test": {
                    "type": "dataset",
                    "dtype": "int64",
                    "shape": [-1]
                }
            }
        }
        # Pass dict directly instead of GroupSchema object
        validator = Hdf5Validator(self.fid, schema_dict)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_dtype_endianness_same_size(self):
        """Test that dtypes with same size but different representation work."""
        data = np.array([1, 2, 3, 4], dtype=np.int32)  # Use native int32
        self.fid.create_dataset("endian_test", data=data)

        # The actual dtype in the file might be '<i4' or '>i4' depending on system
        actual_dtype = self.fid["endian_test"].dtype

        schema_dict = {
            "type": "group",
            "members": {
                "endian_test": {
                    "type": "dataset",
                    "dtype": str(actual_dtype),  # Use the actual dtype from the file
                    "shape": [-1]
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_pattern_members_no_match(self):
        """Test pattern members when no items match the pattern."""
        self.fid.create_dataset("data1", data=np.array([1, 2, 3]))

        schema_dict = {
            "type": "group",
            "patternMembers": {
                "^tensor_.*$": {  # Pattern that won't match "data1"
                    "type": "dataset",
                    "dtype": "int64"
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())  # data1 doesn't match pattern
        self.clear_fid()

    def test_optional_vs_required_members(self):
        """Test mixing of optional and required members."""
        self.fid.create_dataset("required_data", data=np.array([1, 2, 3]))
        # Don't create optional_data

        schema_dict = {
            "type": "group",
            "members": {
                "required_data": {
                    "type": "dataset",
                    "dtype": "int64",
                    "shape": [-1]
                },
                "optional_data": {
                    "type": "dataset",
                    "dtype": "float32",
                    "shape": [-1]
                }
            },
            "required": ["required_data"]  # Only required_data is required
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_unicode_string_dtype(self):
        """Test validation of Unicode string dtypes."""
        string_dtype = h5py.string_dtype(encoding="utf-8", length=10)
        string_data = np.array(["hello", "world"], dtype=string_dtype)
        self.fid.create_dataset("unicode_strings", data=string_data)

        schema_dict = {
            "type": "group",
            "members": {
                "unicode_strings": {
                    "type": "dataset",
                    "dtype": "U10",  # Unicode string of length 10
                    "shape": [-1]
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_deeply_nested_pattern_members(self):
        """Test validation of deeply nested pattern members structure."""
        # Create observables group with nested structure
        obs_grp = self.fid.create_group("observables")
        obs1_grp = obs_grp.create_group("observable_1")
        obs1_grp.create_dataset("data", data=np.array([1.0, 2.0, 3.0], dtype=np.float32))

        # Create nested geolocation group
        geo_grp = obs1_grp.create_group("geolocation")
        geo_grp.create_dataset("latitude", data=np.array([45.0, 46.0], dtype=np.float64))
        geo_grp.create_dataset("longitude", data=np.array([-122.0, -123.0], dtype=np.float64))

        # Create sensors group with pattern members
        sensors_grp = self.fid.create_group("sensors")
        sensor1_grp = sensors_grp.create_group("sensor_A")
        sensor1_grp.create_dataset("calibration", data=np.array([1.0, 1.1, 1.2], dtype=np.float32))

        schema_dict = {
            "type": "group",
            "members": {
                "observables": {
                    "type": "group",
                    "patternMembers": {
                        "^observable_.*$": {
                            "type": "group",
                            "members": {
                                "data": {
                                    "type": "dataset",
                                    "dtype": "float32",
                                    "shape": [-1]
                                },
                                "geolocation": {
                                    "type": "group",
                                    "patternMembers": {
                                        "^.*$": {
                                            "type": "dataset",
                                            "dtype": "float64",
                                            "shape": [-1]
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "sensors": {
                    "type": "group",
                    "patternMembers": {
                        "^sensor_.*$": {
                            "type": "group",
                            "members": {
                                "calibration": {
                                    "type": "dataset",
                                    "dtype": "float32",
                                    "shape": [-1]
                                }
                            }
                        }
                    }
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_anyOf_inside_pattern_members(self):
        """Test anyOf validation inside pattern members"""
        # Create observables group
        obs_grp = self.fid.create_group("observables")
        obs1_grp = obs_grp.create_group("observable_1")

        # Create geolocation group with mixed data types that should match anyOf
        geo_grp = obs1_grp.create_group("geolocation")
        geo_grp.create_dataset("latitude", data=np.array([45.0, 46.0], dtype=np.float64))
        geo_grp.create_dataset("altitude", data=np.array([100, 200], dtype=np.int32))  # Different dtype

        schema_dict = {
            "type": "group",
            "members": {
                "observables": {
                    "type": "group",
                    "patternMembers": {
                        "^observable_.*$": {
                            "type": "group",
                            "members": {
                                "geolocation": {
                                    "type": "group",
                                    "patternMembers": {
                                        "^.*$": {
                                            "anyOf": [
                                                {
                                                    "type": "dataset",
                                                    "dtype": "float64",
                                                    "shape": [-1]
                                                },
                                                {
                                                    "type": "dataset",
                                                    "dtype": "int32",
                                                    "shape": [-1]
                                                },
                                                {
                                                    "type": "group"
                                                }
                                            ]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_object_dtype_attributes(self):
        """Test validation of attributes with object dtype like countries"""
        # Create a dataset and add object dtype attribute
        dset = self.fid.create_dataset("root_data", data=np.array([1, 2, 3]))

        # Create object array attribute (like countries list)
        countries = np.array(["USA", "Canada", "Mexico"], dtype=object)
        dset.attrs["countries"] = countries

        # Also test with a simple string attribute
        dset.attrs["metadata"] = "some_string_value"

        schema_dict = {
            "type": "group",
            "members": {
                "root_data": {
                    "type": "dataset",
                    "dtype": "int64",
                    "shape": [-1],
                    "attrs": [
                        {
                            "name": "countries",
                            "dtype": "object",  # HDF5 stores object arrays as 'object'
                            "shape": [-1],
                            "required": True
                        },
                        {
                            "name": "metadata",
                            "dtype": "U17",  # HDF5 stores strings with specific length
                            "required": False
                        }
                    ]
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_complex_nested_required_attributes(self):
        """Test deeply nested required/optional attributes in complex structures."""
        # Create root group with required attributes (store as numpy strings to get proper dtypes)
        self.fid.attrs["base_time"] = np.bytes_("2025-09-18T10:00:00.000000Z")
        self.fid.attrs["creation_time"] = np.bytes_("2025-09-18T10:00:00.000000Z")
        self.fid.attrs["end_time"] = np.bytes_("2025-09-18T11:00:00.000000Z")
        self.fid.attrs["icd_version"] = np.bytes_("1.0.0")
        self.fid.attrs["product_id"] = np.bytes_("A")
        self.fid.attrs["product_version"] = np.bytes_("2.1.0")
        self.fid.attrs["description"] = np.bytes_("Test description")  # Optional

        # Create complex nested structure with required/optional elements
        obs_grp = self.fid.create_group("observables")
        obs1_grp = obs_grp.create_group("observable_1")
        obs1_grp.create_dataset("data", data=np.array([1.0, 2.0], dtype=np.float32))

        # Add required attributes to the dataset
        obs1_grp["data"].attrs["valid_range"] = np.array([0.0, 100.0], dtype=np.float32)
        # Skip optional attribute to test it's truly optional

        schema_dict = {
            "type": "group",
            "attrs": [
                {"name": "base_time", "dtype": "S27", "required": True},
                {"name": "creation_time", "dtype": "S27", "required": True},
                {"name": "end_time", "dtype": "S27", "required": True},
                {"name": "icd_version", "dtype": "S5", "required": True},
                {"name": "product_id", "dtype": "S1", "required": True},
                {"name": "product_version", "dtype": "S5", "required": True},
                {"name": "description", "dtype": "S16", "required": False}  # Optional
            ],
            "members": {
                "observables": {
                    "type": "group",
                    "patternMembers": {
                        "^observable_.*$": {
                            "type": "group",
                            "members": {
                                "data": {
                                    "type": "dataset",
                                    "dtype": "float32",
                                    "shape": [-1],
                                    "attrs": [
                                        {"name": "valid_range", "dtype": "float32", "shape": [2], "required": True},
                                        {"name": "fill_value", "dtype": "float32", "required": False}  # Optional
                                    ]
                                }
                            },
                            "required": ["data"]
                        }
                    }
                }
            },
            "required": ["observables"]
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_attributes_with_shapes_in_pattern_members(self):
        """Test attributes with shapes and complex dtypes within pattern members."""
        # Create sensors group with pattern members
        sensors_grp = self.fid.create_group("sensors")
        sensor1_grp = sensors_grp.create_group("sensor_Alpha")
        sensor2_grp = sensors_grp.create_group("sensor_Beta")

        # Create datasets with complex attributes having shapes
        cal1 = sensor1_grp.create_dataset("calibration", data=np.array([1.0, 1.1], dtype=np.float32))
        cal1.attrs["coefficients"] = np.array([1.0, 0.5, 0.1], dtype=np.float64)  # 1D array attr
        cal1.attrs["transformation_matrix"] = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # 2D array attr
        cal1.attrs["sensor_id"] = 12345

        cal2 = sensor2_grp.create_dataset("calibration", data=np.array([2.0, 2.1], dtype=np.float32))
        cal2.attrs["coefficients"] = np.array([2.0, 1.5, 0.2], dtype=np.float64)
        cal2.attrs["transformation_matrix"] = np.array([[2.0, 0.1], [0.1, 2.0]], dtype=np.float32)
        cal2.attrs["sensor_id"] = 67890

        # Create complex compound dtype dataset within pattern members
        compound_dtype = np.dtype([
            ("timestamp", "<i8"),
            ("value", "<f4"),
            ("quality", "S10")  # Use byte strings instead of Unicode
        ])
        readings_data = np.array([(1695123456, 25.5, b"good"), (1695123457, 26.0, b"excellent")],
                                dtype=compound_dtype)

        sensor1_grp.create_dataset("readings", data=readings_data)
        sensor2_grp.create_dataset("readings", data=readings_data)

        schema_dict = {
            "type": "group",
            "members": {
                "sensors": {
                    "type": "group",
                    "patternMembers": {
                        "^sensor_.*$": {
                            "type": "group",
                            "members": {
                                "calibration": {
                                    "type": "dataset",
                                    "dtype": "float32",
                                    "shape": [-1],
                                    "attrs": [
                                        {
                                            "name": "coefficients",
                                            "dtype": "float64",
                                            "shape": [3],
                                            "required": True
                                        },
                                        {
                                            "name": "transformation_matrix",
                                            "dtype": "float32",
                                            "shape": [2, 2],
                                            "required": True
                                        },
                                        {
                                            "name": "sensor_id",
                                            "dtype": "int64",
                                            "required": True
                                        }
                                    ]
                                },
                                "readings": {
                                    "type": "dataset",
                                    "dtype": {
                                        "formats": [
                                            {"name": "timestamp", "format": "<i8"},
                                            {"name": "value", "format": "<f4"},
                                            {"name": "quality", "format": "S10"}
                                        ]
                                    },
                                    "shape": [-1]
                                }
                            },
                            "required": ["calibration", "readings"]
                        }
                    }
                }
            },
            "required": ["sensors"]
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_ref_def_members(self):
        """Test $ref and definitions in schema."""
        self.fid.create_dataset("data", data=np.array([1, 2, 3], dtype=np.int32))

        schema_dict = {
            "type": "group",
            "members": {
                "data": {
                    "$ref": "#/$defs/int32_dataset"
                }
            },
            "$defs": {
                "int32_dataset": {
                    "type": "dataset",
                    "dtype": "int32",
                    "shape": [-1]
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_ref_def_members_invalid(self):
        """Test $ref and definitions in schema with invalid data."""
        self.fid.create_dataset("data", data=np.array([1.0, 2.0, 3.0], dtype=np.float32))  # Wrong dtype

        schema_dict = {
            "type": "group",
            "members": {
                "data": {
                    "$ref": "#/$defs/int32_dataset"
                }
            },
            "$defs": {
                "int32_dataset": {
                    "type": "dataset",
                    "dtype": "int32",
                    "shape": [-1]
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

if __name__ == "__main__":
    unittest.main()
