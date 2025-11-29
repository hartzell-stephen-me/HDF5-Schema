import h5py
import numpy as np
import pathlib
import shutil
import unittest
from hdf5schema.schema import GroupSchema
from hdf5schema.validator import Hdf5Validator

THIS_PATH = pathlib.Path(__file__).parent.resolve()
DATA_DIR = THIS_PATH / "data"

class TestBooleanExpressions(unittest.TestCase):

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

    def test_allOf_group_level(self):
        """Test validation using allOf at the group level - all schemas must pass."""
        self.fid.create_dataset("data1", data=np.array([1, 2, 3], dtype=np.int32))
        self.fid.create_dataset("data2", data=np.array([4, 5, 6], dtype=np.int32))

        schema_dict = {
            "type": "group",
            "allOf": [
                {
                    "members": {
                        "data1": {
                            "type": "dataset",
                            "dtype": "int32",
                            "shape": [-1]
                        }
                    },
                },
                {
                    "members": {
                        "data2": {
                            "type": "dataset",
                            "dtype": "int32",
                            "shape": [-1]
                        }
                    },
                }
            ]
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_allOf_group_level_invalid(self):
        """Test invalid case for allOf at the group level - one schema fails."""
        self.fid.create_dataset("data1", data=np.array([1, 2, 3], dtype=np.int32))
        # Missing data2 that's required by second allOf schema

        schema_dict = {
            "type": "group",
            "allOf": [
                {
                    "members": {
                        "data1": {
                            "type": "dataset",
                            "dtype": "int32",
                            "shape": [-1]
                        }
                    },
                    "required": ["data1"]
                },
                {
                    "members": {
                        "data2": {
                            "type": "dataset",
                            "dtype": "int32",
                            "shape": [-1]
                        }
                    },
                    "required": ["data2"]
                }
            ]
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_oneOf_group_level(self):
        """Test validation using oneOf at the group level - exactly one schema must pass."""
        self.fid.create_dataset("data", data=np.array([1, 2, 3], dtype=np.int32))

        schema_dict = {
            "type": "group",
            "oneOf": [
                {
                    "members": {
                        "data": {
                            "type": "dataset",
                            "dtype": "int32",
                            "shape": [-1]
                        }
                    },
                    "required": ["data"]
                },
                {
                    "members": {
                        "data": {
                            "type": "dataset",
                            "dtype": "float32",
                            "shape": [-1]
                        }
                    },
                    "required": ["data"]
                }
            ]
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_oneOf_group_level_multiple_match(self):
        """Test oneOf validation fails when multiple schemas match."""
        self.fid.create_dataset("data", data=np.array([1, 2, 3], dtype=np.int32))

        schema_dict = {
            "type": "group",
            "oneOf": [
                {
                    "members": {
                        "data": {
                            "type": "dataset",
                            "dtype": "int32",
                            "shape": [-1]
                        }
                    },
                    "required": ["data"]
                },
                {
                    "members": {
                        "data": {
                            "type": "dataset",
                            "dtype": "int32",  # Same as first schema - both will match
                            "shape": [-1]
                        }
                    },
                    "required": ["data"]
                }
            ]
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_oneOf_group_level_no_match(self):
        """Test oneOf validation fails when no schemas match."""
        self.fid.create_dataset("data", data=np.array([1, 2, 3], dtype=np.int32))

        schema_dict = {
            "type": "group",
            "oneOf": [
                {
                    "members": {
                        "data": {
                            "type": "dataset",
                            "dtype": "float32",  # Wrong dtype
                            "shape": [-1]
                        }
                    },
                    "required": ["data"]
                },
                {
                    "members": {
                        "data": {
                            "type": "dataset",
                            "dtype": "float64",  # Wrong dtype
                            "shape": [-1]
                        }
                    },
                    "required": ["data"]
                }
            ]
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_enum_dataset_scalar(self):
        """Test enum validation for scalar datasets."""
        self.fid.create_dataset("status", data=np.int32(1))

        schema_dict = {
            "type": "group",
            "members": {
                "status": {
                    "type": "dataset",
                    "dtype": "int32",
                    "shape": [],
                    "enum": [0, 1, 2]  # Allowed status values
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_enum_dataset_scalar_invalid(self):
        """Test enum validation fails for invalid scalar values."""
        self.fid.create_dataset("status", data=np.int32(5))  # Not in enum

        schema_dict = {
            "type": "group",
            "members": {
                "status": {
                    "type": "dataset",
                    "dtype": "int32",
                    "shape": [],
                    "enum": [0, 1, 2]  # Allowed status values
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_enum_dataset_array(self):
        """Test enum validation for array datasets."""
        self.fid.create_dataset("categories", data=np.array([0, 1, 2, 1, 0], dtype=np.int32))

        schema_dict = {
            "type": "group",
            "members": {
                "categories": {
                    "type": "dataset",
                    "dtype": "int32",
                    "shape": [-1],
                    "enum": [0, 1, 2]  # Allowed category values
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_enum_dataset_array_invalid(self):
        """Test enum validation fails for arrays with invalid values."""
        self.fid.create_dataset("categories", data=np.array([0, 1, 3, 1, 0], dtype=np.int32))  # 3 not in enum

        schema_dict = {
            "type": "group",
            "members": {
                "categories": {
                    "type": "dataset",
                    "dtype": "int32",
                    "shape": [-1],
                    "enum": [0, 1, 2]  # Allowed category values
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_const_dataset_scalar(self):
        """Test const validation for scalar datasets."""
        self.fid.create_dataset("version", data=np.int32(42))

        schema_dict = {
            "type": "group",
            "members": {
                "version": {
                    "type": "dataset",
                    "dtype": "int32",
                    "shape": [],
                    "const": 42  # Must be exactly 42
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_const_dataset_scalar_invalid(self):
        """Test const validation fails for wrong scalar values."""
        self.fid.create_dataset("version", data=np.int32(43))  # Wrong value

        schema_dict = {
            "type": "group",
            "members": {
                "version": {
                    "type": "dataset",
                    "dtype": "int32",
                    "shape": [],
                    "const": 42  # Must be exactly 42
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_const_dataset_array(self):
        """Test const validation for array datasets."""
        self.fid.create_dataset("flags", data=np.array([1, 1, 1, 1], dtype=np.int32))

        schema_dict = {
            "type": "group",
            "members": {
                "flags": {
                    "type": "dataset",
                    "dtype": "int32",
                    "shape": [-1],
                    "const": 1  # All values must be 1
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_const_dataset_array_invalid(self):
        """Test const validation fails for arrays with wrong values."""
        self.fid.create_dataset("flags", data=np.array([1, 1, 0, 1], dtype=np.int32))  # Contains 0

        schema_dict = {
            "type": "group",
            "members": {
                "flags": {
                    "type": "dataset",
                    "dtype": "int32",
                    "shape": [-1],
                    "const": 1  # All values must be 1
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_comment_property(self):
        """Test that $comment property is accessible but doesn't affect validation."""
        self.fid.create_dataset("data", data=np.array([1, 2, 3], dtype=np.int32))

        schema_dict = {
            "type": "group",
            "$comment": "This is a test schema with comments",
            "members": {
                "data": {
                    "type": "dataset",
                    "dtype": "int32",
                    "shape": [-1],
                    "$comment": "This dataset contains test data"
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)

        # Validation should pass regardless of comments
        self.assertTrue(validator.is_valid())

        # Comments should be accessible
        self.assertEqual(schema.comment, "This is a test schema with comments")
        data_schema = schema["data"]
        self.assertEqual(data_schema.comment, "This dataset contains test data")

        self.clear_fid()

    def test_not_group_level(self):
        """Test validation using not at the group level - schema must not validate."""
        self.fid.create_dataset("data", data=np.array([1, 2, 3], dtype=np.int32))

        schema_dict = {
            "type": "group",
            "not": {
                "members": {
                    "wrong_data": {  # This member doesn't exist, so not schema won't validate
                        "type": "dataset",
                        "dtype": "int32",
                        "shape": [-1]
                    }
                },
                "required": ["wrong_data"]
            },
            "members": {
                "data": {
                    "type": "dataset",
                    "dtype": "int32",
                    "shape": [-1]
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())  # Should pass because not schema fails to validate
        self.clear_fid()

    def test_not_group_level_invalid(self):
        """Test not validation fails when the not schema actually validates."""
        self.fid.create_dataset("data", data=np.array([1, 2, 3], dtype=np.int32))

        schema_dict = {
            "type": "group",
            "not": {
                "members": {
                    "data": {  # This member exists and matches, so not schema will validate (which is bad)
                        "type": "dataset",
                        "dtype": "int32",
                        "shape": [-1]
                    }
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())  # Should fail because not schema validates
        self.clear_fid()

    def test_not_dataset_dtype(self):
        """Test not validation for dataset dtypes."""
        self.fid.create_dataset("data", data=np.array([1, 2, 3], dtype=np.int32))

        schema_dict = {
            "type": "group",
            "members": {
                "data": {
                    "type": "dataset",
                    "dtype": "int32",
                    "shape": [-1],
                    "not": {
                        "dtype": "float32"  # Data is int32, not float32, so this should pass
                    }
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_not_dataset_dtype_invalid(self):
        """Test not validation fails when dataset matches the not constraint."""
        self.fid.create_dataset("data", data=np.array([1, 2, 3], dtype=np.int32))

        schema_dict = {
            "type": "group",
            "members": {
                "data": {
                    "type": "dataset",
                    "dtype": "int32",  # This matches
                    "shape": [-1],
                    "not": {
                        "dtype": "int32"  # Data is int32, which matches this not constraint (bad)
                    }
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_not_dataset_shape(self):
        """Test not validation for dataset shapes."""
        self.fid.create_dataset("data", data=np.array([1, 2, 3], dtype=np.int32))  # Shape is [3]

        schema_dict = {
            "type": "group",
            "members": {
                "data": {
                    "type": "dataset",
                    "dtype": "int32",
                    "shape": [-1],
                    "not": {
                        "shape": [5]  # Data shape is [3], not [5], so this should pass
                    }
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_not_value(self):
        """Test not validation for specific dataset values."""
        self.fid.create_dataset("status", data=np.int32(1))

        schema_dict = {
            "type": "group",
            "members": {
                "status": {
                    "type": "dataset",
                    "dtype": "int32",
                    "shape": [],
                    "not": {
                        "const": 0  # Status is 1, not 0, so this should pass
                    }
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_not_value_invalid(self):
        """Test not validation fails when dataset matches the const constraint."""
        self.fid.create_dataset("status", data=np.int32(0))  # Matches the not constraint

        schema_dict = {
            "type": "group",
            "members": {
                "status": {
                    "type": "dataset",
                    "dtype": "int32",
                    "shape": [],
                    "not": {
                        "const": 0  # Status is 0, which matches this not constraint (bad)
                    }
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_not_enum_values(self):
        """Test not validation with enum constraints."""
        self.fid.create_dataset("categories", data=np.array([5, 6, 7], dtype=np.int32))  # Not in enum

        schema_dict = {
            "type": "group",
            "members": {
                "categories": {
                    "type": "dataset",
                    "dtype": "int32",
                    "shape": [-1],
                    "not": {
                        "enum": [0, 1, 2]  # Data contains [5,6,7] which are not in this enum, so should pass
                    }
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_not_enum_values_invalid(self):
        """Test not validation fails when dataset values are in the enum."""
        self.fid.create_dataset("categories", data=np.array([0, 1, 2], dtype=np.int32))  # All in enum

        schema_dict = {
            "type": "group",
            "members": {
                "categories": {
                    "type": "dataset",
                    "dtype": "int32",
                    "shape": [-1],
                    "not": {
                        "enum": [0, 1, 2]  # Data contains values that are all in this enum (bad)
                    }
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_anyOf_dtype(self):
        """Test validation using anyOf for multiple acceptable dtypes."""
        self.fid.create_dataset("flexible_data", data=np.array([1, 2, 3], dtype=np.int32))

        schema_dict = {
            "type": "group",
            "members": {
                "flexible_data": {
                    "type": "dataset",
                    "anyOf": [
                        {"dtype": "int32"},
                        {"dtype": "int64"},
                        {"dtype": "float32"}
                    ],
                    "shape": [-1]
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_anyOf_group_level(self):
        """Test validation using anyOf at the group level for multiple acceptable schemas."""
        self.fid.create_dataset("data", data=np.array([1, 2, 3], dtype=np.int32))

        schema_dict = {
            "type": "group",
            "anyOf": [
                {
                    "members": {
                        "data": {
                            "type": "dataset",
                            "dtype": "int32",
                            "shape": [-1]
                        }
                    },
                    "required": ["data"]
                },
                {
                    "members": {
                        "data": {
                            "type": "dataset",
                            "dtype": "float32",
                            "shape": [-1]
                        }
                    },
                    "required": ["data"]
                }
            ]
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_anyOf_group_level_invalid(self):
        """Test invalid case for anyOf at the group level."""
        self.fid.create_dataset("data", data=np.array([1, 2, 3], dtype=np.int32))

        schema_dict = {
            "type": "group",
            "anyOf": [
                {
                    "members": {
                        "data": {
                            "type": "dataset",
                            "dtype": "float64",  # Not matching actual int32
                            "shape": [-1]
                        }
                    },
                    "required": ["data"]
                },
                {
                    "members": {
                        "data": {
                            "type": "dataset",
                            "dtype": "float32",  # Not matching actual int32
                            "shape": [-1]
                        }
                    },
                    "required": ["data"]
                }
            ]
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

if __name__ == "__main__":
    unittest.main()
