import h5py
import numpy as np
import pathlib
import shutil
import unittest
from hdf5schema.schema import GroupSchema
from hdf5schema.validator import Hdf5Validator

THIS_PATH = pathlib.Path(__file__).parent.resolve()
DATA_DIR = THIS_PATH / "data"

class TestConditionalValidation(unittest.TestCase):

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

    def test_conditional_group_if_then_else(self):
        """Test conditional validation (if/then/else) for groups."""
        # Create a group with grounded=True attribute
        grp = self.fid.create_group("geolocation")
        grp.attrs["grounded"] = True
        grp.attrs["mover"] = True

        # Add location dataset (required when grounded=True)
        grp.create_dataset("location", data=np.array([1.0, 2.0], dtype=np.float32))
        grp.create_dataset("time", data=np.array([0.0], dtype=np.float64))

        schema_dict = {
            "type": "group",
            "members": {
                "geolocation": {
                    "type": "group",
                    "if": {
                        "attrs": [
                            {"name": "grounded", "dtype": "bool", "const": True}
                        ]
                    },
                    "then": {
                        "members": {
                            "location": {
                                "type": "dataset",
                                "dtype": "<f4",
                                "shape": [2]
                            },
                            "time": {
                                "type": "dataset",
                                "dtype": "<f8",
                                "shape": [1]
                            }
                        },
                        "required": ["location", "time"]
                    },
                    "else": {
                        "members": {
                            "position": {
                                "type": "dataset",
                                "dtype": "<f4",
                                "shape": [-1, 3]
                            }
                        },
                        "required": ["position"]
                    }
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_conditional_dataset_if_then_else(self):
        """Test conditional validation (if/then/else) for datasets."""
        # Create a float32 dataset
        dset = self.fid.create_dataset("data", data=np.array([1.0, 2.0, 3.0], dtype=np.float32))
        dset.attrs["units"] = b"meters"  # Use bytes to match typical HDF5 string storage

        schema_dict = {
            "type": "group",
            "members": {
                "data": {
                    "type": "dataset",
                    "dtype": "<f4",  # Basic dtype constraint
                    "shape": [3],    # Basic shape constraint
                    "if": {
                        "dtype": "<f4"
                    },
                    "then": {
                        "attrs": [
                            {"name": "units", "dtype": "S6", "required": True}
                        ]
                    },
                    "else": {
                        "shape": []  # scalar if not float32
                    }
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_conditional_else_branch(self):
        """Test conditional validation chooses else branch when if condition fails."""
        # Create a group with grounded=False attribute
        grp = self.fid.create_group("geolocation")
        grp.attrs["grounded"] = False
        grp.attrs["mover"] = True

        # Add position dataset (required when grounded=False)
        grp.create_dataset("position", data=np.array([[1.0, 2.0, 3.0]], dtype=np.float32))

        schema_dict = {
            "type": "group",
            "members": {
                "geolocation": {
                    "type": "group",
                    "if": {
                        "attrs": [
                            {"name": "grounded", "dtype": "bool", "const": True}
                        ]
                    },
                    "then": {
                        "members": {
                            "location": {
                                "type": "dataset",
                                "dtype": "<f4",
                                "shape": [2]
                            }
                        },
                        "required": ["location"]
                    },
                    "else": {
                        "members": {
                            "position": {
                                "type": "dataset",
                                "dtype": "<f4",
                                "shape": [-1, 3]
                            }
                        },
                        "required": ["position"]
                    }
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_conditional_if_true_invalid(self):
        """Test conditional validation fails when if condition is true but then schema is not met."""
        # Create a group with grounded=True attribute
        grp = self.fid.create_group("geolocation")
        grp.attrs["grounded"] = True
        grp.attrs["mover"] = True

        # Missing location and time datasets which are required when grounded=True

        schema_dict = {
            "type": "group",
            "members": {
                "geolocation": {
                    "type": "group",
                    "if": {
                        "attrs": [
                            {"name": "grounded", "dtype": "bool", "const": True}
                        ]
                    },
                    "then": {
                        "members": {
                            "location": {
                                "type": "dataset",
                                "dtype": "<f4",
                                "shape": [2]
                            },
                            "time": {
                                "type": "dataset",
                                "dtype": "<f8",
                                "shape": [1]
                            }
                        },
                        "required": ["location", "time"]
                    },
                    "else": {
                        "members": {
                            "position": {
                                "type": "dataset",
                                "dtype": "<f4",
                                "shape": [-1, 3]
                            }
                        },
                        "required": ["position"]
                    }
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())  # Should fail because then schema is not met
        self.clear_fid()

    def test_conditional_if_false_invalid(self):
        """Test conditional validation fails when if condition is false but else schema is not met."""
        # Create a group with grounded=False attribute
        grp = self.fid.create_group("geolocation")
        grp.attrs["grounded"] = False
        grp.attrs["mover"] = True

        # Missing position dataset which is required when grounded=False

        schema_dict = {
            "type": "group",
            "members": {
                "geolocation": {
                    "type": "group",
                    "if": {
                        "attrs": [
                            {"name": "grounded", "dtype": "bool", "const": True}
                        ]
                    },
                    "then": {
                        "members": {
                            "location": {
                                "type": "dataset",
                                "dtype": "<f4",
                                "shape": [2]
                            }
                        },
                        "required": ["location"]
                    },
                    "else": {
                        "members": {
                            "position": {
                                "type": "dataset",
                                "dtype": "<f4",
                                "shape": [-1, 3]
                            }
                        },
                        "required": ["position"]
                    }
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())  # Should fail because else schema is not met
        self.clear_fid()

    def test_nested_if_then_else(self):
        """Test nested conditional validation (if/then/else)."""
        # Create a group with grounded=True and mover=True attributes
        grp = self.fid.create_group("geolocation")
        grp.attrs["grounded"] = True
        grp.attrs["mover"] = True

        # Add location and time datasets (required when grounded=True)
        grp.create_dataset("location", data=np.array([1.0, 2.0], dtype=np.float32))
        grp.create_dataset("time", data=np.array([0.0], dtype=np.float64))

        schema_dict = {
            "type": "group",
            "members": {
                "geolocation": {
                    "type": "group",
                    "if": {
                        "attrs": [
                            {"name": "grounded", "dtype": "bool", "const": True}
                        ]
                    },
                    "then": {
                        "if": {
                            "attrs": [
                                {"name": "mover", "dtype": "bool", "const": True}
                            ]
                        },
                        "then": {
                            "members": {
                                "location": {
                                    "type": "dataset",
                                    "dtype": "<f4",
                                    "shape": [2]
                                },
                                "time": {
                                    "type": "dataset",
                                    "dtype": "<f8",
                                    "shape": [1]
                                }
                            },
                            "required": ["location", "time"]
                        },
                        "else": {
                            "members": {
                                "location": {
                                    "type": "dataset",
                                    "dtype": "<f4",
                                    "shape": [2]
                                }
                            },
                            "required": ["location"]
                        }
                    },
                    "else": {
                        "members": {
                            "position": {
                                "type": "dataset",
                                "dtype": "<f4",
                                "shape": [-1, 3]
                            }
                        },
                        "required": ["position"]
                    }
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_multiple_attributes_if(self):
        """Test conditional validation with multiple attribute conditions in if."""
        # Create a group with grounded=True and mover=True attributes
        grp = self.fid.create_group("geolocation")
        grp.attrs["grounded"] = True
        grp.attrs["mover"] = True

        # Add location and time datasets (required when grounded=True and mover=True)
        grp.create_dataset("location", data=np.array([1.0, 2.0], dtype=np.float32))
        grp.create_dataset("time", data=np.array([0.0], dtype=np.float64))

        schema_dict = {
            "type": "group",
            "members": {
                "geolocation": {
                    "type": "group",
                    "if": {
                        "attrs": [
                            {"name": "grounded", "dtype": "bool", "const": True},
                            {"name": "mover", "dtype": "bool", "const": True}
                        ]
                    },
                    "then": {
                        "members": {
                            "location": {
                                "type": "dataset",
                                "dtype": "<f4",
                                "shape": [2]
                            },
                            "time": {
                                "type": "dataset",
                                "dtype": "<f8",
                                "shape": [1]
                            }
                        },
                        "required": ["location", "time"]
                    },
                    "else": {
                        "members": {
                            "position": {
                                "type": "dataset",
                                "dtype": "<f4",
                                "shape": [-1, 3]
                            }
                        },
                        "required": ["position"]
                    }
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_dataset_conditional(self):
        """Test conditional validation directly on a dataset."""
        # Create a float32 dataset with units attribute
        dset = self.fid.create_dataset("data", data=np.array([1.0, 2.0, 3.0], dtype=np.float32))
        dset.attrs["units"] = b"meters"  # Use bytes to match typical HDF5 string storage

        schema_dict = {
            "type": "group",
            "members": {
                "data": {
                    "type": "dataset",
                    "dtype": "<f4",  # Basic dtype constraint
                    "shape": [3],    # Basic shape constraint
                    "if": {
                        "dtype": "<f4"
                    },
                    "then": {
                        "attrs": [
                            {"name": "units", "dtype": "S6", "required": True}
                        ]
                    },
                    "else": {
                        "shape": []  # scalar if not float32
                    }
                }
            }
        }

        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

if __name__ == "__main__":
    unittest.main()
