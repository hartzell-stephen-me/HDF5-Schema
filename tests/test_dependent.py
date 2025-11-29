import h5py
import numpy as np
import pathlib
import shutil
import unittest
from hdf5schema.schema import GroupSchema
from hdf5schema.validator import Hdf5Validator

THIS_PATH = pathlib.Path(__file__).parent.resolve()
DATA_DIR = THIS_PATH / "data"


class TestDependentValidation(unittest.TestCase):
    """Test dependentRequired and dependentSchemas validation."""

    def setUp(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.fid = h5py.File(DATA_DIR / "test_dependent.h5", "w")

    def tearDown(self):
        self.fid.close()
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)

    def clear_fid(self):
        self.fid.close()
        self.fid = h5py.File(DATA_DIR / "test_dependent.h5", "w")

    def test_dependent_required_group_valid_with_dependencies(self):
        """Test dependentRequired validation - valid case with all dependencies present."""
        # Create datasets for valid case
        self.fid.create_dataset("sensor_data", data=np.array([1.0, 2.0, 3.0]))
        self.fid.create_dataset("timestamps", data=np.array([0.0, 1.0, 2.0]))
        self.fid.create_dataset("calibration", data=1.5)
        
        schema_dict = {
            "type": "group",
            "dependentRequired": {
                "sensor_data": ["timestamps", "calibration"]
            },
            "members": {
                "sensor_data": {
                    "type": "dataset",
                    "dtype": "<f8",
                    "shape": [-1]
                },
                "timestamps": {
                    "type": "dataset", 
                    "dtype": "<f8",
                    "shape": [-1]
                },
                "calibration": {
                    "type": "dataset",
                    "dtype": "<f8", 
                    "shape": []
                }
            }
        }
        
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_dependent_required_group_invalid_missing_dependencies(self):
        """Test dependentRequired validation - invalid case with missing dependencies."""
        # Create only sensor_data, missing required dependencies
        self.fid.create_dataset("sensor_data", data=np.array([1.0, 2.0, 3.0]))
        
        schema_dict = {
            "type": "group",
            "dependentRequired": {
                "sensor_data": ["timestamps", "calibration"]
            },
            "members": {
                "sensor_data": {
                    "type": "dataset",
                    "dtype": "<f8",
                    "shape": [-1]
                },
                "timestamps": {
                    "type": "dataset", 
                    "dtype": "<f8",
                    "shape": [-1]
                },
                "calibration": {
                    "type": "dataset",
                    "dtype": "<f8", 
                    "shape": []
                }
            }
        }
        
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        
        # Check that we get the expected error messages
        errors = list(validator.iter_errors())
        self.assertGreater(len(errors), 0)
        error_messages = [str(error) for error in errors]
        self.assertTrue(any("timestamps" in msg for msg in error_messages))
        self.assertTrue(any("calibration" in msg for msg in error_messages))
        self.clear_fid()

    def test_dependent_required_group_valid_no_trigger(self):
        """Test dependentRequired validation - valid case with no triggering property."""
        # Create datasets that don't trigger dependentRequired
        self.fid.create_dataset("other_data", data=np.array([4.0, 5.0, 6.0]))
        
        schema_dict = {
            "type": "group",
            "dependentRequired": {
                "sensor_data": ["timestamps", "calibration"]
            },
            "members": {
                "other_data": {
                    "type": "dataset",
                    "dtype": "<f8",
                    "shape": [-1]
                },
                "sensor_data": {
                    "type": "dataset",
                    "dtype": "<f8",
                    "shape": [-1]
                },
                "timestamps": {
                    "type": "dataset", 
                    "dtype": "<f8",
                    "shape": [-1]
                },
                "calibration": {
                    "type": "dataset",
                    "dtype": "<f8", 
                    "shape": []
                }
            }
        }
        
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_dependent_required_multiple_triggers(self):
        """Test dependentRequired with multiple triggering properties."""
        # Create all required datasets
        self.fid.create_dataset("temperature", data=np.array([20.0, 25.0, 30.0]))
        self.fid.create_dataset("temp_units", data="celsius", dtype="S10")
        self.fid.create_dataset("temp_calibration", data=1.0)
        self.fid.create_dataset("pressure", data=np.array([1013.0, 1015.0, 1010.0]))
        self.fid.create_dataset("pressure_units", data="hPa", dtype="S10")
        
        schema_dict = {
            "type": "group",
            "dependentRequired": {
                "temperature": ["temp_units", "temp_calibration"],
                "pressure": ["pressure_units"]
            },
            "members": {
                "temperature": {"type": "dataset", "dtype": "<f8", "shape": [-1]},
                "temp_units": {"type": "dataset", "dtype": "S10", "shape": []},
                "temp_calibration": {"type": "dataset", "dtype": "<f8", "shape": []},
                "pressure": {"type": "dataset", "dtype": "<f8", "shape": [-1]},
                "pressure_units": {"type": "dataset", "dtype": "S10", "shape": []}
            }
        }
        
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_dependent_schemas_group_valid_with_dependency(self):
        """Test dependentSchemas validation - valid case with dependent schema satisfied."""
        # Create datasets that satisfy the dependent schema
        self.fid.create_dataset("experiment_type", data="control", dtype="S10")
        self.fid.create_dataset("control_data", data=np.array([1.0, 2.0, 3.0]))
        
        schema_dict = {
            "type": "group",
            "dependentSchemas": {
                "experiment_type": {
                    "type": "group",
                    "members": {
                        "control_data": {
                            "type": "dataset",
                            "dtype": "<f8",
                            "shape": [-1]
                        }
                    }
                }
            },
            "members": {
                "experiment_type": {
                    "type": "dataset",
                    "dtype": "S10",
                    "shape": []
                },
                "control_data": {
                    "type": "dataset",
                    "dtype": "<f8",
                    "shape": [-1]
                }
            }
        }
        
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_dependent_schemas_group_invalid_missing_dependency(self):
        """Test dependentSchemas validation - invalid case with dependent schema not satisfied."""
        # Create experiment_type but not control_data
        self.fid.create_dataset("experiment_type", data="control", dtype="S10")
        
        schema_dict = {
            "type": "group",
            "dependentSchemas": {
                "experiment_type": {
                    "type": "group",
                    "members": {
                        "control_data": {
                            "type": "dataset",
                            "dtype": "<f8",
                            "shape": [-1]
                        }
                    }
                }
            },
            "members": {
                "experiment_type": {
                    "type": "dataset",
                    "dtype": "S10",
                    "shape": []
                }
            }
        }
        
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_dependent_schemas_group_valid_no_trigger(self):
        """Test dependentSchemas validation - valid case with no triggering property."""
        # Create datasets that don't trigger dependentSchemas
        self.fid.create_dataset("other_data", data=np.array([4.0, 5.0, 6.0]))
        
        schema_dict = {
            "type": "group",
            "dependentSchemas": {
                "experiment_type": {
                    "type": "group",
                    "members": {
                        "control_data": {
                            "type": "dataset",
                            "dtype": "<f8",
                            "shape": [-1]
                        }
                    }
                }
            },
            "members": {
                "other_data": {
                    "type": "dataset",
                    "dtype": "<f8",
                    "shape": [-1]
                },
                "experiment_type": {
                    "type": "dataset",
                    "dtype": "S10",
                    "shape": []
                }
            }
        }
        
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_dependent_required_dataset_valid_with_attributes(self):
        """Test dependentRequired validation for datasets - valid case with dependent attributes."""
        # Create dataset with all required attributes
        ds = self.fid.create_dataset("sensor_data", data=np.array([1.0, 2.0, 3.0]))
        ds.attrs["units"] = b"meters"
        ds.attrs["calibration_date"] = b"2023-01-01"
        ds.attrs["precision"] = 0.01
        
        schema_dict = {
            "type": "group",
            "members": {
                "sensor_data": {
                    "type": "dataset",
                    "dtype": "<f8",
                    "shape": [-1],
                    "dependentRequired": {
                        "units": ["calibration_date", "precision"]
                    },
                    "attrs": [
                        {"name": "units", "dtype": "S20"},
                        {"name": "calibration_date", "dtype": "S20"},
                        {"name": "precision", "dtype": "<f8"}
                    ]
                }
            }
        }
        
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_dependent_required_dataset_invalid_missing_attributes(self):
        """Test dependentRequired validation for datasets - invalid case with missing dependent attributes."""
        # Create dataset with units but missing dependent attributes
        ds = self.fid.create_dataset("sensor_data", data=np.array([1.0, 2.0, 3.0]))
        ds.attrs["units"] = "meters"
        
        schema_dict = {
            "type": "group",
            "members": {
                "sensor_data": {
                    "type": "dataset",
                    "dtype": "<f8",
                    "shape": [-1],
                    "dependentRequired": {
                        "units": ["calibration_date", "precision"]
                    }
                }
            }
        }
        
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        
        # Check error messages
        errors = list(validator.iter_errors())
        self.assertGreater(len(errors), 0)
        error_messages = [str(error) for error in errors]
        self.assertTrue(any("calibration_date" in msg for msg in error_messages))
        self.assertTrue(any("precision" in msg for msg in error_messages))
        self.clear_fid()

    def test_dependent_schemas_dataset_valid_with_dependency(self):
        """Test dependentSchemas validation for datasets - valid case using conditional attributes."""
        # Create dataset with attribute that triggers dependent validation
        ds = self.fid.create_dataset("measurement", data=np.array([1.0, 2.0, 3.0]))
        ds.attrs["type"] = b"calibrated"
        ds.attrs["calibration_factor"] = 1.5  # Required when type="calibrated"
        
        # For datasets, use dependentRequired on attributes instead of dependentSchemas
        schema_dict = {
            "type": "group", 
            "members": {
                "measurement": {
                    "type": "dataset",
                    "dtype": "<f8",
                    "shape": [-1],
                    "dependentRequired": {
                        "type": ["calibration_factor"]
                    },
                    "attrs": [
                        {"name": "type", "dtype": "S20"},
                        {"name": "calibration_factor", "dtype": "<f8"}
                    ]
                }
            }
        }
        
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_dependent_required_nested_groups(self):
        """Test dependentRequired validation in nested groups."""
        # Create nested structure
        experiment_grp = self.fid.create_group("experiment")
        experiment_grp.create_dataset("config", data="standard", dtype="S20")
        experiment_grp.create_dataset("parameters", data=np.array([1.0, 2.0]))
        experiment_grp.create_dataset("validation_data", data=np.array([0.1, 0.2]))
        
        schema_dict = {
            "type": "group",
            "members": {
                "experiment": {
                    "type": "group",
                    "dependentRequired": {
                        "config": ["parameters", "validation_data"]
                    },
                    "members": {
                        "config": {"type": "dataset", "dtype": "S20", "shape": []},
                        "parameters": {"type": "dataset", "dtype": "<f8", "shape": [-1]},
                        "validation_data": {"type": "dataset", "dtype": "<f8", "shape": [-1]}
                    }
                }
            }
        }
        
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_dependent_schemas_complex_dependency(self):
        """Test dependentSchemas with complex nested requirements."""
        # Create complex nested structure
        analysis_grp = self.fid.create_group("analysis")
        analysis_grp.create_dataset("method", data="ml_prediction", dtype="S20")
        
        # Create the structure required by the dependent schema
        model_grp = analysis_grp.create_group("model")
        model_grp.create_dataset("weights", data=np.random.rand(10, 5))
        model_grp.create_dataset("biases", data=np.random.rand(5))
        
        schema_dict = {
            "type": "group",
            "members": {
                "analysis": {
                    "type": "group",
                    "dependentSchemas": {
                        "method": {
                            "type": "group",
                            "members": {
                                "model": {
                                    "type": "group",
                                    "members": {
                                        "weights": {"type": "dataset", "dtype": "<f8", "shape": [-1, -1]},
                                        "biases": {"type": "dataset", "dtype": "<f8", "shape": [-1]}
                                    }
                                }
                            }
                        }
                    },
                    "members": {
                        "method": {"type": "dataset", "dtype": "S20", "shape": []},
                        "model": {
                            "type": "group",
                            "members": {
                                "weights": {"type": "dataset", "dtype": "<f8", "shape": [-1, -1]},
                                "biases": {"type": "dataset", "dtype": "<f8", "shape": [-1]}
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

    def test_combined_dependent_required_and_schemas(self):
        """Test using both dependentRequired and dependentSchemas together."""
        # Create structure with both types of dependencies
        self.fid.create_dataset("sensor_type", data="temperature", dtype="S20")
        self.fid.create_dataset("sensor_id", data="T001", dtype="S10")  # Required by dependentRequired
        self.fid.create_dataset("calibration_curve", data=np.array([1.0, 1.1, 1.2]))  # Required by dependentSchemas
        
        schema_dict = {
            "type": "group",
            "dependentRequired": {
                "sensor_type": ["sensor_id"]
            },
            "dependentSchemas": {
                "sensor_type": {
                    "type": "group",
                    "members": {
                        "calibration_curve": {
                            "type": "dataset",
                            "dtype": "<f8",
                            "shape": [-1]
                        }
                    }
                }
            },
            "members": {
                "sensor_type": {"type": "dataset", "dtype": "S20", "shape": []},
                "sensor_id": {"type": "dataset", "dtype": "S10", "shape": []},
                "calibration_curve": {"type": "dataset", "dtype": "<f8", "shape": [-1]}
            }
        }
        
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_dependent_required_empty_list(self):
        """Test dependentRequired with empty dependency list."""
        self.fid.create_dataset("trigger", data="active", dtype="S10")
        
        schema_dict = {
            "type": "group",
            "dependentRequired": {
                "trigger": []  # Empty list - no dependencies required
            },
            "members": {
                "trigger": {"type": "dataset", "dtype": "S10", "shape": []}
            }
        }
        
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_dependent_validation_with_pattern_members(self):
        """Test dependent validation combined with pattern members."""
        # Create pattern-matching structure
        sensors_grp = self.fid.create_group("sensors")
        sensor1_grp = sensors_grp.create_group("sensor_001")
        sensor1_grp.create_dataset("type", data="temperature", dtype="S20")
        sensor1_grp.create_dataset("calibration", data=1.5)  # Required by dependentRequired
        
        schema_dict = {
            "type": "group",
            "members": {
                "sensors": {
                    "type": "group",
                    "patternMembers": {
                        "^sensor_[0-9]+$": {
                            "type": "group",
                            "dependentRequired": {
                                "type": ["calibration"]
                            },
                            "members": {
                                "type": {"type": "dataset", "dtype": "S20", "shape": []},
                                "calibration": {"type": "dataset", "dtype": "<f8", "shape": []}
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


if __name__ == "__main__":
    unittest.main()
