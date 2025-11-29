from jsonschema.exceptions import ValidationError
import pathlib
import unittest
from hdf5schema.schema import GroupSchema


THIS_PATH = pathlib.Path(__file__).parent.resolve()
DATA_DIR = THIS_PATH / "data"


class TestGroupSchema(unittest.TestCase):

    def test_minimal(self):
        schema = {
            "type": "group",
            "members": {},
        }
        grp_schema = GroupSchema(schema, selector=None)
        grp_schema.validate()
        self.assertTrue(True)

    def test_mispelled_type(self):
        schema = {
            "type": "grp",
            "members": {},
        }
        with self.assertRaises(ValidationError):
            grp_schema = GroupSchema(schema, selector=None)
            grp_schema.validate()
        self.assertTrue(True)

    def test_description(self):
        schema = {
            "type": "group",
            "members": {},
            "description": "Testing",
        }
        grp_schema = GroupSchema(schema, selector=None)
        grp_schema.validate()
        self.assertTrue(True)

    def test_unknown_element(self):
        schema = {
            "type": "dataset",
            "members": {},
            "unknown": "test"
        }
        with self.assertRaises(ValidationError):
            grp_schema = GroupSchema(schema, selector=None)
            grp_schema.validate()
        self.assertTrue(True)

    def test_attrs(self):
        schema = {
            "type": "group",
            "description": "Testing",
            "members": {},
            "attrs": [
                {
                    "name": "A",
                    "dtype": "uint8"
                },
                {
                    "name": "B",
                    "dtype": "uint8"
                }
            ]
        }
        grp_schema = GroupSchema(schema, selector=None)
        grp_schema.validate()
        self.assertTrue(True)

    def test_comment(self):
        schema = {
            "type": "group",
            "$comment": "Root group containing experimental data",
            "members": {
                "data": {
                    "type": "dataset",
                    "dtype": "float64",
                    "shape": [-1]
                }
            }
        }
        grp_schema = GroupSchema(schema, selector=None)
        grp_schema.validate()
        self.assertTrue(True)

    def test_enum_constraint(self):
        schema = {
            "type": "group",
            "members": {},
            "enum": ["sensor_1", "sensor_2", "control"]
        }
        grp_schema = GroupSchema(schema, selector=None)
        grp_schema.validate()
        self.assertTrue(True)

    def test_const_constraint(self):
        schema = {
            "type": "group",
            "members": {},
            "const": "metadata"
        }
        grp_schema = GroupSchema(schema, selector=None)
        grp_schema.validate()
        self.assertTrue(True)

    def test_allof_constraint(self):
        schema = {
            "type": "group",
            "allOf": [
                {
                    "members": {
                        "temperature": {
                            "type": "dataset",
                            "dtype": "float64"
                        }
                    }
                },
                {
                    "members": {
                        "pressure": {
                            "type": "dataset",
                            "dtype": "float64"
                        }
                    }
                }
            ]
        }
        grp_schema = GroupSchema(schema, selector=None)
        grp_schema.validate()
        self.assertTrue(True)

    def test_anyof_constraint(self):
        schema = {
            "type": "group",
            "anyOf": [
                {
                    "members": {
                        "raw_data": {
                            "type": "dataset",
                            "dtype": "float64"
                        }
                    }
                },
                {
                    "members": {
                        "processed_data": {
                            "type": "dataset",
                            "dtype": "float64"
                        }
                    }
                }
            ]
        }
        grp_schema = GroupSchema(schema, selector=None)
        grp_schema.validate()
        self.assertTrue(True)

    def test_oneof_constraint(self):
        schema = {
            "type": "group",
            "oneOf": [
                {
                    "members": {
                        "format_a": {
                            "type": "dataset",
                            "dtype": "float32"
                        }
                    }
                },
                {
                    "members": {
                        "format_b": {
                            "type": "dataset",
                            "dtype": "float64"
                        }
                    }
                }
            ]
        }
        grp_schema = GroupSchema(schema, selector=None)
        grp_schema.validate()
        self.assertTrue(True)

    def test_not_constraint(self):
        schema = {
            "type": "group",
            "members": {
                "valid_data": {
                    "type": "dataset",
                    "dtype": "float64"
                }
            },
            "not": {
                "members": {
                    "deprecated_field": {
                        "type": "dataset",
                        "dtype": "float32"
                    }
                }
            }
        }
        grp_schema = GroupSchema(schema, selector=None)
        grp_schema.validate()
        self.assertTrue(True)

    def test_nested_logical_operators(self):
        schema = {
            "type": "group",
            "$comment": "Complex validation with nested logical operators",
            "allOf": [
                {
                    "anyOf": [
                        {
                            "members": {
                                "sensor_data": {
                                    "type": "dataset",
                                    "dtype": "float64"
                                }
                            }
                        },
                        {
                            "members": {
                                "reference_data": {
                                    "type": "dataset",
                                    "dtype": "float64"
                                }
                            }
                        }
                    ]
                },
                {
                    "not": {
                        "members": {
                            "invalid_marker": {
                                "type": "dataset"
                            }
                        }
                    }
                }
            ]
        }
        grp_schema = GroupSchema(schema, selector=None)
        grp_schema.validate()
        self.assertTrue(True)

    def test_pattern_members_with_constraints(self):
        schema = {
            "type": "group",
            "$comment": "Group with pattern members and constraints",
            "patternMembers": {
                "^sensor_[0-9]+$": {
                    "type": "group",
                    "allOf": [
                        {
                            "members": {
                                "readings": {
                                    "type": "dataset",
                                    "dtype": "float64"
                                }
                            }
                        },
                        {
                            "members": {
                                "timestamps": {
                                    "type": "dataset",
                                    "dtype": "float64"
                                }
                            }
                        }
                    ]
                }
            },
            "enum": ["experiment_1", "experiment_2"]
        }
        grp_schema = GroupSchema(schema, selector=None)
        grp_schema.validate()
        self.assertTrue(True)

    def test_conditional_validation_if_then_else(self):
        schema = {
            "type": "group",
            "members": {
                "sensor_data": {
                    "type": "dataset",
                    "dtype": "float64",
                    "shape": [-1],
                    "if": {
                        "attrs": [
                            {"name": "sensor_type", "const": "temperature"}
                        ]
                    },
                    "then": {
                        "$comment": "Temperature sensors must have specific range",
                        "attrs": [
                            {"name": "min_value", "dtype": "float64", "required": True},
                            {"name": "max_value", "dtype": "float64", "required": True},
                            {"name": "units", "const": "celsius", "required": True}
                        ]
                    },
                    "else": {
                        "$comment": "Other sensors have different requirements",
                        "attrs": [
                            {"name": "calibration_factor", "dtype": "float64", "required": True}
                        ]
                    }
                }
            }
        }
        grp_schema = GroupSchema(schema, selector=None)
        grp_schema.validate()
        self.assertTrue(True)

    def test_nested_conditional_validation(self):
        schema = {
            "type": "group",
            "members": {
                "geolocation": {
                    "type": "group",
                    "if": {
                        "attrs": [
                            {"name": "grounded", "const": True}
                        ]
                    },
                    "then": {
                        "$comment": "Grounded sensors need location data",
                        "members": {
                            "latitude": {"type": "dataset", "dtype": "float64"},
                            "longitude": {"type": "dataset", "dtype": "float64"}
                        },
                        "required": ["latitude", "longitude"]
                    },
                    "else": {
                        "if": {
                            "attrs": [
                                {"name": "mobile", "const": True}
                            ]
                        },
                        "then": {
                            "$comment": "Mobile sensors need trajectory data",
                            "members": {
                                "trajectory": {"type": "dataset", "dtype": "float64", "shape": [-1, 3]}
                            }
                        }
                    }
                }
            }
        }
        grp_schema = GroupSchema(schema, selector=None)
        grp_schema.validate()
        self.assertTrue(True)

    def test_group_with_format_validation_datasets(self):
        schema = {
            "type": "group",
            "$comment": "Group containing datasets with format validation",
            "members": {
                "contact_info": {
                    "type": "group",
                    "members": {
                        "email": {
                            "type": "dataset",
                            "dtype": "S100",
                            "format": "email",
                            "$comment": "Valid email address"
                        },
                        "website": {
                            "type": "dataset",
                            "dtype": "S255",
                            "format": "uri",
                            "$comment": "Valid website URL"
                        },
                        "server_ip": {
                            "type": "dataset",
                            "dtype": "S16",
                            "format": "ipv4",
                            "$comment": "IPv4 address"
                        }
                    }
                },
                "timestamps": {
                    "type": "group",
                    "members": {
                        "created_at": {
                            "type": "dataset",
                            "dtype": "S30",
                            "format": "date-time",
                            "$comment": "ISO 8601 datetime"
                        },
                        "experiment_date": {
                            "type": "dataset",
                            "dtype": "S12",
                            "format": "date",
                            "$comment": "Experiment date"
                        }
                    }
                },
                "identifiers": {
                    "type": "group",
                    "members": {
                        "session_id": {
                            "type": "dataset",
                            "dtype": "S40",
                            "format": "uuid",
                            "$comment": "Unique session identifier"
                        },
                        "hostname": {
                            "type": "dataset",
                            "dtype": "S255",
                            "format": "hostname",
                            "$comment": "Server hostname"
                        }
                    }
                }
            }
        }
        grp_schema = GroupSchema(schema, selector=None)
        grp_schema.validate()
        self.assertTrue(True)

    def test_group_with_string_length_constraints(self):
        schema = {
            "type": "group",
            "$comment": "Group with datasets having string length constraints",
            "members": {
                "user_data": {
                    "type": "group",
                    "members": {
                        "username": {
                            "type": "dataset",
                            "dtype": "S50",
                            "minLength": 3,
                            "maxLength": 20,
                            "$comment": "Username between 3-20 characters"
                        },
                        "description": {
                            "type": "dataset",
                            "dtype": "S500",
                            "maxLength": 250,
                            "$comment": "Optional description up to 250 characters"
                        },
                        "short_code": {
                            "type": "dataset",
                            "dtype": "S10",
                            "minLength": 5,
                            "maxLength": 8,
                            "pattern": r"^[A-Z0-9]+$",
                            "$comment": "Alphanumeric code 5-8 characters"
                        }
                    }
                }
            }
        }
        grp_schema = GroupSchema(schema, selector=None)
        grp_schema.validate()
        self.assertTrue(True)

    def test_group_with_pattern_validation(self):
        schema = {
            "type": "group",
            "$comment": "Group with datasets using pattern validation",
            "members": {
                "validation_codes": {
                    "type": "group",
                    "members": {
                        "product_code": {
                            "type": "dataset",
                            "dtype": "S20",
                            "pattern": r"^[A-Z]{2}-\d{4}$",
                            "$comment": "Product code format like AB-1234"
                        },
                        "batch_number": {
                            "type": "dataset",
                            "dtype": "S15",
                            "pattern": r"^\d{4}-\d{2}-\d{3}$",
                            "$comment": "Batch number format YYYY-MM-NNN"
                        },
                        "serial_number": {
                            "type": "dataset",
                            "dtype": "S30",
                            "pattern": r"^SN[A-Z0-9]{8,12}$",
                            "$comment": "Serial number starting with SN"
                        }
                    }
                }
            }
        }
        grp_schema = GroupSchema(schema, selector=None)
        grp_schema.validate()
        self.assertTrue(True)

    def test_complex_format_validation_with_conditionals(self):
        schema = {
            "type": "group",
            "$comment": "Complex schema with conditionals and format validation",
            "members": {
                "experimental_data": {
                    "type": "group",
                    "if": {
                        "attrs": [
                            {"name": "data_source", "const": "external"}
                        ]
                    },
                    "then": {
                        "$comment": "External data requires URLs and validation",
                        "members": {
                            "source_url": {
                                "type": "dataset",
                                "dtype": "S500",
                                "format": "uri",
                                "minLength": 10,
                                "$comment": "Valid source URL"
                            },
                            "checksum": {
                                "type": "dataset",
                                "dtype": "S64",
                                "pattern": r"^[a-fA-F0-9]{64}$",
                                "$comment": "SHA-256 checksum"
                            }
                        }
                    },
                    "else": {
                        "$comment": "Internal data has different requirements",
                        "members": {
                            "internal_id": {
                                "type": "dataset",
                                "dtype": "S40",
                                "format": "uuid",
                                "$comment": "Internal unique identifier"
                            }
                        }
                    }
                }
            }
        }
        grp_schema = GroupSchema(schema, selector=None)
        grp_schema.validate()
        self.assertTrue(True)

    def test_group_format_validation_schema_properties(self):
        """Test that the GroupSchema class properly exposes format validation properties"""
        schema = {
            "type": "group",
            "members": {
                "test_dataset": {
                    "type": "dataset",
                    "dtype": "S50",
                    "format": "email",
                    "minLength": 5,
                    "maxLength": 100,
                    "pattern": r".*@.*"
                }
            }
        }
        grp_schema = GroupSchema(schema, selector=None)
        grp_schema.validate()
        
        # Check that the dataset member has the format properties
        # Members is a list, so we need to find the right member by checking selector
        dataset_member = None
        for member in grp_schema.members:
            if hasattr(member, 'selector') and member.selector.match('test_dataset'):
                dataset_member = member
                break
        
        self.assertIsNotNone(dataset_member)
        self.assertTrue(dataset_member.has_format())
        self.assertEqual(dataset_member.format, "email")
        self.assertTrue(dataset_member.has_min_length())
        self.assertEqual(dataset_member.min_length, 5)
        self.assertTrue(dataset_member.has_max_length())
        self.assertEqual(dataset_member.max_length, 100)
        self.assertTrue(dataset_member.has_pattern())
        self.assertEqual(dataset_member.pattern, r".*@.*")
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
