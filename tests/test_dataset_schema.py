from jsonschema.exceptions import ValidationError
import pathlib
import unittest
from hdf5schema.schema import DatasetSchema


THIS_PATH = pathlib.Path(__file__).parent.resolve()
DATA_DIR = THIS_PATH / "data"


class TestDatasetSchema(unittest.TestCase):

    def test_minimal(self):
        schema = {
            "type": "dataset",
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_mispelled_type(self):
        schema = {
            "type": "dataet",
        }
        with self.assertRaises(ValidationError):
            schema = DatasetSchema(schema, selector=None)
            schema.validate()
        self.assertTrue(True)

    def test_description(self):
        schema = {
            "type": "dataset",
            "description": "Testing",
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_dtype_scalar(self):
        schema = {
            "type": "dataset",
            "description": "Testing",
            "dtype": "uint8"
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_dtype(self):
        schema = {
            "type": "dataset",
            "description": "Testing",
            "dtype": {
                "formats": [
                    {
                        "name": "A",
                        "format": "uint8"
                    },
                    {
                        "name": "B",
                        "format": "uint8"
                    }
                ]
            }
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_dtype_offset(self):
        schema = {
            "type": "dataset",
            "description": "Testing",
            "dtype": {
                "formats": [
                    {
                        "name": "A",
                        "format": "uint8",
                        "offset": 32
                    },
                    {
                        "name": "B",
                        "format": "uint8",
                        "offset": 64
                    }
                ]
            }
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_dtype_itemsize(self):
        schema = {
            "type": "dataset",
            "description": "Testing",
            "dtype": {
                "formats": [
                    {
                        "name": "A",
                        "format": "uint8",
                        "offset": 32
                    },
                    {
                        "name": "B",
                        "format": "uint8",
                        "offset": 64
                    },
                ],
                "itemsize": 128
            }
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_invalid_dtype(self):
        schema = {
            "type": "dataset",
            "description": "Testing",
            "dtype": {
                "formats": [
                    {
                        "name": "A",
                        "format": "uint8"
                    },
                    {
                        "name": "B",
                    }
                ]
            }
        }
        with self.assertRaises(ValidationError):
            schema = DatasetSchema(schema, selector=None)
            schema.validate()
        self.assertTrue(True)

    def test_unknown_element(self):
        schema = {
            "type": "dataset",
            "unknown": "test"
        }
        with self.assertRaises(ValidationError):
            schema = DatasetSchema(schema, selector=None)
            schema.validate()
        self.assertTrue(True)

    def test_unknown_dtype_element(self):
        schema = {
            "type": "dataset",
            "description": "Testing",
            "dtype": {
                "formats": [
                    {
                        "name": "A",
                        "format": "uint8",
                        "unknown": "test"
                    }
                ]
            }
        }
        with self.assertRaises(ValidationError):
            schema = DatasetSchema(schema, selector=None)
            schema.validate()
        self.assertTrue(True)

    def test_attrs(self):
        schema = {
            "type": "dataset",
            "description": "Testing",
            "attrs": [
                {
                    "name": "A",
                    "format": "uint8"
                },
                {
                    "name": "B",
                    "format": "uint8"
                }
            ]
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_shape(self):
        schema = {
            "type": "dataset",
            "description": "Testing",
            "shape": [-1, 20, -1]
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_comment(self):
        schema = {
            "type": "dataset",
            "$comment": "This is a test dataset for validation",
            "dtype": "float64",
            "shape": [100]
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_enum_constraint(self):
        schema = {
            "type": "dataset",
            "dtype": "int32",
            "shape": [],
            "enum": [1, 2, 3, 4, 5]
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_const_constraint(self):
        schema = {
            "type": "dataset",
            "dtype": "float64",
            "shape": [],
            "const": 9.80665
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_allof_constraint(self):
        schema = {
            "type": "dataset",
            "allOf": [
                {"dtype": "float64"},
                {"shape": [-1]}
            ]
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_anyof_constraint(self):
        schema = {
            "type": "dataset",
            "anyOf": [
                {"dtype": "float32"},
                {"dtype": "float64"}
            ]
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_oneof_constraint(self):
        schema = {
            "type": "dataset",
            "oneOf": [
                {"dtype": "int32", "shape": []},
                {"dtype": "float64", "shape": [-1]}
            ]
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_not_constraint(self):
        schema = {
            "type": "dataset",
            "dtype": "float64",
            "not": {
                "const": -999.0
            }
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_complex_logical_combination(self):
        schema = {
            "type": "dataset",
            "allOf": [
                {
                    "anyOf": [
                        {"dtype": "float32"},
                        {"dtype": "float64"}
                    ]
                },
                {
                    "not": {
                        "shape": [0]
                    }
                }
            ],
            "$comment": "Dataset must be float type and not empty"
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_format_email(self):
        schema = {
            "type": "dataset",
            "dtype": "S100",
            "format": "email",
            "$comment": "Valid email address"
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_format_datetime(self):
        schema = {
            "type": "dataset",
            "dtype": "S30",
            "format": "date-time",
            "$comment": "ISO 8601 datetime format"
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_format_date(self):
        schema = {
            "type": "dataset",
            "dtype": "S12",
            "format": "date",
            "$comment": "ISO date format YYYY-MM-DD"
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_format_time(self):
        schema = {
            "type": "dataset",
            "dtype": "S15",
            "format": "time",
            "$comment": "ISO time format HH:MM:SS"
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_format_hostname(self):
        schema = {
            "type": "dataset",
            "dtype": "S255",
            "format": "hostname",
            "$comment": "Valid hostname or domain name"
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_format_ipv4(self):
        schema = {
            "type": "dataset",
            "dtype": "S16",
            "format": "ipv4",
            "$comment": "IPv4 address in dotted decimal notation"
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_format_ipv6(self):
        schema = {
            "type": "dataset",
            "dtype": "S40",
            "format": "ipv6",
            "$comment": "IPv6 address in standard notation"
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_format_uri(self):
        schema = {
            "type": "dataset",
            "dtype": "S255",
            "format": "uri",
            "$comment": "Valid URI with scheme"
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_format_uuid(self):
        schema = {
            "type": "dataset",
            "dtype": "S40",
            "format": "uuid",
            "$comment": "UUID identifier"
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_format_regex(self):
        schema = {
            "type": "dataset",
            "dtype": "S100",
            "format": "regex",
            "$comment": "Valid regex pattern string"
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_min_length_constraint(self):
        schema = {
            "type": "dataset",
            "dtype": "S100",
            "minLength": 5,
            "$comment": "String must be at least 5 characters"
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_max_length_constraint(self):
        schema = {
            "type": "dataset",
            "dtype": "S100",
            "maxLength": 50,
            "$comment": "String must be at most 50 characters"
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_min_max_length_constraints(self):
        schema = {
            "type": "dataset",
            "dtype": "S100",
            "minLength": 10,
            "maxLength": 50,
            "$comment": "String length must be between 10 and 50 characters"
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_pattern_constraint(self):
        schema = {
            "type": "dataset",
            "dtype": "S20",
            "pattern": r"^[A-Z]{2}-\d{4}$",
            "$comment": "Pattern like AB-1234"
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_combined_string_constraints(self):
        schema = {
            "type": "dataset",
            "dtype": "S100",
            "format": "email",
            "minLength": 8,
            "maxLength": 50,
            "pattern": r".*@.*\.com$",
            "$comment": "Email that ends with .com and has proper length"
        }
        schema = DatasetSchema(schema, selector=None)
        schema.validate()
        self.assertTrue(True)

    def test_invalid_format(self):
        schema = {
            "type": "dataset",
            "dtype": "S50",
            "format": "invalid_format",
            "$comment": "This should fail validation"
        }
        with self.assertRaises(ValidationError):
            schema = DatasetSchema(schema, selector=None)
            schema.validate()
        self.assertTrue(True)

    def test_negative_min_length(self):
        schema = {
            "type": "dataset",
            "dtype": "S50",
            "minLength": -5,
            "$comment": "Negative minLength should fail"
        }
        with self.assertRaises(ValidationError):
            schema = DatasetSchema(schema, selector=None)
            schema.validate()
        self.assertTrue(True)

    def test_negative_max_length(self):
        schema = {
            "type": "dataset",
            "dtype": "S50",
            "maxLength": -10,
            "$comment": "Negative maxLength should fail"
        }
        with self.assertRaises(ValidationError):
            schema = DatasetSchema(schema, selector=None)
            schema.validate()
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
