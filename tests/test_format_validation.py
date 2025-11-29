import h5py
import pathlib
import shutil
import unittest
from hdf5schema.schema import GroupSchema
from hdf5schema.validator import Hdf5Validator

THIS_PATH = pathlib.Path(__file__).parent.resolve()
DATA_DIR = THIS_PATH / "data"


class TestFormatValidation(unittest.TestCase):
    """Test format validation for string datasets."""

    def setUp(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.fid = h5py.File(DATA_DIR / "test_format.h5", "w")

    def tearDown(self):
        self.fid.close()
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)

    def clear_fid(self):
        self.fid.close()
        self.fid = h5py.File(DATA_DIR / "test_format.h5", "w")

    def test_email_format_valid(self):
        """Test valid email format validation."""
        self.fid.create_dataset("email_dataset", data=b"test@example.com", dtype="S50")
        schema_dict = {
            "type": "group",
            "members": {
                "email_dataset": {
                    "type": "dataset",
                    "dtype": "S50",
                    "format": "email"
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_email_format_invalid(self):
        """Test invalid email format validation."""
        self.fid.create_dataset("email_dataset", data=b"invalid-email", dtype="S50")
        schema_dict = {
            "type": "group",
            "members": {
                "email_dataset": {
                    "type": "dataset",
                    "dtype": "S50",
                    "format": "email"
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_timestamp_format_valid(self):
        """Test valid timestamp format validation."""
        self.fid.create_dataset("timestamp_dataset", data=b"2023-10-05T14:48:00Z", dtype="S50")
        schema_dict = {
            "type": "group",
            "members": {
                "timestamp_dataset": {
                    "type": "dataset",
                    "dtype": "S50",
                    "format": "date-time"
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_timestamp_format_invalid(self):
        """Test invalid timestamp format validation."""
        self.fid.create_dataset("timestamp_dataset", data=b"10-05-2023 14:48:00", dtype="S50")
        schema_dict = {
            "type": "group",
            "members": {
                "timestamp_dataset": {
                    "type": "dataset",
                    "dtype": "S50",
                    "format": "date-time"
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_uri_format_valid(self):
        """Test valid URI format validation."""
        self.fid.create_dataset("uri_dataset", data=b"https://example.com/path", dtype="S100")
        schema_dict = {
            "type": "group",
            "members": {
                "uri_dataset": {
                    "type": "dataset",
                    "dtype": "S100",
                    "format": "uri"
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_date_format_valid(self):
        """Test valid date format validation."""
        self.fid.create_dataset("date_dataset", data=b"2023-12-25", dtype="S20")
        schema_dict = {
            "type": "group",
            "members": {
                "date_dataset": {
                    "type": "dataset",
                    "dtype": "S20",
                    "format": "date"
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_date_format_invalid(self):
        """Test invalid date format validation."""
        self.fid.create_dataset("date_dataset", data=b"invalid-date", dtype="S20")
        schema_dict = {
            "type": "group",
            "members": {
                "date_dataset": {
                    "type": "dataset",
                    "dtype": "S20",
                    "format": "date"
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_uuid_format_valid(self):
        """Test valid UUID format validation."""
        self.fid.create_dataset("uuid_dataset", data=b"550e8400-e29b-41d4-a716-446655440000", dtype="S50")
        schema_dict = {
            "type": "group",
            "members": {
                "uuid_dataset": {
                    "type": "dataset",
                    "dtype": "S50",
                    "format": "uuid"
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_uuid_format_invalid(self):
        """Test invalid UUID format validation."""
        self.fid.create_dataset("uuid_dataset", data=b"not-a-uuid", dtype="S50")
        schema_dict = {
            "type": "group",
            "members": {
                "uuid_dataset": {
                    "type": "dataset",
                    "dtype": "S50",
                    "format": "uuid"
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_ipv4_format_valid(self):
        """Test valid IPv4 format validation."""
        self.fid.create_dataset("ip_dataset", data=b"192.168.1.1", dtype="S20")
        schema_dict = {
            "type": "group",
            "members": {
                "ip_dataset": {
                    "type": "dataset",
                    "dtype": "S20",
                    "format": "ipv4"
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_ipv4_format_invalid(self):
        """Test invalid IPv4 format validation."""
        self.fid.create_dataset("ip_dataset", data=b"999.999.999.999", dtype="S20")
        schema_dict = {
            "type": "group",
            "members": {
                "ip_dataset": {
                    "type": "dataset",
                    "dtype": "S20",
                    "format": "ipv4"
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_min_max_length_valid(self):
        """Test valid string length constraints."""
        self.fid.create_dataset("text_dataset", data=b"hello world", dtype="S50")  # 11 chars, within range
        schema_dict = {
            "type": "group",
            "members": {
                "text_dataset": {
                    "type": "dataset",
                    "dtype": "S50",
                    "minLength": 5,
                    "maxLength": 20
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertTrue(validator.is_valid())
        self.clear_fid()

    def test_min_length_invalid(self):
        """Test minimum length constraint violation."""
        self.fid.create_dataset("text_dataset", data=b"short", dtype="S50")  # 5 chars, too short
        schema_dict = {
            "type": "group",
            "members": {
                "text_dataset": {
                    "type": "dataset",
                    "dtype": "S50",
                    "minLength": 10
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()

    def test_max_length_invalid(self):
        """Test maximum length constraint violation."""
        self.fid.create_dataset("text_dataset", data=b"this is too long", dtype="S50")  # Too long
        schema_dict = {
            "type": "group",
            "members": {
                "text_dataset": {
                    "type": "dataset",
                    "dtype": "S50",
                    "maxLength": 5
                }
            }
        }
        schema = GroupSchema(schema_dict, selector=None)
        validator = Hdf5Validator(self.fid, schema)
        self.assertFalse(validator.is_valid())
        self.clear_fid()


if __name__ == "__main__":
    unittest.main()
