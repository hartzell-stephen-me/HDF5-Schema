from dataclasses import dataclass
import h5py
import numpy as np
import pathlib
import re
import uuid
from datetime import datetime
from typing import List, Union
from hdf5schema.exceptions import ValidationError
from hdf5schema.schema import GroupSchema, DatasetSchema, RefSchema
import contextlib


@dataclass
class Hdf5Validator:

    instance: Union[pathlib.Path, str, h5py.File, h5py.Group]
    schema: Union[pathlib.Path, dict, GroupSchema]
    _iter_errors: bool = False

    def __post_init__(self):
        if not (isinstance(self.instance, (h5py.File, h5py.Group))):
            self.instance = h5py.File(self.instance, "r")

        if isinstance(self.schema, (str, pathlib.Path)):
            import json
            with open(self.schema) as f:
                schema_dict = json.load(f)
            self.schema = GroupSchema(schema_dict, selector=None)

    def _handle_error(
        self,
        error: ValidationError
    ):
        # Ensure file is closed on error
        if hasattr(self, "_opened_file") and self._opened_file is not None:
            with contextlib.suppress(Exception):
                self._opened_file.close()
            self._opened_file = None
        if self._iter_errors:
            self._errors.append(error)
        else:
            raise error

    def _dtypes_compatible(self, actual_dtype: np.dtype, expected_dtype: np.dtype) -> bool:
        """
        Check if two dtypes are compatible for validation purposes.

        Parameters
        ----------
        actual_dtype : np.dtype
            The actual dtype found in the HDF5 file
        expected_dtype : np.dtype
            The expected dtype from the schema

        Returns
        -------
        bool
            True if dtypes are compatible, False otherwise

        """
        if actual_dtype == expected_dtype:
            return True

        # Check if they have the same kind and size
        if actual_dtype.kind == expected_dtype.kind and actual_dtype.itemsize == expected_dtype.itemsize:
            return True

        # Check for some common equivalent types
        if actual_dtype.kind in ("i", "u") and expected_dtype.kind in ("i", "u"):
            # For integers, be strict about signedness and size
            return actual_dtype.kind == expected_dtype.kind and actual_dtype.itemsize == expected_dtype.itemsize

        if actual_dtype.kind == "f" and expected_dtype.kind == "f":
            # For floats, allow same precision
            return actual_dtype.itemsize == expected_dtype.itemsize

        # For strings, check if both are string types
        if actual_dtype.kind in ("S", "U") and expected_dtype.kind in ("S", "U"):
            # Both are string types - allow compatibility between byte strings (S) and unicode strings (U)
            # For fixed-length strings, allow actual length to be less than or equal to expected length
            if actual_dtype.kind == "S" and expected_dtype.kind == "U":
                # Byte string to Unicode
                return actual_dtype.itemsize <= expected_dtype.itemsize // 4  # Unicode uses 4 bytes per char
            elif actual_dtype.kind == "U" and expected_dtype.kind == "S":
                # Unicode to byte string
                return actual_dtype.itemsize // 4 <= expected_dtype.itemsize  # Unicode uses 4 bytes per char
            else:
                # Same string type - allow actual to be less than or equal to expected
                return actual_dtype.itemsize <= expected_dtype.itemsize

        return False

    def _validate(
        self,
        item: Union[h5py.Group, h5py.Dataset],
        schema: Union[GroupSchema, DatasetSchema],
    ) -> bool:
        """
        Validate an HDF5 item against that item's schema.

        Parameters
        ----------
        item: Union[h5py.Group, h5py.Dataset]
            HDF5 item to validate
        schema: Union[GroupSchema, DatasetSchema]
            Schema to validate against

        Returns
        -------
        bool
            True if valid, False otherwise

        """
        if isinstance(schema, dict):
            # Create the schema
            if schema["type"] == "object" or schema["type"] == "group":
                schema = GroupSchema(schema, selector=None)
            elif schema["type"] == "dataset":
                schema = DatasetSchema(schema, selector=None)
            else:
                raise ValueError("Recieved unknown schema type {}".format(schema["type"]))

        # Handle RefSchema first - resolve it then validate with resolved schema
        if isinstance(schema, RefSchema):
            resolved_schema = schema.resolve(max_depth=10)
            return self._validate(item, resolved_schema)

        # Now do type compatibility checks
        if ((type(item) == h5py.Dataset) and (type(schema) != DatasetSchema)):
            self._handle_error(ValidationError(f"{item.name} is not a Dataset"))
        elif ((type(item) == h5py.Group) and (type(schema) != GroupSchema)):
            self._handle_error(ValidationError(f"{item.name} is not a Group"))

        # Dispatch to appropriate validation method
        if isinstance(schema, GroupSchema):
            return self._validate_group(item, schema)
        elif isinstance(schema, DatasetSchema):
            return self._validate_dataset(item, schema)
        else:
            raise ValueError(f"Recieved unknown schema type {type(schema)}")

    def __handle_shape_dataset(self, dataset: h5py.Dataset, dataset_schema: DatasetSchema) -> bool:
        """
        Handle shape validation for a dataset by ensuring the dataset shape matches the schema shape.
        """
        if dataset_schema.shape is None:
            return False  # No shape constraint to validate against

        if len(dataset.shape) != len(dataset_schema.shape):
            self._handle_error(ValidationError(f"{dataset.name} shape {dataset.shape} does not match the schema shape {dataset_schema.shape}"))
            return True
        else:
            for axis, axis_size in enumerate(dataset.shape):
                if (dataset_schema.shape[axis] != -1) and (axis_size != dataset_schema.shape[axis]):
                    self._handle_error(ValidationError(f"{dataset.name} shape {dataset.shape} does not match the schema shape {dataset_schema.shape}"))
                    return True
        return False

    def __handle_enum_dataset(self, dataset: h5py.Dataset, dataset_schema: DatasetSchema) -> bool:
        """
        Handle enum validation for a dataset by checking dataset values against allowed values.
        """
        # Check if all dataset values are in the allowed enum values
        try:
            # Handle scalar vs array datasets differently for reading data
            if dataset.shape == ():  # Scalar dataset
                data_value = dataset[()]
                if data_value not in dataset_schema.enum:
                    self._handle_error(ValidationError(f"Dataset {dataset.name} value {data_value} not in allowed enum values {dataset_schema.enum}"))
                    return True
            else:  # Array dataset
                data_values = dataset[:]
                # For array datasets, check if all unique values are in enum
                unique_values = np.unique(data_values)
                for val in unique_values:
                    if val not in dataset_schema.enum:
                        self._handle_error(ValidationError(f"Dataset {dataset.name} contains value {val} not in allowed enum values {dataset_schema.enum}"))
                        return True
        except Exception:
            # Log but don't fail validation
            pass

        return False

    def __handle_const_dataset(self, dataset: h5py.Dataset, dataset_schema: DatasetSchema) -> bool:
        """
        Handle const validation for a dataset by ensuring all dataset values match the const value.
        """
        const_value = dataset_schema.const
        if const_value is None:
            return False

        try:
            # Handle scalar vs array datasets for reading data
            if dataset.shape == ():  # Scalar dataset
                data_value = dataset[()]
                if data_value != const_value:
                    self._handle_error(ValidationError(f"Dataset {dataset.name} value {data_value} != const {const_value}"))
                    return True
            else:  # Array dataset
                data_values = dataset[:]
                if not np.all(data_values == const_value):
                    self._handle_error(ValidationError(f"Dataset {dataset.name} contains values not equal to const {const_value}"))
                    return True
        except Exception:
            # Log but don't fail validation
            pass

        return False

    def __handle_not_dataset(self, dataset: h5py.Dataset, dataset_schema: DatasetSchema) -> bool:
        """
        Handle not validation for a dataset by ensuring the schema does NOT validate successfully.
        """
        has_error = False
        not_schema_dict = dataset_schema.not_schema

        # Check basic properties against the not schema
        not_matches = True

        # Check dtype if specified in not schema
        if "dtype" in not_schema_dict:
            expected_not_dtype = np.dtype(not_schema_dict["dtype"])
            if not self._dtypes_compatible(dataset.dtype, expected_not_dtype):
                not_matches = False

        # Check shape if specified in not schema
        if "shape" in not_schema_dict and not_matches:
            not_shape = not_schema_dict["shape"]
            if not_shape == [-1]:  # wildcard case always matches
                pass
            else:
                actual_matches_not_shape = True
                if len(dataset.shape) != len(not_shape):
                    actual_matches_not_shape = False
                else:
                    for _i, (actual_dim, not_dim) in enumerate(zip(dataset.shape, not_shape)):
                        if not_dim != -1 and actual_dim != not_dim:
                            actual_matches_not_shape = False
                            break
                if not actual_matches_not_shape:
                    not_matches = False

        # Check const constraint if specified in not schema
        if "const" in not_schema_dict and not_matches:
            const_value = not_schema_dict["const"]
            try:
                # Handle scalar vs array datasets for reading data
                if dataset.shape == ():  # Scalar dataset
                    data_value = dataset[()]
                    if data_value != const_value:
                        not_matches = False
                else:  # Array dataset
                    data_values = dataset[:]
                    if not np.all(data_values == const_value):
                        not_matches = False
            except Exception:
                not_matches = False

        # Check enum constraint if specified in not schema
        if "enum" in not_schema_dict and not_matches:
            enum_values = not_schema_dict["enum"]
            try:
                # Handle scalar vs array datasets for reading data
                if dataset.shape == ():  # Scalar dataset
                    data_value = dataset[()]
                    if data_value not in enum_values:
                        not_matches = False
                else:  # Array dataset
                    data_values = dataset[:]
                    unique_values = np.unique(data_values)
                    enum_matches = all(val in enum_values for val in unique_values)
                    if not enum_matches:
                        not_matches = False
            except Exception:
                not_matches = False

        # If the dataset matches the 'not' schema, that's an error
        if not_matches:
            self._handle_error(ValidationError(f"Dataset {dataset.name} matched 'not' schema (should not validate)"))
            has_error = True

        return has_error

    def _validate_string_format(self, value: str, format_type: str) -> bool:
        """
        Validate a string value against a specific format type.

        Parameters
        ----------
        value : str
            The string value to validate
        format_type : str
            The format type to validate against

        Returns
        -------
        bool
            True if the value matches the format, False otherwise

        """
        if format_type == "email":
            # Basic email validation regex
            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            return re.match(email_pattern, value) is not None

        elif format_type == "uri":
            # Basic URI validation - check for scheme://
            uri_pattern = r"^[a-zA-Z][a-zA-Z0-9+.-]*://.*"
            return re.match(uri_pattern, value) is not None

        elif format_type == "date-time":
            # ISO 8601 datetime format validation
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
                return True
            except ValueError:
                return False

        elif format_type == "date":
            # ISO date format YYYY-MM-DD
            date_pattern = r"^\d{4}-\d{2}-\d{2}$"
            if not re.match(date_pattern, value):
                return False
            try:
                datetime.strptime(value, "%Y-%m-%d")
                return True
            except ValueError:
                return False

        elif format_type == "time":
            # ISO time format HH:MM:SS or HH:MM:SS.fff
            time_pattern = r"^\d{2}:\d{2}:\d{2}(\.\d+)?$"
            if not re.match(time_pattern, value):
                return False
            try:
                datetime.strptime(value.split(".")[0], "%H:%M:%S")
                return True
            except ValueError:
                return False

        elif format_type == "uuid":
            # UUID format validation
            try:
                uuid.UUID(value)
                return True
            except ValueError:
                return False

        elif format_type == "ipv4":
            # IPv4 address validation
            ipv4_pattern = r"^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
            return re.match(ipv4_pattern, value) is not None

        elif format_type == "ipv6":
            # Basic IPv6 validation (simplified)
            ipv6_pattern = r"^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$|^::1$|^::$"
            return re.match(ipv6_pattern, value) is not None

        elif format_type == "hostname":
            # Basic hostname validation
            hostname_pattern = r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?))*$"
            return re.match(hostname_pattern, value) is not None

        elif format_type == "regex":
            # For regex format, just check if it's a valid regex pattern
            try:
                re.compile(value)
                return True
            except re.error:
                return False

        return False  # Unknown format types fail by default

    def __handle_format_dataset(self, dataset: h5py.Dataset, dataset_schema: DatasetSchema) -> bool:
        """
        Handle format validation for string datasets.
        """
        format_type = dataset_schema.format
        has_error = False

        # Only validate string-like datasets (including object dtype that might contain strings)
        if dataset.dtype.kind not in ("S", "U", "O"):
            self._handle_error(ValidationError(f"Dataset {dataset.name} format validation requires string data, got {dataset.dtype}"))
            return True

        try:
            # Handle scalar vs array datasets
            if dataset.shape == ():  # Scalar dataset
                value = dataset[()]
                if isinstance(value, (bytes, np.bytes_)):
                    value = value.decode("utf-8", errors="ignore")

                if not self._validate_string_format(str(value), format_type):
                    self._handle_error(ValidationError(f"Dataset {dataset.name} value '{value}' does not match format '{format_type}'"))
                    has_error = True
            else:  # Array dataset
                data_values = dataset[:]
                for i, value in enumerate(data_values.flat):
                    if isinstance(value, (bytes, np.bytes_)):
                        value = value.decode("utf-8", errors="ignore")

                    if not self._validate_string_format(str(value), format_type):
                        self._handle_error(ValidationError(f"Dataset {dataset.name}[{i}] value '{value}' does not match format '{format_type}'"))
                        has_error = True
        except Exception as e:
            self._handle_error(ValidationError(f"Error validating format for dataset {dataset.name}: {e}"))
            has_error = True

        return has_error

    def __handle_length_dataset(self, dataset: h5py.Dataset, dataset_schema: DatasetSchema) -> bool:
        """
        Handle minLength and maxLength validation for string datasets.
        """
        has_error = False

        # Only validate string-like datasets (including object dtype that might contain strings)
        if dataset.dtype.kind not in ("S", "U", "O"):
            if dataset_schema.has_min_length() or dataset_schema.has_max_length():
                self._handle_error(ValidationError(f"Dataset {dataset.name} length validation requires string data, got {dataset.dtype}"))
                return True
            return False

        try:
            # Handle scalar vs array datasets
            if dataset.shape == ():  # Scalar dataset
                value = dataset[()]
                if isinstance(value, (bytes, np.bytes_)):
                    value = value.decode("utf-8", errors="ignore")

                str_value = str(value)
                if dataset_schema.has_min_length() and len(str_value) < dataset_schema.min_length:
                    self._handle_error(ValidationError(f"Dataset {dataset.name} value length {len(str_value)} < minLength {dataset_schema.min_length}"))
                    has_error = True
                if dataset_schema.has_max_length() and len(str_value) > dataset_schema.max_length:
                    self._handle_error(ValidationError(f"Dataset {dataset.name} value length {len(str_value)} > maxLength {dataset_schema.max_length}"))
                    has_error = True
            else:  # Array dataset
                data_values = dataset[:]
                for i, value in enumerate(data_values.flat):
                    if isinstance(value, (bytes, np.bytes_)):
                        value = value.decode("utf-8", errors="ignore")

                    str_value = str(value)
                    if dataset_schema.has_min_length() and len(str_value) < dataset_schema.min_length:
                        self._handle_error(ValidationError(f"Dataset {dataset.name}[{i}] value length {len(str_value)} < minLength {dataset_schema.min_length}"))
                        has_error = True
                    if dataset_schema.has_max_length() and len(str_value) > dataset_schema.max_length:
                        self._handle_error(ValidationError(f"Dataset {dataset.name}[{i}] value length {len(str_value)} > maxLength {dataset_schema.max_length}"))
                        has_error = True
        except Exception as e:
            self._handle_error(ValidationError(f"Error validating string length for dataset {dataset.name}: {e}"))
            has_error = True

        return has_error

    def __handle_pattern_dataset(self, dataset: h5py.Dataset, dataset_schema: DatasetSchema) -> bool:
        """
        Handle regex pattern validation for string datasets.
        """
        pattern = dataset_schema.pattern
        has_error = False

        # Only validate string-like datasets (including object dtype that might contain strings)
        if dataset.dtype.kind not in ("S", "U", "O"):
            self._handle_error(ValidationError(f"Dataset {dataset.name} pattern validation requires string data, got {dataset.dtype}"))
            return True

        try:
            # Compile the regex pattern
            regex_pattern = re.compile(pattern)

            # Handle scalar vs array datasets
            if dataset.shape == ():  # Scalar dataset
                value = dataset[()]
                if isinstance(value, (bytes, np.bytes_)):
                    value = value.decode("utf-8", errors="ignore")

                if not regex_pattern.match(str(value)):
                    self._handle_error(ValidationError(f"Dataset {dataset.name} value '{value}' does not match pattern '{pattern}'"))
                    has_error = True
            else:  # Array dataset
                data_values = dataset[:]
                for i, value in enumerate(data_values.flat):
                    if isinstance(value, (bytes, np.bytes_)):
                        value = value.decode("utf-8", errors="ignore")

                    if not regex_pattern.match(str(value)):
                        self._handle_error(ValidationError(f"Dataset {dataset.name}[{i}] value '{value}' does not match pattern '{pattern}'"))
                        has_error = True
        except re.error as e:
            self._handle_error(ValidationError(f"Invalid regex pattern '{pattern}' for dataset {dataset.name}: {e}"))
            has_error = True
        except Exception as e:
            self._handle_error(ValidationError(f"Error validating pattern for dataset {dataset.name}: {e}"))
            has_error = True

        return has_error

    def __handle_dependent_required_dataset(self, dataset: h5py.Dataset, dataset_schema: DatasetSchema) -> bool:
        """
        Handle dependentRequired validation for a dataset.
        For datasets, this typically validates based on attribute presence.
        """
        has_error = False
        for property_name, required_properties in dataset_schema.dependent_required.items():
            # Check if the triggering property exists as an attribute
            if property_name in dataset.attrs:
                # If it exists, check that all dependent required properties also exist as attributes
                for required_prop in required_properties:
                    if required_prop not in dataset.attrs:
                        self._handle_error(ValidationError(
                            f"Attribute '{property_name}' exists in dataset {dataset.name} but required dependent attribute '{required_prop}' is missing"
                        ))
                        has_error = True
        return has_error

    def __handle_dependent_schemas_dataset(self, dataset: h5py.Dataset, dataset_schema: DatasetSchema) -> bool:
        """
        Handle dependentSchemas validation for a dataset.
        For datasets, this typically validates based on attribute presence and applies schemas to the dataset.
        """
        has_error = False
        for property_name, dependent_schema_dict in dataset_schema.dependent_schemas.items():
            # Check if the triggering property exists as an attribute
            if property_name in dataset.attrs:
                # If it exists, validate the dataset against the dependent schema
                # Create a temporary DatasetSchema from the dependent schema dict
                try:
                    temp_dataset_schema = DatasetSchema(dependent_schema_dict, dataset_schema.selector, dataset_schema.parent)
                    if not self._validate_dataset(dataset, temp_dataset_schema):
                        has_error = True
                except ValidationError:
                    has_error = True
        return has_error

    def __handle_conditional_dataset(self, dataset: h5py.Dataset, dataset_schema: DatasetSchema) -> bool:
        """
        Handle conditional validation (if/then/else) for a dataset, including nested conditionals.
        Returns True if there's a validation error, False if valid.
        """
        if_schema_dict = dataset_schema.if_schema
        then_schema_dict = dataset_schema.then_schema
        else_schema_dict = dataset_schema.else_schema

        # Evaluate the 'if' condition
        if_matches = self._evaluate_dataset_condition(dataset, if_schema_dict)

        # Apply appropriate consequence
        if if_matches and then_schema_dict:
            # Condition is true, validate against 'then' schema
            return self._validate_dataset_conditional_consequence(dataset, then_schema_dict)
        elif not if_matches and else_schema_dict:
            # Condition is false, validate against 'else' schema
            return self._validate_dataset_conditional_consequence(dataset, else_schema_dict)

        # If no applicable consequence, no error
        return False

    def _validate_dataset_conditional_consequence(self, dataset: h5py.Dataset, consequence_dict: dict) -> bool:
        """
        Validate a dataset against a conditional consequence, handling nested conditionals.
        Returns True if there's an error, False if valid.
        """
        # Check if the consequence has nested conditionals
        if "if" in consequence_dict:
            # Handle nested conditional by creating a temporary DatasetSchema
            temp_schema = DatasetSchema(consequence_dict, None, None, None)
            return self.__handle_conditional_dataset(dataset, temp_schema)

        # No nested conditional, apply the consequence directly
        return self._validate_dataset_against_condition(dataset, consequence_dict)

    def _evaluate_dataset_condition(self, dataset: h5py.Dataset, condition_dict: dict) -> bool:
        """
        Evaluate whether a dataset matches a condition schema.
        """
        if not condition_dict:
            return True

        # Check dtype condition
        if "dtype" in condition_dict:
            expected_dtype = np.dtype(condition_dict["dtype"])
            if not self._dtypes_compatible(dataset.dtype, expected_dtype):
                return False

        # Check shape condition
        if "shape" in condition_dict:
            expected_shape = condition_dict["shape"]
            if len(dataset.shape) != len(expected_shape):
                return False
            for _i, (actual_dim, expected_dim) in enumerate(zip(dataset.shape, expected_shape)):
                if expected_dim != -1 and actual_dim != expected_dim:
                    return False

        # Check const condition
        if "const" in condition_dict:
            const_value = condition_dict["const"]
            try:
                if dataset.shape == ():
                    data_value = dataset[()]
                    if data_value != const_value:
                        return False
                else:
                    data_values = dataset[:]
                    if not np.all(data_values == const_value):
                        return False
            except Exception:
                return False

        # Check enum condition
        if "enum" in condition_dict:
            enum_values = condition_dict["enum"]
            try:
                if dataset.shape == ():
                    data_value = dataset[()]
                    if data_value not in enum_values:
                        return False
                else:
                    data_values = dataset[:]
                    unique_values = np.unique(data_values)
                    if not all(val in enum_values for val in unique_values):
                        return False
            except Exception:
                return False

        # Check attribute conditions
        if "attrs" in condition_dict:
            for attr_condition in condition_dict["attrs"]:
                attr_name = attr_condition["name"]
                if attr_name not in dataset.attrs:
                    return False

                attr_value = dataset.attrs[attr_name]

                # Check attribute const
                if "const" in attr_condition:
                    if attr_value != attr_condition["const"]:
                        return False

                # Check attribute dtype
                if "dtype" in attr_condition:
                    expected_attr_dtype = np.dtype(attr_condition["dtype"])
                    attr_array = np.asarray(attr_value)
                    if not self._dtypes_compatible(attr_array.dtype, expected_attr_dtype):
                        return False

        return True

    def _validate_dataset_against_condition(self, dataset: h5py.Dataset, condition_dict: dict) -> bool:
        """
        Validate a dataset against a conditional schema (then/else).
        Returns True if there's an error, False if valid.
        """
        has_error = False

        # Validate dtype constraint
        if "dtype" in condition_dict:
            expected_dtype = np.dtype(condition_dict["dtype"])
            if not self._dtypes_compatible(dataset.dtype, expected_dtype):
                self._handle_error(ValidationError(f"Dataset {dataset.name} dtype {dataset.dtype} does not match conditional constraint {expected_dtype}"))
                has_error = True

        # Validate shape constraint
        if "shape" in condition_dict:
            expected_shape = condition_dict["shape"]
            if len(dataset.shape) != len(expected_shape):
                self._handle_error(ValidationError(f"Dataset {dataset.name} shape {dataset.shape} does not match conditional constraint {expected_shape}"))
                has_error = True
            else:
                for _i, (actual_dim, expected_dim) in enumerate(zip(dataset.shape, expected_shape)):
                    if expected_dim != -1 and actual_dim != expected_dim:
                        self._handle_error(ValidationError(f"Dataset {dataset.name} shape {dataset.shape} does not match conditional constraint {expected_shape}"))
                        has_error = True

        # Validate const constraint
        if "const" in condition_dict:
            const_value = condition_dict["const"]
            try:
                if dataset.shape == ():
                    data_value = dataset[()]
                    if data_value != const_value:
                        self._handle_error(ValidationError(f"Dataset {dataset.name} value {data_value} does not match conditional const {const_value}"))
                        has_error = True
                else:
                    data_values = dataset[:]
                    if not np.all(data_values == const_value):
                        self._handle_error(ValidationError(f"Dataset {dataset.name} values do not match conditional const {const_value}"))
                        has_error = True
            except Exception:
                self._handle_error(ValidationError(f"Dataset {dataset.name} failed conditional const validation"))
                has_error = True

        # Validate enum constraint
        if "enum" in condition_dict:
            enum_values = condition_dict["enum"]
            try:
                if dataset.shape == ():
                    data_value = dataset[()]
                    if data_value not in enum_values:
                        self._handle_error(ValidationError(f"Dataset {dataset.name} value {data_value} not in conditional enum {enum_values}"))
                        has_error = True
                else:
                    data_values = dataset[:]
                    unique_values = np.unique(data_values)
                    for val in unique_values:
                        if val not in enum_values:
                            self._handle_error(ValidationError(f"Dataset {dataset.name} contains value {val} not in conditional enum {enum_values}"))
                            has_error = True
            except Exception:
                self._handle_error(ValidationError(f"Dataset {dataset.name} failed conditional enum validation"))
                has_error = True

        # Validate attribute constraints
        if "attrs" in condition_dict:
            for attr_constraint in condition_dict["attrs"]:
                attr_name = attr_constraint["name"]

                # Check if required attribute is present
                if attr_constraint.get("required", False) and attr_name not in dataset.attrs:
                    self._handle_error(ValidationError(f"Required conditional attribute {attr_name} missing from dataset {dataset.name}"))
                    has_error = True
                    continue

                if attr_name in dataset.attrs:
                    attr_value = dataset.attrs[attr_name]

                    # Validate attribute const
                    if "const" in attr_constraint:
                        if attr_value != attr_constraint["const"]:
                            self._handle_error(ValidationError(f"Dataset {dataset.name} attribute {attr_name} value {attr_value} does not match conditional const {attr_constraint['const']}"))
                            has_error = True

                    # Validate attribute dtype
                    if "dtype" in attr_constraint:
                        expected_attr_dtype = np.dtype(attr_constraint["dtype"])
                        attr_array = np.asarray(attr_value)
                        if not self._dtypes_compatible(attr_array.dtype, expected_attr_dtype):
                            self._handle_error(ValidationError(f"Dataset {dataset.name} attribute {attr_name} dtype {attr_array.dtype} does not match conditional constraint {expected_attr_dtype}"))
                            has_error = True

        return has_error

    def _apply_dataset_conditional(self, dataset: h5py.Dataset, dataset_schema: DatasetSchema) -> DatasetSchema:
        """
        Apply conditional constraints to create an effective dataset schema, handling nested conditionals.
        """
        return self._resolve_dataset_conditional(dataset, dataset_schema)

    def _resolve_dataset_conditional(self, dataset: h5py.Dataset, dataset_schema: DatasetSchema) -> DatasetSchema:
        """
        Recursively resolve conditional constraints, including nested conditionals.
        """
        if_schema_dict = dataset_schema.if_schema
        then_schema_dict = dataset_schema.then_schema
        else_schema_dict = dataset_schema.else_schema

        # Evaluate the 'if' condition
        if_matches = self._evaluate_dataset_condition(dataset, if_schema_dict)

        # Select the appropriate consequence
        consequence_dict = None
        if if_matches and then_schema_dict:
            consequence_dict = then_schema_dict
        elif not if_matches and else_schema_dict:
            consequence_dict = else_schema_dict

        # Create base schema from original (without conditionals)
        base_schema_dict = dataset_schema.schema.copy()
        for key in ["if", "then", "else"]:
            base_schema_dict.pop(key, None)

        if consequence_dict:
            # Check if consequence has nested conditionals
            if "if" in consequence_dict:
                # Handle nested conditional by creating temporary schema and recursing
                temp_schema = DatasetSchema(consequence_dict, dataset_schema.selector, dataset_schema.parent, dataset_schema.root_schema)
                nested_resolved = self._resolve_dataset_conditional(dataset, temp_schema)
                consequence_dict = nested_resolved.schema

            # Merge consequence into base schema
            for key, value in consequence_dict.items():
                if key == "attrs":
                    # Merge attributes
                    existing_attrs = base_schema_dict.get("attrs", [])
                    base_schema_dict["attrs"] = existing_attrs + value
                else:
                    # Override other properties
                    base_schema_dict[key] = value

        # Create new DatasetSchema with merged schema
        return DatasetSchema(base_schema_dict, dataset_schema.selector, dataset_schema.parent, dataset_schema.root_schema)

    def __handle_attributes_dataset(self, dataset: h5py.Dataset, dataset_schema: DatasetSchema) -> bool:
        """
        Handle attribute validation for a dataset by ensuring all attributes match the schema attributes.
        """
        has_error = False
        seen_schema_attrs = set()

        for attr_name, attr_value in dataset.attrs.items():
            if attr_name not in dataset_schema.attrs:
                self._handle_error(ValidationError(f"{dataset.name} attribute {attr_name} is not included in schema"))
                has_error = True
            else:
                seen_schema_attrs.add(attr_name)
                dataset_schema_attr = dataset_schema.attrs[attr_name]
                expected_dtype = np.dtype(dataset_schema_attr["dtype"])

                # Convert attribute value to numpy array for consistent dtype checking
                attr_array = np.asarray(attr_value)

                if not self._dtypes_compatible(attr_array.dtype, expected_dtype):
                    self._handle_error(ValidationError(f"{dataset.name} attribute {attr_name} has dtype {attr_array.dtype} but schema expects {expected_dtype}"))
                    has_error = True
                schema_shape = dataset_schema_attr.get("shape")
                if schema_shape is not None:
                    if len(attr_value.shape) != len(schema_shape):
                        self._handle_error(ValidationError(f"{dataset.name} attribute {attr_name} shape {attr_value.shape} does not match the schema shape {schema_shape}"))
                        has_error = True
                    else:
                        for axis, axis_size in enumerate(attr_value.shape):
                            if (schema_shape[axis] != -1) and (axis_size != schema_shape[axis]):
                                self._handle_error(ValidationError(f"{dataset.name} attribute {attr_name} shape {attr_value.shape} does not match the schema shape {schema_shape}"))
                                has_error = True

        # Check for missing required attributes
        for attr_name, attr in dataset_schema.attrs.items():
            if attr.get("required", False) and attr_name not in seen_schema_attrs:
                self._handle_error(ValidationError(f"Required schema attribute {attr_name} is not included in {dataset} attributes"))
                has_error = True

        return has_error

    def _validate_dataset(
        self,
        dataset: h5py.Dataset,
        dataset_schema: DatasetSchema,
    ) -> bool:
        """
        Validate an HDF5 dataset against that dataset's schema.

        Parameters
        ----------
        dataset: h5py.Dataset
            Dataset to validate
        dataset_schema: DatasetSchema
            Corresponding dataset schema

        Returns
        -------
        bool:
            `True` if valid

        """
        has_error = False
        # Check the dtype
        if (dataset_schema.dtype is not None) and (not self._dtypes_compatible(dataset.dtype, dataset_schema.dtype)):
            self._handle_error(ValidationError(f"{dataset.name} dtype {dataset.dtype} is not compatible with schema dtype {dataset_schema.dtype}"))
            has_error = True

        # Check the shape
        if dataset_schema.shape is not None:
            has_error = self.__handle_shape_dataset(dataset, dataset_schema) or has_error

        # Handle enum constraints for datasets (validate dataset values against allowed values)
        if dataset_schema.has_enum():
            has_error = self.__handle_enum_dataset(dataset, dataset_schema) or has_error

        # Handle const constraints for datasets (validate dataset values match exact value)
        if dataset_schema.has_const():
            has_error = self.__handle_const_dataset(dataset, dataset_schema) or has_error

        # Handle format constraints for datasets (validate string format)
        if dataset_schema.has_format():
            has_error = self.__handle_format_dataset(dataset, dataset_schema) or has_error

        # Handle length constraints for datasets (validate string length)
        if dataset_schema.has_min_length() or dataset_schema.has_max_length():
            has_error = self.__handle_length_dataset(dataset, dataset_schema) or has_error

        # Handle pattern constraints for datasets (validate regex pattern)
        if dataset_schema.has_pattern():
            has_error = self.__handle_pattern_dataset(dataset, dataset_schema) or has_error

        # Handle dependentRequired constraints for datasets
        if dataset_schema.has_dependent_required():
            has_error = self.__handle_dependent_required_dataset(dataset, dataset_schema) or has_error

        # Handle dependentSchemas constraints for datasets
        if dataset_schema.has_dependent_schemas():
            has_error = self.__handle_dependent_schemas_dataset(dataset, dataset_schema) or has_error

        # Handle not constraints for datasets
        if dataset_schema.has_not():
            has_error = self.__handle_not_dataset(dataset, dataset_schema) or has_error

        # Handle conditional constraints for datasets (if/then/else)
        effective_schema = dataset_schema
        if dataset_schema.has_conditional():
            # Create a new schema with conditional constraints applied
            effective_schema = self._apply_dataset_conditional(dataset, dataset_schema)
            if effective_schema is None:
                has_error = True

        # Check that all the dataset attributes match the effective schema attributes
        has_error = self.__handle_attributes_dataset(dataset, effective_schema) or has_error

        return not has_error

    def __handle_any_of_group(self, group: h5py.Group, group_schema: GroupSchema) -> bool:
        """
        Handle anyOf validation for a group by trying each alternative schema until one passes.
        """
        for alt_schema in group_schema.any_of_schemas:
            try:
                # Temporarily disable error collection for alternative testing
                original_iter_errors = self._iter_errors
                self._iter_errors = False

                result = self._validate_group(group, alt_schema)

                # Restore original error collection mode
                self._iter_errors = original_iter_errors

                if result:
                    return True  # Found a matching alternative
            except ValidationError:
                continue  # Try next alternative

        # If we get here, none of the alternatives matched
        self._handle_error(ValidationError(f"Group {group.name} failed all anyOf alternatives"))
        return False

    def __handle_all_of_group(self, group: h5py.Group, group_schema: GroupSchema) -> bool:
        """
        Handle allOf validation for a group by ensuring all schemas validate successfully.
        """
        for all_schema in group_schema.all_of_schemas:
            # For allOf, validate only the constraints specified in each schema
            # Doesn't require that the schema be complete (unlike regular group validation)
            has_error = False

            # Check items that are specified in this allOf schema
            for schema_item in all_schema:
                if schema_item.name in group:
                    try:
                        self._validate(group[schema_item.name], schema_item)
                    except ValidationError as e:
                        self._handle_error(ValidationError(f"Group {group.name} failed allOf schema: {e}"))
                        has_error = True
                else:
                    if schema_item.required:
                        self._handle_error(ValidationError(f"Group {group.name} failed allOf schema: required item {schema_item.name} missing"))
                        has_error = True

            if has_error:
                return False

        return True

    def __handle_one_of_group(self, group: h5py.Group, group_schema: GroupSchema) -> bool:
        """
        Handle oneOf validation for a group by ensuring exactly one schema validates successfully.
        """
        valid_count = 0
        for one_schema in group_schema.one_of_schemas:
            # For oneOf, only the constraints specified in each schema
            schema_valid = True

            # Temporarily disable error collection for oneOf testing
            original_iter_errors = self._iter_errors
            self._iter_errors = False

            try:
                # Check items that are specified in this oneOf schema
                for schema_item in one_schema:
                    if schema_item.name in group:
                        try:
                            self._validate(group[schema_item.name], schema_item)
                        except ValidationError:
                            schema_valid = False
                            break
                    else:
                        if schema_item.required:
                            schema_valid = False
                            break

                if schema_valid:
                    valid_count += 1

            except ValidationError:
                schema_valid = False
            finally:
                # Restore original error collection mode
                self._iter_errors = original_iter_errors

        if valid_count == 0:
            self._handle_error(ValidationError(f"Group {group.name} failed all oneOf alternatives"))
            return False
        elif valid_count > 1:
            self._handle_error(ValidationError(f"Group {group.name} matched multiple oneOf alternatives (expected exactly one)"))
            return False

        return True

    def __handle_not_group(self, group: h5py.Group, group_schema: GroupSchema) -> bool:
        """
        Handle not validation for a group by ensuring the schema does NOT validate successfully.
        """
        not_schema = group_schema.not_schema

        # Temporarily disable error collection for not testing
        original_iter_errors = self._iter_errors
        self._iter_errors = False

        schema_valid = True
        try:
            # Check items that are specified in the not schema
            for schema_item in not_schema:
                if schema_item.name in group:
                    try:
                        self._validate(group[schema_item.name], schema_item)
                    except ValidationError:
                        schema_valid = False
                        break
                else:
                    if schema_item.required:
                        schema_valid = False
                        break

        except ValidationError:
            schema_valid = False
        finally:
            # Restore original error collection mode
            self._iter_errors = original_iter_errors

        # For 'not', if the schema validates successfully, that's an error
        if schema_valid:
            self._handle_error(ValidationError(f"Group {group.name} matched 'not' schema (should not validate)"))
            return False

        return True

    def __handle_conditional_group(self, group: h5py.Group, group_schema: GroupSchema) -> bool:
        """
        Handle conditional validation (if/then/else) for a group, including nested conditionals.
        """
        if_schema = group_schema.if_schema
        then_schema = group_schema.then_schema
        else_schema = group_schema.else_schema

        # Evaluate the 'if' condition
        if_matches = self._evaluate_group_condition(group, if_schema)

        # Apply appropriate consequence
        if if_matches and then_schema:
            # Condition is true, validate against 'then' schema
            return self._apply_conditional_schema(group, then_schema, group_schema)
        elif not if_matches and else_schema:
            # Condition is false, validate against 'else' schema
            return self._apply_conditional_schema(group, else_schema, group_schema)

        # If no applicable consequence, validation passes
        return True

    def _apply_conditional_schema(self, group: h5py.Group, consequence_schema: GroupSchema, parent_schema: GroupSchema) -> bool:
        """
        Apply a conditional consequence schema, handling nested conditionals.
        """
        # Check if the consequence schema itself has conditionals (nested)
        if consequence_schema.has_conditional():
            # Handle nested conditional
            return self.__handle_conditional_group(group, consequence_schema)

        # Create a merged schema from parent and consequence
        base_schema = {
            "type": "group",
            "members": {},
            "attrs": parent_schema.schema.get("attrs", [])
        }

        # Merge consequence schema
        if "members" in consequence_schema.schema:
            base_schema["members"].update(consequence_schema.schema["members"])
        if "required" in consequence_schema.schema:
            base_schema["required"] = consequence_schema.schema["required"]
        if "attrs" in consequence_schema.schema:
            base_schema["attrs"].extend(consequence_schema.schema["attrs"])

        # Create a new GroupSchema from the merged schema and validate against it
        merged_schema = GroupSchema(base_schema, parent_schema.selector, parent=parent_schema.parent)
        return self._validate_group(group, merged_schema)

    def _evaluate_group_condition(self, group: h5py.Group, if_schema: GroupSchema) -> bool:
        """
        Evaluate whether a group matches a condition schema.
        """
        if not if_schema:
            return True

        # Check attributes directly from the schema dict for if condition
        if_schema_dict = if_schema.schema

        # Check attribute conditions - support multiple attributes with AND logic
        if "attrs" in if_schema_dict:
            for attr_condition in if_schema_dict["attrs"]:
                attr_name = attr_condition["name"]
                if attr_name not in group.attrs:
                    return False

                attr_value = group.attrs[attr_name]

                # Check attribute const
                if "const" in attr_condition:
                    if attr_value != attr_condition["const"]:
                        return False

                # Check attribute dtype
                if "dtype" in attr_condition:
                    expected_attr_dtype = np.dtype(attr_condition["dtype"])
                    attr_array = np.asarray(attr_value)
                    if not self._dtypes_compatible(attr_array.dtype, expected_attr_dtype):
                        return False

        # Check member conditions
        if "members" in if_schema_dict:
            for member_name, _member_condition in if_schema_dict["members"].items():
                if member_name == "required":
                    continue
                if member_name not in group:
                    return False
                # For now, just check if member exists; could add more detailed validation

        return True

    def __handle_enum_group(self, group: h5py.Group, group_schema: GroupSchema) -> bool:
        """
        Handle enum validation for a group by checking group name or structure against allowed values.
        """
        group_name = group.name.split("/")[-1] if group.name != "/" else "/"
        if group_name not in group_schema.enum:
            self._handle_error(ValidationError(f"Group name '{group_name}' not in allowed enum values {group_schema.enum}"))
            return False
        return True

    def __handle_const_group(self, group: h5py.Group, group_schema: GroupSchema) -> bool:
        """
        Handle const validation for a group by checking group name or structure matches exact value.
        """
        group_name = group.name.split("/")[-1] if group.name != "/" else "/"
        if group_name != group_schema.const:
            self._handle_error(ValidationError(f"Group name '{group_name}' does not match const value '{group_schema.const}'"))
            return False
        return True

    def __handle_dependent_required_group(self, group: h5py.Group, group_schema: GroupSchema) -> bool:
        """
        Handle dependentRequired validation for a group.
        If a property exists, then the dependent properties must also exist.
        """
        has_error = False
        for property_name, required_properties in group_schema.dependent_required.items():
            # Check if the triggering property exists in the group
            if property_name in group:
                # If it exists, check that all dependent required properties also exist
                for required_prop in required_properties:
                    if required_prop not in group:
                        self._handle_error(ValidationError(
                            f"Property '{property_name}' exists in {group.name} but required dependent property '{required_prop}' is missing"
                        ))
                        has_error = True
        return not has_error

    def __handle_dependent_schemas_group(self, group: h5py.Group, group_schema: GroupSchema) -> bool:
        """
        Handle dependentSchemas validation for a group.
        If a property exists, then apply the corresponding dependent schema constraints.
        """
        has_error = False
        for property_name, dependent_schema in group_schema.dependent_schemas.items():
            # Check if the triggering property exists in the group
            if property_name in group:
                # Apply the dependent schema - validate that all its members exist and are valid
                for member_schema in dependent_schema.members:
                    member_name = member_schema.name
                    if member_name in group:
                        # Member exists, validate it against the dependent schema
                        member_item = group[member_name]
                        try:
                            if not self._validate(member_item, member_schema):
                                has_error = True
                        except ValidationError:
                            has_error = True
                    else:
                        # Member required by dependent schema but missing
                        # Check if it's a pattern member or if all dependent schema members are implicitly required
                        self._handle_error(ValidationError(
                            f"Dependent schema for '{property_name}' requires member '{member_name}' which is missing from {group.name}"
                        ))
                        has_error = True
        return not has_error

    def __handle_attributes_group(self, group: h5py.Group, group_schema: GroupSchema) -> bool:
        """
        Handle attribute validation for a group by ensuring all attributes match the schema attributes.
        """
        has_error = False
        # Check that all items in the group are in the schema
        for item_name in group:
            if item_name not in group_schema:
                self._handle_error(ValidationError(f"{group.name} not in schema {group_schema}"))
                has_error = True
            else:
                target_schema = group_schema[item_name]
                # If multiple alternative schemas returned, try each until one passes
                if isinstance(target_schema, list):
                    alt_valid = False
                    for alt_schema in target_schema:
                        try:
                            self._validate(group[item_name], alt_schema)
                            alt_valid = True
                            break
                        except ValidationError:
                            continue
                    if not alt_valid:
                        self._handle_error(ValidationError(f"{group[item_name].name} failed all alternative schemas"))
                        has_error = True
                else:
                    self._validate(group[item_name], target_schema)

        # Check that all required items in the schema are in the group
        for schema_item in group_schema:
            if not schema_item.required:
                continue
            if schema_item.name not in group:
                self._handle_error(ValidationError(f"Required item {schema_item.name} is not in {group.name}"))
                has_error = True

        return not has_error

    def _validate_group(
        self,
        group: h5py.Group,
        group_schema: GroupSchema,
    ) -> bool:
        """
        Validate an HDF5 group against that group's schema.

        Parameters
        ----------
        group: h5py.Group
            Group to validate
        group_schema: GroupSchema
            Corresponding group schema

        Returns
        -------
        bool:
            `True` if valid

        """
        # Handle logical operators - these are mutually exclusive and take precedence
        # Handle anyOf at group level - try each alternative until one passes
        if group_schema.has_any_of():
            return self.__handle_any_of_group(group, group_schema)

        # Handle allOf at group level - all schemas must pass
        if group_schema.has_all_of():
            return self.__handle_all_of_group(group, group_schema)

        # Handle oneOf at group level - exactly one schema must pass
        if group_schema.has_one_of():
            return self.__handle_one_of_group(group, group_schema)

        # Handle not at group level - the schema must NOT validate successfully
        if group_schema.has_not():
            return self.__handle_not_group(group, group_schema)

        # Handle conditional validation at group level (if/then/else)
        if group_schema.has_conditional():
            return self.__handle_conditional_group(group, group_schema)

        # If no logical operators, proceed with standard validation
        no_errors = True

        # Handle enum constraints (validate group name or structure against allowed values)
        if group_schema.has_enum():
            no_errors = no_errors and self.__handle_enum_group(group, group_schema)

        # Handle const constraints (validate group name or structure matches exact value)
        if group_schema.has_const():
            no_errors = no_errors and self.__handle_const_group(group, group_schema)

        # Handle dependentRequired constraints
        if group_schema.has_dependent_required():
            no_errors = no_errors and self.__handle_dependent_required_group(group, group_schema)

        # Handle dependentSchemas constraints
        if group_schema.has_dependent_schemas():
            no_errors = no_errors and self.__handle_dependent_schemas_group(group, group_schema)

        # Handle regular group validation (members and attributes)
        no_errors = no_errors and self.__handle_attributes_group(group, group_schema)
        return no_errors

    def is_valid(self) -> bool:
        """
        Check if instance is valid given schema.

        This method is faster than `iter_errors` but may be less helpful for debugging an instance.

        Returns
        -------
        bool:
            `True` if valid. Otherwise raises first validation error

        """
        self._errors = []
        self._iter_errors = False
        try:
            return self._validate(self.instance, self.schema)
        except ValidationError:
            return False

    def iter_errors(self) -> List[ValidationError]:
        """
        Find all validation errors when comparing instance to schema.

        This method is slower than `is_valid` but may be more helpful for debugging an instance.

        Returns
        -------
        List[ValidationError]:
            Returns list of validation errors. An empty list means the instance is valid

        """
        self._errors = []
        self._iter_errors = True
        self._validate(self.instance, self.schema)
        return self._errors
