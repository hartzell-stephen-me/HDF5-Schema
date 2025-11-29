# HDF5 Schema - HDF5 Schema is Python library for defining schemas for HDF5 files and verifying compliance.

![Logo](/docs/images/python-hdf5-schema-icon.png)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![PyPI](https://img.shields.io/badge/pypi-hdf5schema-blue)](https://pypi.org/project/hdf5schema/)

## Introduction

HDF5 Schema is Python library for defining schemas for HDF5 files and verifying compliance.

## Features

- Validate HDF5 files against JSON schemas
- Support for groups, datasets, and attributes
- Pattern-based validation using regular expressions
- Command-line interface for validation and schema generation

## Installation

`hdf5schema` is available on [PyPI](https://pypi.org/project/hdf5schema/). You can install using [pip](https://pip.pypa.io/en/stable/):

```bash
pip install hdf5schema
```

## Usage

### Command Line Interface

#### Validating HDF5 Files

Validate an HDF5 file against a JSON schema:

```bash
# Basic validation
python -m hdf5schema.validate data.h5 schema.json

# Validate with verbose output
python -m hdf5schema.validate data.h5 schema.json --verbose

# Quiet mode (only show errors)
python -m hdf5schema.validate data.h5 schema.json --quiet
```

#### Generating Schemas

Generate a JSON schema from an existing HDF5 file:

```bash
# Generate schema to stdout
python -m hdf5schema.generate_schema input.h5

# Generate schema to file with pretty formatting  
python -m hdf5schema.generate_schema input.h5 -o schema.json --pretty

# Generate schema for a specific group
python -m hdf5schema.generate_schema input.h5 -g /my/group -o schema.json
```

### Python API

#### Validation

```python
from hdf5schema.validator import Hdf5Validator

# Validate a file
validator = Hdf5Validator("data.h5", "schema.json")

# Check if valid (returns boolean)
if validator.is_valid():
    print("File is valid!")
else:
    print("File is invalid")

# Get detailed error information
errors = list(validator.iter_errors())
if errors:
    print(f"Found {len(errors)} validation errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Validation passed!")
```

#### Schema Generation

```python
from hdf5schema.generate_schema import generate_schema
import json
from pathlib import Path

# Generate schema from HDF5 file
schema = generate_schema(Path("input.h5"))

# Save to file
with open("schema.json", "w") as f:
    json.dump(schema, f, indent=2)

# Generate schema from specific group
schema = generate_schema(Path("input.h5"), group_path="/sensors")
```

## Creating JSON Schemas

### Basic Schema Structure

HDF5 schemas are JSON documents that describe the expected structure of HDF5 files. Here's the basic structure:

```json
{
    "type": "group",
    "members": {
        "dataset_name": {
            "type": "dataset",
            "dtype": "<f8",
            "shape": [100, 50]
        },
        "subgroup_name": {
            "type": "group",
            "members": {
                // nested structure
            }
        }
    },
    "required": ["dataset_name"],
    "attrs": [
        {
            "name": "description",
            "dtype": "U256"
        }
    ]
}
```

### Dataset Schemas

Datasets define the structure of HDF5 datasets:

```json
{
    "type": "dataset",
    "dtype": "<f8",           // Data type (float64, little-endian)
    "shape": [100, 50],       // Expected shape
    "attrs": [                // Optional attributes
        {
            "name": "units",
            "dtype": "U32"
        }
    ]
}
```

#### Common Data Types

- `"<f8"` - 64-bit float (double), little-endian
- `"<f4"` - 32-bit float, little-endian  
- `"<i8"` - 64-bit signed integer, little-endian
- `"<u4"` - 32-bit unsigned integer, little-endian
- `"S256"` - Fixed-length ASCII string (256 bytes)
- `"U128"` - Fixed-length Unicode string (128 characters)
- `"|b1"` - Boolean (1 byte)

#### Compound Data Types

For structured data (like C structs):

```json
{
    "type": "dataset",
    "dtype": {
        "formats": [
            {
                "name": "timestamp",
                "format": "<f8"
            },
            {
                "name": "value",
                "format": "<f4",
                "offset": 8
            },
            {
                "name": "label",
                "format": "S32",
                "offset": 12
            }
        ],
        "itemsize": 44
    },
    "shape": [1000]
}
```

#### Flexible Shapes

Use `-1` for variable dimensions:

```json
{
    "type": "dataset",
    "dtype": "<f8",
    "shape": [-1, 3]  // Any number of rows, exactly 3 columns
}
```

### Group Schemas

Groups can contain other groups and datasets:

```json
{
    "type": "group",
    "members": {
        "data": {
            "type": "dataset",
            "dtype": "<f8",
            "shape": [-1]
        },
        "metadata": {
            "type": "group", 
            "members": {
                "info": {
                    "type": "dataset",
                    "dtype": "U512",
                    "shape": [1]
                }
            }
        }
    },
    "required": ["data"],     // Required members
    "attrs": [                // Group attributes
        {
            "name": "version",
            "dtype": "U10",
            "required": true
        }
    ]
}
```

### Pattern-Based Validation

Use `patternMembers` for dynamic member names:

```json
{
    "type": "group",
    "patternMembers": {
        "^sensor_[0-9]+$": {           // Matches sensor_1, sensor_2, etc.
            "type": "group",
            "members": {
                "data": {
                    "type": "dataset",
                    "dtype": "<f4",
                    "shape": [-1]
                }
            }
        },
        "^target-.+$": {               // Matches target-anything
            "type": "dataset",
            "dtype": "<f8",
            "shape": [3]
        }
    }
}
```
### Using references and definitions

Use `$defs` to define reusable data and use `$ref` to reference a definition. This can also be used for recursion:

```json
{
    "$defs": {
        "observable": {
            "type": "group",
            "description": "<observable>",
            "members": {
                "observables": {
                    "type": "group",
                    "patternMembers": {
                        ".*": {
                            "$ref": "#/$defs/observable"
                        }
                    }
                }
            }
        }
    },
    "type": "group", 
    "members": {
        "observables": {
            "type": "group",
            "patternMembers": {
                ".*": {
                    "$ref": "#/$defs/observable"
                }
            }
        }
    }
}
```

Another example with a dataset:

```json
{
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
```

### Comment

Use `$comment` to add documentation and explanations to your schemas. Comments don't affect validation but help document the schema's purpose:

```json
{
    "type": "group",
    "$comment": "Root group containing sensor data from multiple experiments",
    "members": {
        "temperature_data": {
            "type": "dataset",
            "dtype": "<f8",
            "shape": [-1],
            "$comment": "Temperature readings in Celsius, sampled at 1Hz"
        },
        "metadata": {
            "type": "group",
            "$comment": "Experimental metadata and configuration parameters",
            "members": {
                "experiment_id": {
                    "type": "dataset",
                    "dtype": "U32"
                }
            }
        }
    }
}
```

### Enum

Use `enum` to restrict dataset values to a specific set of allowed values. Useful for categorical data or status codes:

```json
{
    "type": "group",
    "members": {
        "sensor_status": {
            "type": "dataset",
            "dtype": "int32",
            "shape": [],
            "enum": [0, 1, 2],
            "$comment": "0=offline, 1=online, 2=maintenance"
        },
        "measurement_quality": {
            "type": "dataset", 
            "dtype": "int32",
            "shape": [-1],
            "enum": [1, 2, 3, 4, 5],
            "$comment": "Quality scores from 1 (poor) to 5 (excellent)"
        },
        "experiment_type": {
            "type": "dataset",
            "dtype": "U16", 
            "shape": [],
            "enum": ["control", "treatment_a", "treatment_b"]
        }
    }
}
```

### Const

Use `const` to require that a dataset contains exactly one specific value. Useful for version numbers, constants, or validation markers:

```json
{
    "type": "group",
    "members": {
        "schema_version": {
            "type": "dataset",
            "dtype": "U10",
            "shape": [],
            "const": "1.2.0",
            "$comment": "Must be exactly version 1.2.0"
        },
        "calibration_constant": {
            "type": "dataset",
            "dtype": "<f8",
            "shape": [],
            "const": 9.80665,
            "$comment": "Standard gravity constant"
        },
        "flags": {
            "type": "dataset",
            "dtype": "int32", 
            "shape": [100],
            "const": 1,
            "$comment": "All values must be 1 (valid data flag)"
        }
    }
}
```

### Boolean Schema Combination

These keywords allow you to create complex validation logic by combining multiple schema constraints.

#### allOf

Requires that ALL specified schemas must be satisfied. Use when you need multiple constraints to be true simultaneously:

```json
{
    "type": "group",
    "allOf": [
        {
            "$comment": "Must have temperature data",
            "members": {
                "temperature": {
                    "type": "dataset",
                    "dtype": "<f8",
                    "shape": [-1]
                }
            }
        },
        {
            "$comment": "Must have pressure data", 
            "members": {
                "pressure": {
                    "type": "dataset",
                    "dtype": "<f8",
                    "shape": [-1]
                }
            }
        },
        {
            "$comment": "Must have timestamps",
            "members": {
                "timestamps": {
                    "type": "dataset", 
                    "dtype": "<f8",
                    "shape": [-1]
                }
            }
        }
    ]
}
```

#### anyOf

Requires that AT LEAST ONE of the specified schemas must be satisfied. Use for alternative valid structures:

```json
{
    "type": "group", 
    "members": {
        "sensor_data": {
            "type": "dataset",
            "shape": [-1],
            "$comment": "Can be either float32 or float64 data",
            "anyOf": [
                {"dtype": "<f4"},
                {"dtype": "<f8"}
            ]
        }
    }
}
```

Alternative group-level example:

```json
{
    "type": "group",
    "$comment": "Data can be stored in either format",
    "anyOf": [
        {
            "$comment": "Format A: separate x,y,z datasets",
            "members": {
                "x_data": {"type": "dataset", "dtype": "<f8", "shape": [-1]},
                "y_data": {"type": "dataset", "dtype": "<f8", "shape": [-1]},
                "z_data": {"type": "dataset", "dtype": "<f8", "shape": [-1]}
            }
        },
        {
            "$comment": "Format B: single 3D dataset",
            "members": {
                "xyz_data": {"type": "dataset", "dtype": "<f8", "shape": [-1, 3]}
            }
        }
    ]
}
```

#### oneOf

Requires that EXACTLY ONE of the specified schemas must be satisfied. Use when you need mutually exclusive alternatives:

```json
{
    "type": "group",
    "$comment": "Data must be in exactly one of these formats",
    "oneOf": [
        {
            "$comment": "Raw data format",
            "members": {
                "raw_data": {
                    "type": "dataset",
                    "dtype": "<f8",
                    "shape": [-1]
                }
            }
        },
        {
            "$comment": "Processed data format", 
            "members": {
                "processed_data": {
                    "type": "dataset",
                    "dtype": "<f8", 
                    "shape": [-1]
                },
                "processing_metadata": {
                    "type": "group",
                    "members": {
                        "algorithm": {"type": "dataset", "dtype": "U32"}
                    }
                }
            }
        }
    ]
}
```

#### not

Requires that the specified schema must NOT be satisfied. Use for exclusions or negative constraints:

```json
{
    "type": "group",
    "members": {
        "sensor_readings": {
            "type": "dataset",
            "dtype": "<f8",
            "shape": [-1],
            "not": {
                "const": -999.0
            },
            "$comment": "Values must not be the error code -999.0"
        },
        "data_quality": {
            "type": "dataset",
            "dtype": "int32",
            "shape": [-1], 
            "not": {
                "enum": [0, -1]
            },
            "$comment": "Quality codes must not be 0 (invalid) or -1 (error)"
        }
    }
}
```

Group-level `not` example:

```json
{
    "type": "group",
    "not": {
        "members": {
            "deprecated_data": {
                "type": "dataset",
                "dtype": "<f4"
            }
        }
    },
    "$comment": "This group must not contain deprecated_data with float32 dtype"
}
```

#### dependentRequired

Requires certain members to be present when a specific member exists. Use when some fields are only required in the presence of other fields:

```json
{
    "type": "group",
    "members": {
        "sensor_type": {
            "type": "dataset",
            "dtype": "U32",
            "enum": ["temperature", "pressure", "humidity"]
        },
        "temperature_range": {
            "type": "dataset",
            "dtype": "<f8",
            "shape": [2],
            "$comment": "Min and max temperature values"
        },
        "pressure_units": {
            "type": "dataset", 
            "dtype": "U16",
            "$comment": "Units for pressure measurements"
        },
        "humidity_calibration": {
            "type": "dataset",
            "dtype": "<f8",
            "$comment": "Calibration factor for humidity sensor"
        }
    },
    "dependentRequired": {
        "sensor_type": ["temperature_range"],
        "pressure_units": ["sensor_type"],
        "humidity_calibration": ["sensor_type"]
    },
    "$comment": "When sensor_type exists, temperature_range is required. When pressure_units or humidity_calibration exist, sensor_type is required"
}
```

Dataset-level example:

```json
{
    "type": "dataset",
    "dtype": "<f8",
    "shape": [-1],
    "attrs": [
        {"name": "units", "dtype": "U32"},
        {"name": "scale_factor", "dtype": "<f8"},
        {"name": "add_offset", "dtype": "<f8"}
    ],
    "dependentRequired": {
        "scale_factor": ["units"],
        "add_offset": ["scale_factor", "units"]
    },
    "$comment": "If scale_factor is present, units is required. If add_offset is present, both scale_factor and units are required"
}
```

#### dependentSchemas

Applies additional schema validation when specific members are present. Use when the presence of one field changes the validation rules for other fields:

```json
{
    "type": "group",
    "members": {
        "data_format": {
            "type": "dataset",
            "dtype": "U16",
            "enum": ["raw", "processed", "compressed"]
        },
        "raw_data": {
            "type": "dataset",
            "dtype": "<f8",
            "shape": [-1]
        },
        "processing_params": {
            "type": "group",
            "members": {
                "algorithm": {"type": "dataset", "dtype": "U32"},
                "parameters": {"type": "dataset", "dtype": "<f8", "shape": [-1]}
            }
        },
        "compression_ratio": {
            "type": "dataset",
            "dtype": "<f8"
        }
    },
    "dependentSchemas": {
        "data_format": {
            "if": {
                "members": {
                    "data_format": {"const": "processed"}
                }
            },
            "then": {
                "required": ["processing_params"],
                "members": {
                    "processing_params": {
                        "required": ["algorithm"]
                    }
                }
            }
        }
    },
    "$comment": "When data_format is 'processed', processing_params becomes required with mandatory algorithm"
}
```

Pattern-based dependent validation:

```json
{
    "type": "group",
    "patternMembers": {
        "^sensor_\\d+$": {
            "type": "group",
            "members": {
                "status": {"type": "dataset", "dtype": "U16"},
                "error_log": {"type": "dataset", "dtype": "U256", "shape": [-1]},
                "last_maintenance": {"type": "dataset", "dtype": "U32"}
            },
            "dependentSchemas": {
                "status": {
                    "if": {
                        "members": {
                            "status": {"const": "error"}
                        }
                    },
                    "then": {
                        "required": ["error_log"]
                    }
                }
            }
        }
    },
    "$comment": "For any sensor group, if status is 'error', error_log becomes required"
}
```

#### Combining Multiple Keywords

You can combine these keywords for sophisticated validation logic:

```json
{
    "type": "group",
    "members": {
        "experiment_data": {
            "type": "dataset", 
            "allOf": [
                {
                    "$comment": "Must be numeric data",
                    "anyOf": [
                        {"dtype": "<f4"},
                        {"dtype": "<f8"},
                        {"dtype": "<i4"},
                        {"dtype": "<i8"}
                    ]
                },
                {
                    "$comment": "Must not be empty",
                    "not": {
                        "shape": [0]
                    }
                }
            ]
        },
        "status_code": {
            "type": "dataset",
            "dtype": "int32",
            "shape": [],
            "oneOf": [
                {"const": 200, "$comment": "Success"},
                {"enum": [400, 401, 403], "$comment": "Client errors"},
                {"enum": [500, 502, 503], "$comment": "Server errors"}
            ]
        }
    }
}
```

### Conditional Expressions

Conditional validation allows you to apply different validation rules based on the content of your data using `if`, `then`, and `else` keywords.

#### If-then-else Pattern

Use conditional validation when you need different validation rules depending on data values or structure:

```json
{
    "type": "group",
    "members": {
        "sensor_data": {
            "type": "dataset",
            "dtype": "<f8",
            "shape": [-1],
            "if": {
                "attrs": [
                    {"name": "sensor_type", "const": "temperature"}
                ]
            },
            "then": {
                "$comment": "Temperature sensors must have specific range",
                "attrs": [
                    {"name": "min_value", "dtype": "<f8", "required": true},
                    {"name": "max_value", "dtype": "<f8", "required": true},
                    {"name": "units", "const": "celsius", "required": true}
                ]
            },
            "else": {
                "$comment": "Other sensors have different requirements",
                "attrs": [
                    {"name": "calibration_factor", "dtype": "<f8", "required": true}
                ]
            }
        },
        "geolocation": {
            "type": "group",
            "if": {
                "attrs": [
                    {"name": "grounded", "const": true}
                ]
            },
            "then": {
                "$comment": "Grounded sensors need location data",
                "members": {
                    "latitude": {"type": "dataset", "dtype": "<f8"},
                    "longitude": {"type": "dataset", "dtype": "<f8"}
                },
                "required": ["latitude", "longitude"]
            },
            "else": {
                "if": {
                    "attrs": [
                        {"name": "mover", "const": true}
                    ]
                },
                "then": {
                    "$comment": "Mobile sensors need trajectory data",
                    "members": {
                        "trajectory": {
                            "type": "dataset",
                            "dtype": "<f8",
                            "shape": [-1, 3]
                        }
                    },
                    "required": ["trajectory"]
                }
            }
        }
    }
}
```

### String Format Validation

Use the `format` keyword to validate string content against common formats. This ensures data quality for structured string data like emails, dates, and identifiers.

#### date-time

Validates ISO 8601 datetime strings (`YYYY-MM-DDTHH:MM:SS` or with timezone):

```json
{
    "type": "group",
    "members": {
        "timestamp": {
            "type": "dataset",
            "dtype": "S30",
            "format": "date-time",
            "$comment": "Must be valid ISO 8601 datetime"
        }
    }
}
```

Valid examples: `"2023-12-25T14:30:00"`, `"2023-12-25T14:30:00Z"`, `"2023-12-25T14:30:00+05:00"`

#### date

Validates ISO date strings (`YYYY-MM-DD`):

```json
{
    "type": "dataset",
    "dtype": "S12", 
    "format": "date",
    "$comment": "Birth date in ISO format"
}
```

Valid examples: `"2023-12-25"`, `"1990-01-01"`

#### time

Validates ISO time strings (`HH:MM:SS` or `HH:MM:SS.fff`):

```json
{
    "type": "dataset",
    "dtype": "S15",
    "format": "time",
    "$comment": "Time of day in 24-hour format"
}
```

Valid examples: `"14:30:00"`, `"09:15:30.123"`

#### email

Validates email addresses using standard email format rules:

```json
{
    "type": "group",
    "members": {
        "contact_email": {
            "type": "dataset",
            "dtype": "S100",
            "format": "email",
            "$comment": "Valid email address required"
        }
    }
}
```

Valid examples: `"user@example.com"`, `"test.email+tag@domain.org"`

#### hostname

Validates hostnames and domain names according to DNS standards:

```json
{
    "type": "dataset",
    "dtype": "S255",
    "format": "hostname",
    "$comment": "Valid hostname or domain name"
}
```

Valid examples: `"example.com"`, `"server.example.org"`, `"localhost"`

#### ipv4

Validates IPv4 addresses in dotted decimal notation:

```json
{
    "type": "dataset", 
    "dtype": "S16",
    "format": "ipv4",
    "$comment": "Valid IPv4 address"
}
```

Valid examples: `"192.168.1.1"`, `"10.0.0.1"`, `"127.0.0.1"`

#### ipv6

Validates IPv6 addresses in standard notation:

```json
{
    "type": "dataset",
    "dtype": "S40", 
    "format": "ipv6",
    "$comment": "Valid IPv6 address"
}
```

Valid examples: `"2001:0db8:85a3:0000:0000:8a2e:0370:7334"`, `"::1"`

#### uri

Validates Uniform Resource Identifiers (URIs) with proper scheme:

```json
{
    "type": "dataset",
    "dtype": "S500",
    "format": "uri",
    "$comment": "Valid URI with scheme"
}
```

Valid examples: `"https://example.com/data"`, `"file:///path/to/file"`, `"ftp://server.com"`

#### uuid

Validates UUID (Universally Unique Identifier) strings:

```json
{
    "type": "dataset",
    "dtype": "S40",
    "format": "uuid", 
    "$comment": "Valid UUID identifier"
}
```

Valid examples: `"550e8400-e29b-41d4-a716-446655440000"`, `"6ba7b810-9dad-11d1-80b4-00c04fd430c8"`

#### regex

Validates that the string itself is a valid regular expression pattern:

```json
{
    "type": "dataset",
    "dtype": "S100",
    "format": "regex",
    "$comment": "Valid regex pattern string"
}
```

Valid examples: `"^[A-Za-z]+$"`, `"\\d{3}-\\d{3}-\\d{4}"`, `".*@.*\\.com$"`

### String Length and Pattern Validation

#### minLength

Enforces minimum string length for validation:

```json
{
    "type": "group",
    "members": {
        "password": {
            "type": "dataset",
            "dtype": "S128",
            "minLength": 8,
            "$comment": "Password must be at least 8 characters"
        },
        "product_code": {
            "type": "dataset", 
            "dtype": "S20",
            "minLength": 5,
            "maxLength": 15,
            "$comment": "Product codes between 5-15 characters"
        }
    }
}
```

#### maxLength

Enforces maximum string length for validation:

```json
{
    "type": "dataset",
    "dtype": "S50",
    "maxLength": 30,
    "$comment": "Username must not exceed 30 characters"
}
```

#### pattern

Validates strings against regular expression patterns for custom format validation:

```json
{
    "type": "group", 
    "members": {
        "phone_number": {
            "type": "dataset",
            "dtype": "S20",
            "pattern": "^\\d{3}-\\d{3}-\\d{4}$",
            "$comment": "US phone number format: XXX-XXX-XXXX"
        },
        "license_plate": {
            "type": "dataset",
            "dtype": "S10", 
            "pattern": "^[A-Z]{3}-\\d{4}$",
            "$comment": "Format: ABC-1234"
        },
        "experiment_id": {
            "type": "dataset",
            "dtype": "S15",
            "pattern": "^EXP-\\d{8}-[A-Z]{2}$", 
            "$comment": "Format: EXP-20231225-AB"
        }
    }
}
```

#### Combining String Validation

You can combine multiple string validation constraints for robust data validation:

```json
{
    "type": "group",
    "members": {
        "user_email": {
            "type": "dataset", 
            "dtype": "S100",
            "format": "email",
            "minLength": 6,
            "maxLength": 50,
            "pattern": ".*@company\\.com$",
            "$comment": "Company email: 6-50 chars, valid email, must end with @company.com"
        },
        "secure_token": {
            "type": "dataset",
            "dtype": "S64", 
            "minLength": 32,
            "maxLength": 64,
            "pattern": "^[A-Fa-f0-9]+$",
            "$comment": "Hexadecimal token, 32-64 characters"
        }
    }
}
```

String array validation example:

```json
{
    "type": "dataset",
    "dtype": "S50",
    "shape": [-1],
    "format": "email", 
    "$comment": "Array of email addresses - each element must be valid email"
}
```

### Advanced Examples

#### Multi-sensor data file schema:

```json
{
    "type": "group",
    "members": {
        "metadata": {
            "type": "group",
            "members": {
                "experiment_info": {
                    "type": "dataset",
                    "dtype": {
                        "formats": [
                            {"name": "start_time", "format": "<f8"},
                            {"name": "duration", "format": "<f8", "offset": 8},
                            {"name": "researcher", "format": "S64", "offset": 16}
                        ],
                        "itemsize": 80
                    },
                    "shape": [1]
                }
            }
        }
    },
    "patternMembers": {
        "^sensor_[0-9]+$": {
            "description": "Individual sensor data",
            "type": "group",
            "members": {
                "readings": {
                    "type": "dataset", 
                    "dtype": "<f8",
                    "shape": [-1, 3]
                },
                "timestamps": {
                    "type": "dataset",
                    "dtype": "<f8", 
                    "shape": [-1]
                }
            },
            "required": ["readings", "timestamps"],
            "attrs": [
                {"name": "sensor_type", "dtype": "U32", "required": true},
                {"name": "calibration_date", "dtype": "U32"}
            ]
        }
    },
    "required": ["metadata"],
    "attrs": [
        {"name": "schema_version", "dtype": "U10", "required": true},
        {"name": "created_by", "dtype": "U64"}
    ]
}
```

### Schema Validation Tips

1. **Start simple**: Begin with basic structure, add complexity gradually
2. **Use pattern members**: For dynamic content like `sensor_1`, `sensor_2`, etc.
3. **Validate incrementally**: Test your schema on sample files as you develop
4. **Document your schema**: Use `"description"` fields to explain complex patterns
5. **Consider optional vs required**: Only mark truly essential fields as required

## Error Handling and Troubleshooting

### Common Validation Errors

1. **Missing required members**: A required group/dataset is not present
   ```
   ValidationError: Required member 'data' not found in group '/'
   ```

2. **Type mismatches**: Dataset found where group expected, or vice versa
   ```
   ValidationError: Expected group at '/metadata' but found dataset
   ```

3. **Shape mismatches**: Dataset shape doesn't match schema
   ```
   ValidationError: Shape mismatch for '/data': expected [100, 3], got [100, 4]
   ```

4. **Data type mismatches**: Incorrect dtype
   ```
   ValidationError: Dtype mismatch for '/values': expected '<f8', got '<f4'
   ```

### Debugging Tips

1. **Use verbose validation** to see all errors:
   ```python
   validator = Hdf5Validator("file.h5", "schema.json")
   for error in validator.iter_errors():
       print(f"Path: {error.path}, Message: {error.message}")
   ```

2. **Generate a schema** from your HDF5 file to understand its structure:
   ```bash
   python -m hdf5schema.generate_schema your_file.h5 --pretty
   ```

3. **Validate step by step** by creating schemas for individual groups first

4. **Check your regex patterns** - use online regex testers for complex patterns

### Performance Considerations

- Large files: Validation time scales with file size and complexity
- Pattern matching: Complex regex patterns may slow validation
- Memory usage: Very large datasets are not loaded into memory during validation

## Running the Test Suite

```bash
python -m pytest
```

## About

The author of this Python package is Stephen Hartzell. Stephen got tired of writing one-off solutions to compare HDF5 files to a schema and wrote this package to solve this problem for himself and hopefully others.

`hdf5schema` is on [GitHub](https://github.com/python-hdf5schema/hdf5schema).

Get in touch, via GitHub or otherwise, if you've got something to contribute, it'd be most welcome!