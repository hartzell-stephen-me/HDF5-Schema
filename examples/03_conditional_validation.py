"""
Conditional Validation Example
===============================

This example demonstrates conditional schema validation using if/then/else
constructs to apply different validation rules based on data characteristics.
"""
import gc
import h5py
import tempfile
from pathlib import Path

from hdf5schema.validator import Hdf5Validator


def create_image_data(filepath: Path, data_type: str) -> None:
    """
    Create HDF5 file with image data.

    Parameters
    ----------
    filepath : Path
        Output HDF5 file path
    data_type : str
        Either 'rgb' for color images or 'grayscale' for single-channel
    """
    with h5py.File(filepath, 'w') as f:
        f.attrs['data_type'] = data_type
        f.attrs['image_count'] = 10

        if data_type == 'rgb':
            # RGB images: shape (10, 256, 256, 3)
            import numpy as np
            f.create_dataset(
                'images',
                data=np.random.randint(0, 256, size=(10, 256, 256, 3), dtype=np.uint8),
                dtype='uint8'
            )
            f['images'].attrs['channels'] = 3
            f['images'].attrs['color_space'] = 'RGB'
        else:
            # Grayscale images: shape (10, 256, 256)
            import numpy as np
            f.create_dataset(
                'images',
                data=np.random.randint(0, 256, size=(10, 256, 256), dtype=np.uint8),
                dtype='uint8'
            )
            f['images'].attrs['channels'] = 1


def create_conditional_schema() -> dict:
    """
    Create a schema with conditional validation.

    If data_type is 'rgb', then images must have shape (10, 256, 256, 3)
    and have a 'color_space' attribute.

    If data_type is 'grayscale', then images must have shape (10, 256, 256)
    and have only 1 channel.
    """
    return {
        "type": "group",
        "attrs": [
            {"name": "data_type", "dtype": "U128"},
            {"name": "image_count", "dtype": "<i8"}
        ],
        "members": {
            "images": {
                "type": "dataset",
                "dtype": "|u1"  # uint8
            }
        },
        "required": ["images"],
        # Conditional validation based on data_type attribute
        "if": {
            "attrs": [
                {"name": "data_type", "dtype": "U128", "const": "rgb"}
            ]
        },
        "then": {
            "members": {
                "images": {
                    "type": "dataset",
                    "dtype": "|u1",
                    "shape": [10, 256, 256, 3],
                    "attrs": [
                        {"name": "channels", "dtype": "<i8", "const": 3},
                        {"name": "color_space", "dtype": "U128"}
                    ]
                }
            }
        },
        "else": {
            "members": {
                "images": {
                    "type": "dataset",
                    "dtype": "|u1",
                    "shape": [10, 256, 256],
                    "attrs": [
                        {"name": "channels", "dtype": "<i8", "const": 1}
                    ]
                }
            }
        }
    }


def main():
    """Run the conditional validation example."""
    print("=" * 60)
    print("Conditional Schema Validation Example")
    print("=" * 60)

    schema = create_conditional_schema()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Validate RGB image data
        print("\n1. Validating RGB image data...")
        rgb_file = Path(tmpdir) / "rgb_images.h5"
        create_image_data(rgb_file, 'rgb')

        validator = Hdf5Validator(rgb_file, schema)
        errors = list(validator.iter_errors())

        if errors:
            print(f"   [FAIL] Found {len(errors)} errors:")
            for error in errors:
                print(f"     - {error}")
        else:
            print("   [PASS] RGB validation passed!")
            with h5py.File(rgb_file, 'r') as f:
                print(f"     - Shape: {f['images'].shape}")
                print(f"     - Data type: {f.attrs['data_type']}")
                print(f"     - Color space: {f['images'].attrs['color_space']}")

        del validator  # Clean up validator

        # Test 2: Validate grayscale image data
        print("\n2. Validating grayscale image data...")
        gray_file = Path(tmpdir) / "gray_images.h5"
        create_image_data(gray_file, 'grayscale')

        validator = Hdf5Validator(gray_file, schema)
        errors = list(validator.iter_errors())

        if errors:
            print(f"   [FAIL] Found {len(errors)} errors:")
            for error in errors:
                print(f"     - {error}")
        else:
            print("   [PASS] Grayscale validation passed!")
            with h5py.File(gray_file, 'r') as f:
                print(f"     - Shape: {f['images'].shape}")
                print(f"     - Data type: {f.attrs['data_type']}")
                print(f"     - Channels: {f['images'].attrs['channels']}")

        del validator  # Clean up validator

        # Test 3: Create invalid RGB data (wrong shape)
        print("\n3. Validating INVALID RGB data (wrong shape)...")
        invalid_file = Path(tmpdir) / "invalid_rgb.h5"

        import numpy as np
        with h5py.File(invalid_file, 'w') as f:
            f.attrs['data_type'] = 'rgb'
            f.attrs['image_count'] = 10
            # Wrong shape: missing channel dimension
            f.create_dataset('images', data=np.random.randint(0, 256, size=(10, 256, 256), dtype=np.uint8))
            f['images'].attrs['channels'] = 3
            f['images'].attrs['color_space'] = 'RGB'

        validator = Hdf5Validator(invalid_file, schema)
        errors = list(validator.iter_errors())

        if errors:
            print(f"   [PASS] Correctly detected {len(errors)} errors:")
            for error in errors:
                print(f"     - {error}")
        else:
            print("   [FAIL] No errors found (unexpected!)")

        # Explain the conditional logic
        print("\n4. Conditional Logic Explanation:")
        print("   IF data_type == 'rgb':")
        print("     THEN images must be shape (10, 256, 256, 3)")
        print("          and have 'color_space' attribute")
        print("   ELSE (grayscale):")
        print("     images must be shape (10, 256, 256)")
        print("     and have channels == 1")

        del validator  # Clean up last validator
        # Force garbage collection to ensure HDF5 files are closed
        gc.collect()

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
