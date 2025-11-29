"""
Schema Generation Example
==========================

This example demonstrates automatic schema generation from existing HDF5 files.
This is useful for:
- Quickly creating initial schemas from reference files
- Documenting existing HDF5 file structures
- Creating templates that can be refined with additional constraints
"""
import gc
import h5py
import json
import numpy as np
import tempfile
from pathlib import Path

from hdf5schema.generate_schema import generate_schema
from hdf5schema.validator import Hdf5Validator


def create_complex_hdf5_file(filepath: Path) -> None:
    """Create a complex HDF5 file with nested groups and various data types."""
    with h5py.File(filepath, 'w') as f:
        # Root attributes
        f.attrs['version'] = '2.0'
        f.attrs['created_by'] = 'schema_generation_example'

        # Experiment metadata group
        metadata = f.create_group('metadata')
        metadata.attrs['experiment_id'] = 'EXP-2025-001'
        metadata.create_dataset('description', data='Long-term environmental monitoring')
        metadata.create_dataset('start_date', data='2025-01-01')
        metadata.create_dataset('duration_days', data=365, dtype='int32')

        # Results group with nested structure
        results = f.create_group('results')

        # Temperature measurements
        temp_group = results.create_group('temperature')
        temp_group.attrs['unit'] = 'celsius'
        temp_group.create_dataset('values', data=np.random.randn(100) * 5 + 20, dtype='float64')
        temp_group.create_dataset('timestamps', data=np.arange(100), dtype='int64')

        # Humidity measurements
        humid_group = results.create_group('humidity')
        humid_group.attrs['unit'] = 'percent'
        humid_group.create_dataset('values', data=np.random.randn(100) * 10 + 50, dtype='float64')
        humid_group.create_dataset('timestamps', data=np.arange(100), dtype='int64')

        # Compound dtype example
        compound_dtype = np.dtype([
            ('sensor_id', 'i4'),
            ('location', 'S32'),  # Use bytes instead of Unicode for HDF5 compatibility
            ('calibration_date', 'S16')
        ])
        sensor_info = np.array([
            (1, 'Building A - Room 101', '2025-01-01'),
            (2, 'Building B - Room 202', '2025-01-02')
        ], dtype=compound_dtype)
        f.create_dataset('sensors', data=sensor_info)


def main():
    """Run the schema generation example."""
    print("=" * 60)
    print("Schema Generation Example")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_file = Path(tmpdir) / "complex_data.h5"
        schema_file = Path(tmpdir) / "generated_schema.json"

        # Step 1: Create complex HDF5 file
        print("\n1. Creating complex HDF5 file...")
        create_complex_hdf5_file(hdf5_file)
        print(f"   Created: {hdf5_file}")

        # Show file structure
        print("\n2. HDF5 File Structure:")
        with h5py.File(hdf5_file, 'r') as f:
            def print_structure(name, obj):
                indent = "   " * (name.count('/') + 1)
                if isinstance(obj, h5py.Group):
                    print(f"{indent}[DIR] {name.split('/')[-1] or '/'}")
                    if obj.attrs:
                        for attr_name in obj.attrs:
                            print(f"{indent}  @{attr_name}: {obj.attrs[attr_name]}")
                elif isinstance(obj, h5py.Dataset):
                    print(f"{indent}[DATA] {name.split('/')[-1]} - shape: {obj.shape}, dtype: {obj.dtype}")
                    if obj.attrs:
                        for attr_name in obj.attrs:
                            print(f"{indent}  @{attr_name}: {obj.attrs[attr_name]}")

            print("   [DIR] / (root)")
            if f.attrs:
                for attr_name in f.attrs:
                    print(f"     @{attr_name}: {f.attrs[attr_name]}")
            f.visititems(print_structure)

        # Step 3: Generate schema
        print("\n3. Generating schema from HDF5 file...")
        schema = generate_schema(hdf5_file)

        # Save to file
        with open(schema_file, 'w') as f:
            json.dump(schema, f, indent=2)
        print(f"   Generated schema saved to: {schema_file}")

        # Step 4: Display generated schema (abbreviated)
        print("\n4. Generated Schema (sample):")
        print(json.dumps(schema, indent=2)[:1000] + "\n   ... (truncated)")

        # Step 5: Validate the original file against generated schema
        print("\n5. Validating original file against generated schema...")
        validator = Hdf5Validator(hdf5_file, schema)
        errors = list(validator.iter_errors())

        if errors:
            print(f"   [FAIL] Found {len(errors)} errors:")
            for error in errors:
                print(f"     - {error}")
        else:
            print("   [PASS] Validation passed! (Schema matches the source file)")

        del validator  # Clean up validator

        # Step 6: Test with a modified file (should fail validation)
        print("\n6. Testing schema with MODIFIED file...")
        modified_file = Path(tmpdir) / "modified_data.h5"

        with h5py.File(modified_file, 'w') as f:
            f.attrs['version'] = '2.0'
            f.attrs['created_by'] = 'schema_generation_example'
            # Missing metadata group - should fail validation
            results = f.create_group('results')
            temp_group = results.create_group('temperature')
            temp_group.create_dataset('values', data=np.random.randn(50), dtype='float64')

        validator = Hdf5Validator(modified_file, schema)
        errors = list(validator.iter_errors())

        if errors:
            print(f"   [PASS] Correctly detected {len(errors)} validation errors:")
            for error in errors[:3]:  # Show first 3 errors
                print(f"     - {error}")
            if len(errors) > 3:
                print(f"     ... and {len(errors) - 3} more errors")
        else:
            print("   [FAIL] No errors found (unexpected!)")

        # Step 7: Generate schema from specific group
        print("\n7. Generating schema from specific group path...")
        results_schema = generate_schema(hdf5_file, group_path='/results')
        print("   Generated schema for '/results' group:")
        print(json.dumps(results_schema, indent=2)[:500] + "\n   ... (truncated)")

        # Tips for schema refinement
        print("\n8. Schema Refinement Tips:")
        print("   - Generated schemas are basic and require manual refinement")
        print("   - Add pattern matching for dynamic member names")
        print("   - Mark optional members (generated schemas mark all as required)")
        print("   - Add conditional validation (if/then/else)")
        print("   - Add constraints (enum, const, min/max values)")
        print("   - Use $ref to avoid duplication in repeated structures")

        del validator  # Clean up last validator
        # Force garbage collection to ensure HDF5 files are closed
        gc.collect()

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
