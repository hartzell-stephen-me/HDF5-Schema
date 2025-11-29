import h5py
import pathlib
from typing import Union
from hdf5schema.validator import Hdf5Validator
import typer
from pathlib import Path

def validate(
    instance: Union[pathlib.Path, str, h5py.File, h5py.Group],
    schema: Union[pathlib.Path, str, dict],
    validator: Hdf5Validator = None
) -> bool:
    """
    Validate an HDF5 file from a given schema.

    Parameters
    ----------
    instance: Union[pathlib.Path, str, h5py.File, h5py.Group]
        HDF5 file to validate against the provided schema
    schema: Union[pathlib.Path, str, dict]
        Path to or content of (dict) the schema to validate against
    validator: Hdf5Validator, optional
        Validator class to use. Defaults to the default Hdf5Validator

    Returns
    -------
    bool:
        True if the instance is valid. Otherwise raises a ValidationError.

    """
    if validator is None:
        validator = Hdf5Validator(instance=instance, schema=schema)
    else:
        validator.schema = schema
        validator.instance = instance
        # Reinitialize the class
        validator.__post_init__()

    return validator.is_valid()

def main(
    hdf5_file: str = typer.Argument(..., help="Path to the HDF5 file to validate"),
    schema_file: str = typer.Argument(..., help="Path to the JSON schema file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed error information"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output except errors")
) -> None:
    """Validate HDF5 file against JSON schema."""
    # Check if files exist
    if not Path(hdf5_file).exists():
        typer.echo(f"HDF5 file not found: {hdf5_file}", err=True)
        raise typer.Exit(1)

    if not Path(schema_file).exists():
        typer.echo(f"Schema file not found: {schema_file}", err=True)
        raise typer.Exit(1)

    if not quiet:
        typer.echo(f"Validating {hdf5_file} against {schema_file}...")

    try:
        validator = Hdf5Validator(hdf5_file, schema_file)
        errors = list(validator.iter_errors())

        if errors:
            typer.echo(f"Found {len(errors)} validation errors:", err=True)
            for i, error in enumerate(errors, 1):
                if verbose:
                    typer.echo(f"  {i}. {error}", err=True)
                else:
                    error_str = str(error)
                    if len(error_str) > 100:
                        error_str = error_str[:100] + "..."
                    typer.echo(f"  {i}. {error_str}", err=True)
            raise typer.Exit(1)
        else:
            if not quiet:
                typer.echo("Validation passed - no errors found!")

    except Exception as e:
        typer.echo(f"Error during validation: {e}", err=True)
        if verbose:
            import traceback
            typer.echo(traceback.format_exc(), err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    typer.run(main)
