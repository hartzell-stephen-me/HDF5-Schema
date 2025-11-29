"""
This is a good starting point for generating a JSON schema from an HDF5 file.
It covers datasets and groups, including attributes, dtypes (simple and compound),
but does not work with creating more complex schema features like patterns,
optional members, etc. It is intended to be a foundation that can be extended
as needed.
"""
import json
import pathlib
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import typer


def _dtype_to_schema(dtype: np.dtype) -> Any:
    """
    Convert a NumPy dtype to the schema's dtype representation.

    Returns either a dtype string (e.g., "<f8", "S128") or a compound dtype
    object: {"formats": [{"name", "format", "offset"?}, ...], "itemsize"?}.
    """
    # Structured/compound dtype
    if dtype.fields:
        formats: List[Dict[str, Any]] = []
        for name, (subdtype, offset) in dtype.fields.items():
            entry: Dict[str, Any] = {
                "name": name,
                "format": _dtype_to_schema(np.dtype(subdtype)),
            }
            # Only include offsets if present and non-zero
            if isinstance(offset, int) and offset:
                entry["offset"] = offset
            formats.append(entry)

        obj: Dict[str, Any] = {"formats": formats}
        # Preserve explicit itemsize if padding exists
        obj["itemsize"] = int(dtype.itemsize)
        return obj

    # Simple dtype -> normalized string
    kind = dtype.kind
    if kind == "S":
        # Fixed-length ASCII bytes
        return f"S{dtype.itemsize}"
    if kind == "U":
        # Fixed-length Unicode (itemsize is bytes, 4 bytes per char)
        return f"U{dtype.itemsize // 4}"
    # Normalize to little-endian for portability if applicable
    try:
        return np.dtype(dtype).newbyteorder("<").str
    except Exception:
        return str(dtype)


def _attr_to_schema(name: str, value: Any) -> Dict[str, Any]:
    """Build an attribute schema entry from an HDF5 attribute value."""
    arr = np.asarray(value)
    entry: Dict[str, Any] = {
        "name": name,
        "dtype": _dtype_to_schema(arr.dtype),
    }
    # Only add shape if non-scalar
    if arr.shape:
        entry["shape"] = list(arr.shape)
    return entry


def _dataset_to_schema(dset: h5py.Dataset) -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        "type": "dataset",
        "dtype": _dtype_to_schema(dset.dtype),
        "shape": list(dset.shape),
    }
    # Attributes
    if len(dset.attrs) > 0:
        attrs = []
        for key in dset.attrs:
            try:
                attrs.append(_attr_to_schema(key, dset.attrs[key]))
            except Exception:
                # Best-effort: fallback to string dtype
                attrs.append({"name": key, "dtype": "str"})
        schema["attrs"] = attrs
    return schema


def _group_to_schema(group: h5py.Group) -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        "type": "group",
        "members": {},
    }
    # Group attributes
    if len(group.attrs) > 0:
        attrs = []
        for key in group.attrs:
            try:
                attrs.append(_attr_to_schema(key, group.attrs[key]))
            except Exception:
                attrs.append({"name": key, "dtype": "str"})
        schema["attrs"] = attrs

    # Members (explicit)
    required: List[str] = []
    for name, obj in group.items():
        if isinstance(obj, h5py.Group):
            schema["members"][name] = _group_to_schema(obj)
        elif isinstance(obj, h5py.Dataset):
            schema["members"][name] = _dataset_to_schema(obj)
        else:
            # Ignore other HDF5 object types for now (links, etc.)
            continue
        required.append(name)

    if required:
        schema["members"]["required"] = required
    return schema


def generate_schema(
    file_path: pathlib.Path, group_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a JSON-serializable schema dict from an HDF5 file/group.

    Parameters
    ----------
    file_path: Path to the HDF5 file
    group_path: Optional HDF5 path within the file to start from (default "/")

    """
    with h5py.File(str(file_path), "r") as f:
        grp = f[group_path] if group_path else f["/"]
        if isinstance(grp, h5py.Dataset):
            return _dataset_to_schema(grp)
        return _group_to_schema(grp)


def main(
    input_file: pathlib.Path = typer.Argument(..., help="Path to input .h5/.hdf5 file"),
    group: Optional[str] = typer.Option(None, "-g", "--group", help="HDF5 group path to use as root (default: /)"),
    output: Optional[pathlib.Path] = typer.Option(None, "-o", "--output", help="Write schema JSON to this file (default: stdout)"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON")
) -> None:
    """Generate schema from HDF5 file."""
    if not input_file.exists():
        typer.echo(f"Input file not found: {input_file}", err=True)
        raise typer.Exit(1)

    try:
        typer.echo(f"Generating schema from {input_file}...")
        schema = generate_schema(input_file, group)

        text = (
            json.dumps(schema, indent=2, ensure_ascii=False)
            if pretty
            else json.dumps(schema, separators=(",", ":"), ensure_ascii=False)
        )

        if output:
            output.write_text(text + "\n", encoding="utf-8")
            typer.echo(f"Schema written to {output}")
        else:
            typer.echo(text)

    except Exception as e:
        typer.echo(f"Error generating schema: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
