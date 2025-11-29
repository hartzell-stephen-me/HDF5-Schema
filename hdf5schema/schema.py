from dataclasses import dataclass
import json
from jsonschema import validate
import numpy as np
import pathlib
import re
from typing import Dict, List, Tuple, Union
from hdf5schema.exceptions import SchemaError
from hdf5schema.schemas import GROUP_META_SCHEMA, DATASET_META_SCHEMA


def resolve_ref(ref_path: str, root_schema: Dict) -> Dict:
    """
    Resolve a JSON Schema $ref reference.

    Parameters
    ----------
        ref_path: str
            The reference path (e.g., "#/$defs/observable")
        root_schema: Dict
            The root schema dictionary

    Returns
    -------
        The resolved schema definition

    """
    if not ref_path.startswith("#/"):
        raise SchemaError(f"Only local references supported, got: {ref_path}")

    # Remove the "#/" prefix and split the path
    path_parts = ref_path[2:].split("/")

    # Navigate through the schema
    current = root_schema
    for part in path_parts:
        if part not in current:
            raise SchemaError(f"Reference path not found: {ref_path}")
        current = current[part]

    return current


@dataclass
class Schema:

    schema: Dict
    selector: re.Pattern = None
    parent: Union["GroupSchema", None] = None
    root_schema: Dict = None

    def __post_init__(self):
        if type(self.selector) == str:
            self.selector = re.compile(self.selector)

    @property
    def attrs(self) -> Dict:
        attrs_dict = {}
        if "attrs" in self.schema:
            for attr in self.schema["attrs"]:
                attrs_dict[attr["name"]] = {
                    "dtype": attr["dtype"],
                }
                if "shape" in attr:
                    attrs_dict[attr["name"]]["shape"] = attr["shape"]
                if "required" in attr:
                    attrs_dict[attr["name"]]["required"] = attr["required"]
        return attrs_dict

    @property
    def path(self) -> str:
        rev_path = []
        if self.parent is not None:
            rev_path.append(self.name)
        ancestor = self.parent
        while ancestor is not None:
            # Use string form of ancestor selector for path
            if hasattr(ancestor, "name"):
                rev_path.append(ancestor.name)
            else:
                rev_path.append(str(ancestor.selector))
            ancestor = ancestor.parent
        return "/" + "/".join(rev_path[::-1])

    @property
    def required(self) -> bool:
        if self.parent is None or ("required" in self.parent.schema) and (self.name in self.parent.schema["required"]):
            return True
        return False

    @property
    def type(self) -> str:
        return self.schema["type"]

    @property
    def enum(self) -> List:
        """Return enum values if they exist."""
        return self.schema.get("enum", [])

    def has_enum(self) -> bool:
        """Check if this schema has enum constraints."""
        return "enum" in self.schema

    @property
    def const(self):
        """Return const value if it exists."""
        return self.schema.get("const")

    def has_const(self) -> bool:
        """Check if this schema has const constraint."""
        return "const" in self.schema

    @property
    def comment(self) -> str:
        """Return $comment if it exists."""
        return self.schema.get("$comment", "")

    @property
    def format(self) -> str:
        """Return format validation string if it exists."""
        return self.schema.get("format", "")

    def has_format(self) -> bool:
        """Check if this schema has format validation."""
        return "format" in self.schema

    @property
    def min_length(self) -> int:
        """Return minimum string length if specified."""
        return self.schema.get("minLength")

    def has_min_length(self) -> bool:
        """Check if this schema has minimum length constraint."""
        return "minLength" in self.schema

    @property
    def max_length(self) -> int:
        """Return maximum string length if specified."""
        return self.schema.get("maxLength")

    def has_max_length(self) -> bool:
        """Check if this schema has maximum length constraint."""
        return "maxLength" in self.schema

    @property
    def pattern(self) -> str:
        """Return regex pattern if specified."""
        return self.schema.get("pattern", "")

    def has_pattern(self) -> bool:
        """Check if this schema has pattern constraint."""
        return "pattern" in self.schema

    @property
    def name(self) -> str:
        """
        Human-friendly name for this schema node.

        For literal members, this is the exact member key. For pattern members,
        this is the regex pattern string.
        """
        if isinstance(self.selector, re.Pattern):
            return self.selector.pattern
        return str(self.selector)

@dataclass
class GroupSchema(Schema):

    _resolution_stack: set = None

    def __post_init__(self):
        self._members = []
        self._any_of_schemas = []  # Store anyOf alternatives
        self._all_of_schemas = []  # Store allOf schemas
        self._one_of_schemas = []  # Store oneOf schemas
        self._not_schema = None  # Store not schema
        self._if_schema = None  # Store if condition
        self._then_schema = None  # Store then consequence
        self._else_schema = None  # Store else alternative
        self._dependent_required = {}  # Store dependentRequired
        self._dependent_schemas = {}  # Store dependentSchemas

        # Initialize resolution stack for cycle detection
        if self._resolution_stack is None:
            self._resolution_stack = set()

        # If this is the root schema (no parent), set itself as root_schema
        if self.parent is None and self.root_schema is None:
            self.root_schema = self.schema

        # Handle anyOf at the group level
        if "anyOf" in self.schema:
            for alt_schema in self.schema["anyOf"]:
                # Create alternative GroupSchema instances
                alt_group_schema = GroupSchema(alt_schema, self.selector, parent=self.parent)
                self._any_of_schemas.append(alt_group_schema)
            return super().__post_init__()

        # Handle allOf at the group level
        if "allOf" in self.schema:
            for all_schema in self.schema["allOf"]:
                # Create GroupSchema instances that must all be satisfied
                # Ensure the allOf schema has the group type
                if "type" not in all_schema:
                    all_schema = {"type": "group", **all_schema}
                all_group_schema = GroupSchema(all_schema, self.selector, parent=self.parent)
                self._all_of_schemas.append(all_group_schema)
            return super().__post_init__()

        # Handle oneOf at the group level
        if "oneOf" in self.schema:
            for one_schema in self.schema["oneOf"]:
                # Create GroupSchema instances where exactly one must be satisfied
                # Ensure the oneOf schema has the group type
                if "type" not in one_schema:
                    one_schema = {"type": "group", **one_schema}
                one_group_schema = GroupSchema(one_schema, self.selector, parent=self.parent)
                self._one_of_schemas.append(one_group_schema)
            return super().__post_init__()

        # Handle not at the group level
        if "not" in self.schema:
            not_schema = self.schema["not"]
            # Ensure the not schema has the group type
            if "type" not in not_schema:
                not_schema = {"type": "group", **not_schema}
            self._not_schema = GroupSchema(not_schema, self.selector, parent=self.parent)
            return super().__post_init__()

        # Handle conditional schemas (if/then/else)
        if "if" in self.schema:
            if_schema = self.schema["if"]
            # Ensure the if schema has the group type
            if "type" not in if_schema:
                if_schema = {"type": "group", **if_schema}
            self._if_schema = GroupSchema(if_schema, self.selector, parent=self.parent)

            if "then" in self.schema:
                then_schema = self.schema["then"]
                if "type" not in then_schema:
                    then_schema = {"type": "group", **then_schema}
                self._then_schema = GroupSchema(then_schema, self.selector, parent=self.parent)

            if "else" in self.schema:
                else_schema = self.schema["else"]
                if "type" not in else_schema:
                    else_schema = {"type": "group", **else_schema}
                self._else_schema = GroupSchema(else_schema, self.selector, parent=self.parent)

            return super().__post_init__()

        # Handle dependentRequired
        if "dependentRequired" in self.schema:
            self._dependent_required = self.schema["dependentRequired"]

        # Handle dependentSchemas
        if "dependentSchemas" in self.schema:
            for prop_name, dependent_schema in self.schema["dependentSchemas"].items():
                # Ensure the dependent schema has the group type if not specified
                if "type" not in dependent_schema:
                    dependent_schema = {"type": "group", **dependent_schema}
                self._dependent_schemas[prop_name] = GroupSchema(dependent_schema, self.selector, parent=self.parent)

        for member_type in ["members", "patternMembers"]:
            if member_type not in self.schema:
                continue

            # Standard explicit members
            if member_type == "members":
                required_members = []
                member_selectors = []
                for member_selector, member_schema in self.schema[member_type].items():
                    member_selectors.append(member_selector)
                    if member_selector == "required":
                        required_members = member_schema
                        continue

                    member = self._create_member(member_schema, member_selector)
                    self._members.append(member)

                # Check that all required members exist
                if not set(required_members).issubset(set(member_selectors)):
                    raise SchemaError(f"Group {self.path} is missing required members")

            # Pattern members with regex selectors (e.g. patternMembers)
            elif member_type == "patternMembers":
                for pattern, member_schema in self.schema[member_type].items():
                    # patternMembers may specify an anyOf list of possible schemas
                    if "anyOf" in member_schema:
                        for alt_schema in member_schema["anyOf"]:

                            member = self._create_member(alt_schema, pattern)
                            self._members.append(member)
                    else:
                        member = self._create_member(member_schema, pattern)
                        self._members.append(member)

        return super().__post_init__()


    def _create_member(self, member_schema: Dict, member_selector: str) -> Union["GroupSchema", "DatasetSchema", "RefSchema"]:
        """
        Create a member schema instance from its schema and selector.

        Parameters
        ----------
        member_schema: dict
            The schema dictionary for the member
        member_selector: str
            The regex pattern string or literal name for the member

        Returns
        -------
        Union["GroupSchema", "DatasetSchema", "RefSchema"]:
            The created member schema instance.

        """
        # Handle $ref by creating a RefSchema (lazy resolution)
        if "$ref" in member_schema:
            member = RefSchema(member_schema, member_selector, parent=self, root_schema=self.root_schema or self._get_root_schema())
            # Propagate resolution stack for cycle detection
            if hasattr(self, "_resolution_stack") and self._resolution_stack:
                member._resolution_stack = self._resolution_stack.copy()
            return member
        else:
            if "type" not in member_schema:
                raise SchemaError(f"Group {self.path} Member {member_selector} doesn't have a type")
            if member_schema["type"] == "group":
                return GroupSchema(member_schema, member_selector, parent=self, root_schema=self.root_schema or self._get_root_schema())
            elif member_schema["type"] == "dataset":
                return DatasetSchema(member_schema, member_selector, parent=self, root_schema=self.root_schema or self._get_root_schema())
            else:
                raise SchemaError(
                    f"Group {self.path}: Member {member_selector} has unknown type {member_schema['type']}"
                )

    def _get_root_schema(self) -> Dict:
        """Get the root schema by traversing up the parent chain."""
        current = self
        while current.parent is not None:
            current = current.parent
        return current.schema

    def _pattern_specificity(self, pattern: re.Pattern, target_name: str) -> int:
        """
        Calculate pattern specificity for prioritizing matches.

        Higher values indicate more specific matches.
        Exact matches get highest priority, then patterns by decreasing specificity.
        This is for resolving multiple pattern matches within pattern members.

        Parameters
        ----------
        pattern: re.Pattern
            The regex pattern to evaluate.
        target_name: str
            The target name to match against the pattern.

        Returns
        -------
        int:
            The specificity score for the pattern.

        """
        pattern_str = pattern.pattern

        # Exact string match gets highest priority
        if pattern_str == target_name:
            return 1000

        # Count non-metacharacter specificity indicators
        specificity = 0

        # Longer patterns are generally more specific
        specificity += len(pattern_str)

        # Patterns with anchors (^, $) are more specific
        if pattern_str.startswith("^"):
            specificity += 50
        if pattern_str.endswith("$"):
            specificity += 50

        # Patterns with literal characters are more specific than pure wildcards
        metachar_set = r".\*+?[](){}|^$"
        literal_chars = len([c for c in pattern_str if c not in metachar_set])
        specificity += literal_chars * 10

        # Penalize overly generic patterns
        if pattern_str in [".*", ".+", ".*?", ".+?"]:
            specificity -= 100

        return specificity

    def __contains__(self, name: str) -> bool:
        return any(member.selector.match(name) for member in self.members)

    def __getitem__(self, name: str) -> Union["GroupSchema", "DatasetSchema", None]:
        """Return the member matching name, prioritizing more specific patterns."""
        matching_items = []
        for member in self.members:
            if member.selector.match(name):
                matching_items.append(member)

        if len(matching_items) == 0:
            return None
        elif len(matching_items) == 1:
            return matching_items[0]
        else:
            # Multiple matches - group by pattern specificity
            patterns_by_specificity = {}
            for member in matching_items:
                specificity = self._pattern_specificity(member.selector, name)
                if specificity not in patterns_by_specificity:
                    patterns_by_specificity[specificity] = []
                patterns_by_specificity[specificity].append(member)

            # Get the highest specificity group
            max_specificity = max(patterns_by_specificity.keys())
            most_specific_matches = patterns_by_specificity[max_specificity]

            if len(most_specific_matches) == 1:
                return most_specific_matches[0]
            else:
                # Multiple matches with same specificity (e.g., anyOf alternatives)
                return most_specific_matches

    def __iter__(self) -> List[Union["GroupSchema", "DatasetSchema"]]:
        return iter(self.members)

    @classmethod
    def from_file(cls, schema: Union[pathlib.Path, str]):
        with open(schema) as fid:
            schema = json.load(fid)
        validate(instance=schema, schema=GROUP_META_SCHEMA)
        return cls(schema, "/")

    @property
    def members(self) -> List[Union["GroupSchema", "DatasetSchema"]]:
        return self._members

    @property
    def any_of_schemas(self) -> List["GroupSchema"]:
        """Return anyOf alternatives if they exist."""
        return getattr(self, "_any_of_schemas", [])

    def has_any_of(self) -> bool:
        """Check if this schema has anyOf alternatives."""
        return len(self.any_of_schemas) > 0

    @property
    def all_of_schemas(self) -> List["GroupSchema"]:
        """Return allOf schemas if they exist."""
        return getattr(self, "_all_of_schemas", [])

    def has_all_of(self) -> bool:
        """Check if this schema has allOf schemas."""
        return len(self.all_of_schemas) > 0

    @property
    def one_of_schemas(self) -> List["GroupSchema"]:
        """Return oneOf schemas if they exist."""
        return getattr(self, "_one_of_schemas", [])

    def has_one_of(self) -> bool:
        """Check if this schema has oneOf schemas."""
        return len(self.one_of_schemas) > 0

    @property
    def not_schema(self) -> "GroupSchema":
        """Return not schema if it exists."""
        return getattr(self, "_not_schema", None)

    def has_not(self) -> bool:
        """Check if this schema has not schema."""
        return self.not_schema is not None

    @property
    def if_schema(self) -> "GroupSchema":
        """Return if condition schema if it exists."""
        return getattr(self, "_if_schema", None)

    @property
    def then_schema(self) -> "GroupSchema":
        """Return then consequence schema if it exists."""
        return getattr(self, "_then_schema", None)

    @property
    def else_schema(self) -> "GroupSchema":
        """Return else alternative schema if it exists."""
        return getattr(self, "_else_schema", None)

    def has_conditional(self) -> bool:
        """Check if this schema has conditional (if/then/else) logic."""
        return self.if_schema is not None

    @property
    def dependent_required(self) -> dict:
        """Return dependentRequired constraints if they exist."""
        return getattr(self, "_dependent_required", {})

    def has_dependent_required(self) -> bool:
        """Check if this schema has dependentRequired constraints."""
        return len(self.dependent_required) > 0

    @property
    def dependent_schemas(self) -> dict:
        """Return dependentSchemas if they exist."""
        return getattr(self, "_dependent_schemas", {})

    def has_dependent_schemas(self) -> bool:
        """Check if this schema has dependentSchemas."""
        return len(self.dependent_schemas) > 0

    def validate(self):
        validate(instance=self.schema, schema=GROUP_META_SCHEMA)
        for member in self.members:
            member.validate()


@dataclass
class DatasetSchema(Schema):

    @property
    def dtype(self) -> Union[np.dtype, None]:
        if "dtype" not in self.schema:
            return None
        if type(self.schema["dtype"]) == str:
            return np.dtype(self.schema["dtype"])
        # Support list of field definitions: [{"name": "field", "dtype": "<f8"}, ...]
        if isinstance(self.schema["dtype"], list):
            names = []
            formats = []
            for field in self.schema["dtype"]:
                if ("name" not in field) or ("dtype" not in field):
                    raise ValueError("List dtype entries must have 'name' and 'dtype'")
                names.append(field["name"])
                formats.append(field["dtype"])
            return np.dtype({"names": names, "formats": formats})
        elif type(self.schema["dtype"]) == dict:
            dtype_kwargs = {"names": [], "formats": []}
            offsets: List[Union[int, None]] = []
            any_offsets = False
            for fmt in self.schema["dtype"]["formats"]:
                dtype_kwargs["names"].append(fmt["name"])
                dtype_kwargs["formats"].append(fmt["format"])
                if "offset" in fmt:
                    any_offsets = True
                    offsets.append(fmt["offset"])
                else:
                    offsets.append(None)
            # Only include offsets if every field provided one
            if any_offsets and all(o is not None for o in offsets):
                dtype_kwargs["offsets"] = [int(o) for o in offsets]  # type: ignore[arg-type]
            if "itemsize" in self.schema["dtype"]:
                dtype_kwargs["itemsize"] = self.schema["dtype"]["itemsize"]
            return np.dtype(dtype_kwargs)
        raise ValueError(f"Invalid schema dtype: {type(self.schema['type'])}")

    @property
    def shape(self) -> Union[Tuple, None]:
        if "shape" in self.schema:
            return self.schema["shape"]

    @property
    def enum(self) -> List:
        """Return enum values if they exist."""
        return self.schema.get("enum", [])

    def has_enum(self) -> bool:
        """Check if this schema has enum constraints."""
        return "enum" in self.schema

    @property
    def const(self):
        """Return const value if it exists."""
        return self.schema.get("const")

    def has_const(self) -> bool:
        """Check if this schema has const constraint."""
        return "const" in self.schema

    @property
    def comment(self) -> str:
        """Return $comment if it exists."""
        return self.schema.get("$comment", "")

    @property
    def not_schema(self) -> dict:
        """Return not schema if it exists."""
        return self.schema.get("not")

    def has_not(self) -> bool:
        """Check if this schema has not constraint."""
        return "not" in self.schema

    @property
    def if_schema(self) -> dict:
        """Return if condition schema if it exists."""
        return self.schema.get("if")

    @property
    def then_schema(self) -> dict:
        """Return then consequence schema if it exists."""
        return self.schema.get("then")

    @property
    def else_schema(self) -> dict:
        """Return else alternative schema if it exists."""
        return self.schema.get("else")

    def has_conditional(self) -> bool:
        """Check if this schema has conditional (if/then/else) logic."""
        return "if" in self.schema

    @property
    def dependent_required(self) -> dict:
        """Return dependentRequired constraints if they exist."""
        return self.schema.get("dependentRequired", {})

    def has_dependent_required(self) -> bool:
        """Check if this schema has dependentRequired constraints."""
        return "dependentRequired" in self.schema

    @property
    def dependent_schemas(self) -> dict:
        """Return dependentSchemas if they exist."""
        return self.schema.get("dependentSchemas", {})

    def has_dependent_schemas(self) -> bool:
        """Check if this schema has dependentSchemas."""
        return "dependentSchemas" in self.schema

    def validate(self):
        validate(instance=self.schema, schema=DATASET_META_SCHEMA)


@dataclass
class RefSchema(Schema):
    """
    A schema that represents a JSON Schema $ref reference.
    This class delays resolution until actual validation to prevent infinite recursion.
    """

    ref_path: str = None
    _resolved_schema: Schema = None
    _resolution_stack: set = None

    def __post_init__(self):
        # Extract the $ref path
        if "$ref" in self.schema:
            self.ref_path = self.schema["$ref"]

        # Initialize resolution stack for cycle detection
        if self._resolution_stack is None:
            self._resolution_stack = set()

        return super().__post_init__()

    def resolve(self, max_depth: int = 10) -> Schema:
        """
        Resolve the $ref to the actual schema, with cycle detection and depth limiting.

        Parameters
        ----------
        max_depth: int
            Maximum resolution depth to prevent infinite recursion

        Returns
        -------
        Schema:
            The resolved Schema instance.

        """
        if self._resolved_schema is not None:
            return self._resolved_schema

        # Check for resolution cycles
        if self.ref_path in self._resolution_stack:
            # For recursive references, return a minimal valid GroupSchema that represents the cyclic reference
            # This allows validation to proceed without infinite recursion
            root = self.root_schema or self._get_root_schema()
            resolved_schema_dict = resolve_ref(self.ref_path, root)
            stub_schema = GroupSchema({"type": "group", "members": {}, "description": "Recursive reference"},
                                    self.selector, self.parent, root)
            # Don't set resolution stack on stub to avoid further recursion
            return stub_schema

        if len(self._resolution_stack) >= max_depth:
            # Prevent infinite recursion by returning a stub
            return GroupSchema({"type": "group", "members": {}}, self.selector, self.parent, self.root_schema)

        # Add current ref to resolution stack
        self._resolution_stack.add(self.ref_path)

        try:
            # Resolve the reference
            root = self.root_schema or self._get_root_schema()
            resolved_schema_dict = resolve_ref(self.ref_path, root)

            # Create the appropriate schema type
            if resolved_schema_dict["type"] == "group":
                resolved = GroupSchema(resolved_schema_dict, self.selector, self.parent, root)
                # Pass resolution stack to prevent cycles in nested refs
                resolved._resolution_stack = self._resolution_stack.copy()
            elif resolved_schema_dict["type"] == "dataset":
                resolved = DatasetSchema(resolved_schema_dict, self.selector, self.parent, root)
            else:
                raise SchemaError(f"Unknown schema type in $ref resolution: {resolved_schema_dict['type']}")

            self._resolved_schema = resolved
            return resolved

        finally:
            # Remove current ref from resolution stack
            self._resolution_stack.discard(self.ref_path)

    @property
    def type(self) -> str:
        return self.resolve().type

    @property
    def name(self) -> str:
        return self.resolve().name

    def _get_root_schema(self) -> Dict:
        """Get the root schema by traversing up the parent chain."""
        current = self
        while current.parent is not None:
            current = current.parent
        return current.root_schema or current.schema
