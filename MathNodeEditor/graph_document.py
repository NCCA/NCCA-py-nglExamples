"""Validation for saved MathNodeEditor graph documents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

from math_graph import (
    DEFAULT_TRANSFORM_ROTATION_ORDER,
    GENERATOR_OPERATIONS,
    MESH_VIEWER_INPUT_NAMES,
    OPERATION_ARITY,
    OPERATION_PARAMETER_TYPES,
    TRANSFORM_ROTATION_ORDERS,
    TYPE_SHAPES,
    GraphError,
    MathType,
    Operation,
)

SCHEMA_VERSION = 1
# A whole new node kind (see the last paragraph of "Adding a new operation"
# in math_graph.py) needs a name added here and a matching case in
# _validate_node and _input_count below, or saved graphs containing it fail
# validate_document before they even reach the canvas.
NODE_KINDS = frozenset(
    {"value", "operation", "generator", "output", "obj_loader", "mesh_viewer"}
)
SHADING_MODES = frozenset({"Solid Colour", "Diffuse"})


def _sequence(value: object, label: str) -> Sequence[object]:
    """Return a list-like value, or report its document field name."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise GraphError(f"{label} must be a list")
    return value


def _required(entry: Mapping[str, object], field: str, node_id: str) -> object:
    """Return a required node field with a useful error if it is absent."""
    if field not in entry:
        raise GraphError(f"Node {node_id!r} is missing {field!r}")
    return entry[field]


def _number(value: object, label: str) -> float:
    """Return a finite JSON number, rejecting booleans and infinities."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise GraphError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise GraphError(f"{label} must be a finite number") from error
    if not isfinite(number):
        raise GraphError(f"{label} must be a finite number")
    return number


def _named_enum(
    entry: Mapping[str, object],
    field: str,
    enum_type: type[MathType] | type[Operation],
    label: str,
    node_id: str,
) -> MathType | Operation:
    """Read an enum member saved by name rather than display text."""
    value = _required(entry, field, node_id)
    if not isinstance(value, str):
        raise GraphError(f"Unknown {label} {value!r}")
    try:
        return enum_type[value]
    except KeyError as error:
        raise GraphError(f"Unknown {label} {value!r}") from error


def _components(value: object, math_type: MathType, label: str) -> None:
    """Validate one scalar, vector, matrix or quaternion component list."""
    values = _sequence(value, label)
    rows, columns = TYPE_SHAPES[math_type]
    expected_count = rows * columns
    if len(values) != expected_count:
        raise GraphError(f"{math_type.value} needs {expected_count} components")
    for component in values:
        _number(component, f"{label} component")


def _vectors(value: object, columns: int, label: str) -> None:
    """Validate a saved list of fixed-width numeric vectors."""
    for index, vector in enumerate(_sequence(value, label)):
        components = _sequence(vector, f"{label} entry {index}")
        if len(components) != columns:
            raise GraphError(f"{label} entry {index} needs {columns} components")
        for component in components:
            _number(component, f"{label} entry {index} component")


def _corner_index(value: object, label: str, optional: bool) -> None:
    """Validate one Obj corner index without yet checking array bounds."""
    if optional and value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise GraphError(f"{label} must be a non-negative integer")


def _faces(value: object) -> None:
    """Validate the nested triangle/corner shape stored by an Obj Loader."""
    for face_index, triangle in enumerate(_sequence(value, "Obj Loader faces")):
        corners = _sequence(triangle, f"Obj Loader face {face_index}")
        if len(corners) != 3:
            raise GraphError(f"Obj Loader face {face_index} needs three corners")
        for corner_index, corner in enumerate(corners):
            indices = _sequence(
                corner, f"Obj Loader face {face_index} corner {corner_index}"
            )
            if len(indices) != 3:
                raise GraphError(
                    f"Obj Loader face {face_index} corner {corner_index} "
                    "needs three indices"
                )
            label = f"Obj Loader face {face_index} corner {corner_index}"
            _corner_index(indices[0], f"{label} vertex index", optional=False)
            _corner_index(indices[1], f"{label} UV index", optional=True)
            _corner_index(indices[2], f"{label} normal index", optional=True)


def _operation(entry: Mapping[str, object], node_id: str) -> Operation:
    """Return the operation selected by an operation or generator node."""
    operation = _named_enum(entry, "operation", Operation, "operation", node_id)
    assert isinstance(operation, Operation)
    return operation


def _validate_node(entry: Mapping[str, object], node_id: str) -> None:
    """Validate every field read while rebuilding one graphics node."""
    if "name" in entry and (
        not isinstance(entry["name"], str) or not entry["name"].strip()
    ):
        raise GraphError(f"Node {node_id!r} name must be non-empty text")
    _number(_required(entry, "x", node_id), f"Node {node_id!r} x")
    _number(_required(entry, "y", node_id), f"Node {node_id!r} y")
    kind = entry["kind"]

    if kind == "value":
        math_type = _named_enum(entry, "math_type", MathType, "math type", node_id)
        assert isinstance(math_type, MathType)
        _components(
            _required(entry, "components", node_id),
            math_type,
            f"Node {node_id!r} components",
        )
    elif kind == "operation":
        operation = _operation(entry, node_id)
        if operation in GENERATOR_OPERATIONS:
            raise GraphError(f"{operation.value} must be a generator node")
    elif kind == "generator":
        operation = _operation(entry, node_id)
        if operation not in GENERATOR_OPERATIONS:
            raise GraphError(f"{operation.value} must be an operation node")
        parameters = _sequence(
            _required(entry, "parameters", node_id),
            f"Node {node_id!r} parameters",
        )
        parameter_types = OPERATION_PARAMETER_TYPES[operation]
        if len(parameters) != len(parameter_types):
            raise GraphError(
                f"{operation.value} needs {len(parameter_types)} parameters"
            )
        for index, (parameter, math_type) in enumerate(
            zip(parameters, parameter_types, strict=True)
        ):
            _components(parameter, math_type, f"Parameter {index}")
        if operation is Operation.TRANSFORM:
            rotation_order = entry.get(
                "rotation_order", DEFAULT_TRANSFORM_ROTATION_ORDER
            )
            if (
                not isinstance(rotation_order, str)
                or rotation_order not in TRANSFORM_ROTATION_ORDERS
            ):
                raise GraphError(f"Unknown Transform rotation order {rotation_order!r}")
        elif "rotation_order" in entry:
            raise GraphError(f"{operation.value} does not use a rotation order")
    elif kind == "obj_loader":
        array_ids = _sequence(
            _required(entry, "array_ids", node_id), "Obj Loader array ids"
        )
        if len(array_ids) != 4:
            raise GraphError("Obj Loader needs four array ids")
        if any(not isinstance(array_id, str) or not array_id for array_id in array_ids):
            raise GraphError("Obj Loader array ids must be non-empty strings")
        if len(set(array_ids)) != len(array_ids):
            raise GraphError("Obj Loader array ids must be unique")
        if array_ids[0] != node_id:
            raise GraphError("Obj Loader id must match its Vertices array id")
        _vectors(_required(entry, "vertices", node_id), 3, "Obj Loader vertices")
        _faces(_required(entry, "faces", node_id))
        _vectors(_required(entry, "uvs", node_id), 2, "Obj Loader UVs")
        _vectors(_required(entry, "normals", node_id), 3, "Obj Loader normals")
        status = entry.get("status", "")
        if not isinstance(status, str):
            raise GraphError("Obj Loader status must be text")
    elif kind == "mesh_viewer":
        shading_mode = entry.get("shading_mode", "Solid Colour")
        if shading_mode not in SHADING_MODES:
            raise GraphError(f"Unknown shading mode {shading_mode!r}")
        if "wireframe" in entry and not isinstance(entry["wireframe"], bool):
            raise GraphError("Mesh Viewer wireframe must be true or false")


def _input_count(entry: Mapping[str, object]) -> int:
    """Return the number of sockets which accept connections on a node."""
    kind = entry["kind"]
    if kind == "output":
        return 1
    if kind == "mesh_viewer":
        return len(MESH_VIEWER_INPUT_NAMES)
    if kind == "operation":
        return OPERATION_ARITY[_operation(entry, str(entry["id"]))]
    return 0


def validate_document(data: object) -> None:
    """Check a document completely before it replaces the current scene."""
    if not isinstance(data, Mapping):
        raise GraphError("Document must be an object")
    version = data.get("schema_version")
    if "schema_version" in data and (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != SCHEMA_VERSION
    ):
        raise GraphError(f"Unsupported graph schema version {version!r}")
    for field in ("nodes", "connections"):
        if field not in data:
            raise GraphError(f"Document is missing {field!r}")

    nodes = _sequence(data["nodes"], "Document nodes")
    node_ids: set[str] = set()
    node_entries: dict[str, Mapping[str, object]] = {}
    for entry in nodes:
        if not isinstance(entry, Mapping):
            raise GraphError("Each document node must be an object")
        node_id = entry.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise GraphError("Node id must be a non-empty string")
        if node_id in node_ids:
            raise GraphError(f"Duplicate node id {node_id!r}")
        kind = entry.get("kind")
        if kind not in NODE_KINDS:
            raise GraphError(f"Unknown node kind {kind!r}")
        node_ids.add(node_id)
        node_entries[node_id] = entry
        _validate_node(entry, node_id)

    source_ids: set[str] = set()
    source_owners: dict[str, str] = {}
    owned_array_ids: set[str] = set()
    for node_id, entry in node_entries.items():
        kind = entry["kind"]
        if kind in {"value", "operation", "generator"}:
            source_ids.add(node_id)
            source_owners[node_id] = node_id
        elif kind == "obj_loader":
            for index, array_id in enumerate(entry["array_ids"]):
                if array_id in owned_array_ids:
                    raise GraphError(f"Duplicate Obj Loader array id {array_id!r}")
                if index > 0 and array_id in node_ids:
                    raise GraphError(
                        f"Obj Loader array id {array_id!r} is already in use"
                    )
                owned_array_ids.add(array_id)
                source_ids.add(array_id)
                source_owners[array_id] = node_id

    connections = _sequence(data["connections"], "Document connections")
    occupied_inputs: set[tuple[str, int]] = set()
    for connection in connections:
        if not isinstance(connection, Mapping):
            raise GraphError("Each document connection must be an object")
        source_id = connection.get("source")
        if not isinstance(source_id, str) or not source_id:
            raise GraphError("Connection source must be a node id")
        if source_id not in source_ids:
            if source_id in node_ids:
                raise GraphError(f"Node {source_id!r} cannot be a connection source")
            raise GraphError(f"Unknown connection source {source_id!r}")
        target_id = connection.get("target")
        if not isinstance(target_id, str) or not target_id:
            raise GraphError("Connection target must be a node id")
        if target_id not in node_ids:
            raise GraphError(f"Unknown connection target {target_id!r}")
        if source_owners[source_id] == target_id:
            raise GraphError("A node cannot connect directly to itself")
        input_index = connection.get("input")
        input_count = _input_count(node_entries[target_id])
        if (
            not isinstance(input_index, int)
            or isinstance(input_index, bool)
            or not 0 <= input_index < input_count
        ):
            raise GraphError(
                f"Connection to {target_id!r} has invalid input {input_index!r}"
            )
        target_input = (target_id, input_index)
        if target_input in occupied_inputs:
            raise GraphError(
                f"Node {target_id!r} input {input_index} has multiple connections"
            )
        occupied_inputs.add(target_input)
