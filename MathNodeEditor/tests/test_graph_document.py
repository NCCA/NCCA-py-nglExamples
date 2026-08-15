"""Tests for saved MathNodeEditor document validation."""

import json
import sys
from importlib import import_module
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _graph_document_module():
    """Load the graph document module from the demo directory."""
    return import_module("graph_document")


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ([], "Document must be an object"),
        ({}, "Document is missing 'nodes'"),
        ({"nodes": []}, "Document is missing 'connections'"),
        ({"nodes": {}, "connections": []}, "Document nodes must be a list"),
    ],
)
def test_document_rejects_an_invalid_top_level_shape(
    document: object, message: str
) -> None:
    graph_document = _graph_document_module()

    with pytest.raises(graph_document.GraphError, match=message):
        graph_document.validate_document(document)


def test_bundled_examples_use_the_current_schema() -> None:
    graph_document = _graph_document_module()
    examples_dir = Path(__file__).parent.parent / "examples"

    for example_path in examples_dir.glob("*.json"):
        document = json.loads(example_path.read_text())
        assert document["schema_version"] == graph_document.SCHEMA_VERSION


@pytest.mark.parametrize("version", [None, True, 1.0, "1", 2])
def test_document_rejects_a_non_integer_or_unknown_schema_version(
    version: object,
) -> None:
    graph_document = _graph_document_module()
    document = {"schema_version": version, "nodes": [], "connections": []}

    with pytest.raises(graph_document.GraphError, match="schema version"):
        graph_document.validate_document(document)


def test_document_rejects_duplicate_node_ids() -> None:
    graph_document = _graph_document_module()
    document = {
        "schema_version": 1,
        "nodes": [
            {
                "id": "node-1",
                "x": 0.0,
                "y": 0.0,
                "kind": "value",
                "math_type": "FLOAT",
                "components": [1.0],
            },
            {"id": "node-1", "x": 100.0, "y": 0.0, "kind": "output"},
        ],
        "connections": [],
    }

    with pytest.raises(graph_document.GraphError, match="Duplicate node id 'node-1'"):
        graph_document.validate_document(document)


@pytest.mark.parametrize(
    ("node", "message"),
    [
        (
            {"id": [], "x": 0.0, "y": 0.0, "kind": "output"},
            "Node id must be a non-empty string",
        ),
        (
            {"id": "node-1", "y": 0.0, "kind": "output"},
            "Node 'node-1' is missing 'x'",
        ),
        (
            {"id": "node-1", "x": 10**1000, "y": 0.0, "kind": "output"},
            "Node 'node-1' x must be a finite number",
        ),
        (
            {
                "id": "node-1",
                "x": 0.0,
                "y": 0.0,
                "kind": "value",
                "math_type": "FLOAT",
                "components": [1.0, 2.0],
            },
            "Float needs 1 components",
        ),
        (
            {
                "id": "node-1",
                "x": 0.0,
                "y": 0.0,
                "kind": "operation",
                "operation": "PERSPECTIVE",
            },
            "Perspective must be a generator node",
        ),
        (
            {
                "id": "node-1",
                "x": 0.0,
                "y": 0.0,
                "kind": "obj_loader",
                "array_ids": ["node-1", "node-2", "node-3"],
                "vertices": [],
                "faces": [],
                "uvs": [],
                "normals": [],
            },
            "Obj Loader needs four array ids",
        ),
        (
            {
                "id": "node-1",
                "x": 0.0,
                "y": 0.0,
                "kind": "mesh_viewer",
                "shading_mode": "Toon",
            },
            "Unknown shading mode 'Toon'",
        ),
    ],
)
def test_document_rejects_invalid_node_details(
    node: dict[str, object], message: str
) -> None:
    graph_document = _graph_document_module()
    document = {"schema_version": 1, "nodes": [node], "connections": []}

    with pytest.raises(graph_document.GraphError, match=message):
        graph_document.validate_document(document)


def test_document_rejects_an_unknown_connection_source() -> None:
    graph_document = _graph_document_module()
    document = {
        "schema_version": 1,
        "nodes": [
            {"id": "node-1", "x": 0.0, "y": 0.0, "kind": "output"},
        ],
        "connections": [
            {"source": "missing", "target": "node-1", "input": 0},
        ],
    }

    with pytest.raises(graph_document.GraphError, match="Unknown connection source"):
        graph_document.validate_document(document)


def test_document_rejects_an_unknown_connection_target() -> None:
    graph_document = _graph_document_module()
    document = {
        "schema_version": 1,
        "nodes": [
            {
                "id": "node-1",
                "x": 0.0,
                "y": 0.0,
                "kind": "value",
                "math_type": "FLOAT",
                "components": [1.0],
            },
        ],
        "connections": [
            {"source": "node-1", "target": "missing", "input": 0},
        ],
    }

    with pytest.raises(graph_document.GraphError, match="Unknown connection target"):
        graph_document.validate_document(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [("source", []), ("target", None)],
)
def test_document_rejects_non_string_connection_ids(field: str, value: object) -> None:
    graph_document = _graph_document_module()
    connection = {"source": "node-1", "target": "node-2", "input": 0}
    connection[field] = value
    document = {
        "schema_version": 1,
        "nodes": [
            {
                "id": "node-1",
                "x": 0.0,
                "y": 0.0,
                "kind": "value",
                "math_type": "FLOAT",
                "components": [1.0],
            },
            {"id": "node-2", "x": 100.0, "y": 0.0, "kind": "output"},
        ],
        "connections": [connection],
    }

    with pytest.raises(
        graph_document.GraphError, match=f"Connection {field} must be a node id"
    ):
        graph_document.validate_document(document)


@pytest.mark.parametrize(
    ("target_node", "invalid_input"),
    [
        ({"id": "node-2", "x": 100.0, "y": 0.0, "kind": "output"}, 1),
        (
            {
                "id": "node-2",
                "x": 100.0,
                "y": 0.0,
                "kind": "operation",
                "operation": "ADD",
            },
            2,
        ),
        ({"id": "node-2", "x": 100.0, "y": 0.0, "kind": "mesh_viewer"}, 5),
        (
            {
                "id": "node-2",
                "x": 100.0,
                "y": 0.0,
                "kind": "value",
                "math_type": "FLOAT",
                "components": [2.0],
            },
            0,
        ),
    ],
)
def test_document_rejects_an_invalid_connection_input(
    target_node: dict[str, object],
    invalid_input: int,
) -> None:
    graph_document = _graph_document_module()
    document = {
        "schema_version": 1,
        "nodes": [
            {
                "id": "node-1",
                "x": 0.0,
                "y": 0.0,
                "kind": "value",
                "math_type": "FLOAT",
                "components": [1.0],
            },
            target_node,
        ],
        "connections": [
            {"source": "node-1", "target": "node-2", "input": invalid_input},
        ],
    }

    with pytest.raises(
        graph_document.GraphError, match=f"invalid input {invalid_input}"
    ):
        graph_document.validate_document(document)
