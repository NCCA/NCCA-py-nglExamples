"""Tests for saved MathNodeEditor document validation."""

import json
import sys
from importlib import import_module
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

EXPECTED_EXAMPLE_NODE_NAMES = {
    "homogeneous_coordinates_demo.json": [
        "Translation Matrix",
        "Homogeneous Point (w=1)",
        "Homogeneous Direction (w=0)",
        "Transform Point",
        "Transform Direction",
        "Translated Point Result",
        "Unchanged Direction Result",
    ],
    "lambert_diffuse_demo.json": [
        "Object Normal",
        "Light Direction",
        "Unit Normal",
        "Unit Light Direction",
        "N dot L",
        "Diffuse Intensity Result",
    ],
    "mat2_rotation_demo.json": [
        "Original 2D Point",
        "90 Degree Rotation",
        "Rotated Point",
        "Rotation Transpose",
        "Recovered Point",
        "Rotated Point Result",
        "Recovered Point Result",
    ],
    "mesh_pipeline_demo.json": [
        "Cube Geometry",
        "Y Rotation",
        "Rotated Vertices",
        "Rotation 3x3",
        "Inverse Rotation",
        "Normal Matrix",
        "Rotated Normals",
        "Surface Colour",
        "Rotating Cube",
    ],
    "mvp_demo.json": [
        "Model Transform",
        "View Matrix",
        "Projection Matrix",
        "View @ Model",
        "Projection @ View Model",
        "MVP Result",
    ],
    "mvp_mesh_demo.json": [
        "Cube Geometry",
        "Model Transform",
        "Model Vertices",
        "Model 3x3",
        "Inverse Model",
        "Normal Matrix",
        "Model Normals",
        "Surface Colour",
        "Modelled Cube",
    ],
    "normal_matrix_demo.json": [
        "Object Normal",
        "Unit Object Normal",
        "Non-uniform Scale",
        "Linear Model Matrix",
        "Naive Normal Transform",
        "Unit Naive Normal",
        "Inverse Linear Matrix",
        "Normal Matrix",
        "Correct Normal Transform",
        "Unit Correct Normal",
        "Naive Normal Result",
        "Correct Normal Result",
    ],
    "projection_comparison_demo.json": [
        "Perspective Projection",
        "Orthographic Projection",
        "Asymmetric Frustum",
        "Perspective Matrix Result",
        "Orthographic Matrix Result",
        "Frustum Matrix Result",
    ],
    "quaternion_rotation_demo.json": [
        "Y Axis 90 Degrees",
        "X Direction",
        "Rotate Vector",
        "Rotation Quaternion Result",
        "Rotated Vector Result",
    ],
    "quaternion_slerp_demo.json": [
        "Start Orientation",
        "End Orientation",
        "Blend Factor",
        "Halfway Orientation",
        "X Direction",
        "Rotate by Halfway Orientation",
        "Halfway Quaternion Result",
        "Halfway Direction Result",
    ],
    "transform_order_demo.json": [
        "Input Point",
        "Scale Matrix",
        "Translation Matrix",
        "Scale @ Translate",
        "Translate @ Scale",
        "Apply Scale @ Translate",
        "Apply Translate @ Scale",
        "Scale @ Translate Result",
        "Translate @ Scale Result",
    ],
    "triangle_normal_demo.json": [
        "Point A",
        "Point B",
        "Point C",
        "Edge AB",
        "Edge AC",
        "AB Cross AC",
        "Unit Face Normal",
        "Cross Product Result",
        "Unit Normal Result",
    ],
    "vec3_multiply_demo.json": [
        "First Vector",
        "Second Vector",
        "Component Product",
        "Product Result",
    ],
    "vector_arithmetic_demo.json": [
        "Object Position",
        "Frame Movement",
        "Moved Position",
        "Moved Position Result",
        "Target Position",
        "Target - Position",
        "Displacement Result",
        "Squared Distance",
        "Squared Distance Result",
        "Unit Direction",
        "Unit Direction Result",
    ],
}


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


def test_bundled_example_catalogue_is_complete() -> None:
    examples_dir = Path(__file__).parent.parent / "examples"

    assert {path.name for path in examples_dir.glob("*.json")} == set(
        EXPECTED_EXAMPLE_NODE_NAMES
    )


def test_bundled_examples_are_valid_documents() -> None:
    graph_document = _graph_document_module()
    examples_dir = Path(__file__).parent.parent / "examples"

    for example_path in examples_dir.glob("*.json"):
        graph_document.validate_document(json.loads(example_path.read_text()))


def test_bundled_examples_name_every_node() -> None:
    examples_dir = Path(__file__).parent.parent / "examples"

    for filename, names in EXPECTED_EXAMPLE_NODE_NAMES.items():
        document = json.loads((examples_dir / filename).read_text())
        assert [node.get("name") for node in document["nodes"]] == names


def test_examples_readme_links_every_graph() -> None:
    examples_dir = Path(__file__).parent.parent / "examples"
    readme_path = examples_dir / "README.md"

    assert readme_path.exists()
    readme = readme_path.read_text()
    for filename in EXPECTED_EXAMPLE_NODE_NAMES:
        assert f"]({filename})" in readme


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


@pytest.mark.parametrize("name", ["", "   ", 42])
def test_document_rejects_invalid_node_name(name: object) -> None:
    graph_document = _graph_document_module()
    document = {
        "schema_version": 1,
        "nodes": [
            {
                "id": "node-1",
                "name": name,
                "x": 0.0,
                "y": 0.0,
                "kind": "output",
            }
        ],
        "connections": [],
    }

    with pytest.raises(
        graph_document.GraphError,
        match="Node 'node-1' name must be non-empty text",
    ):
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


@pytest.mark.parametrize("rotation_order", [None, 42, "bad"])
def test_document_rejects_an_unknown_transform_rotation_order(
    rotation_order: object,
) -> None:
    graph_document = _graph_document_module()
    document = {
        "schema_version": 1,
        "nodes": [
            {
                "id": "node-1",
                "x": 0.0,
                "y": 0.0,
                "kind": "generator",
                "operation": "TRANSFORM",
                "parameters": [
                    [0.0, 0.0, 0.0],
                    [30.0, 45.0, 60.0],
                    [1.0, 1.0, 1.0],
                ],
                "rotation_order": rotation_order,
            }
        ],
        "connections": [],
    }

    with pytest.raises(graph_document.GraphError, match="rotation order"):
        graph_document.validate_document(document)


def test_document_rejects_rotation_order_on_another_generator() -> None:
    graph_document = _graph_document_module()
    document = {
        "schema_version": 1,
        "nodes": [
            {
                "id": "node-1",
                "x": 0.0,
                "y": 0.0,
                "kind": "generator",
                "operation": "PERSPECTIVE",
                "parameters": [[45.0], [1.778], [0.1], [100.0]],
                "rotation_order": "zyx",
            }
        ],
        "connections": [],
    }

    with pytest.raises(
        graph_document.GraphError,
        match="Perspective does not use a rotation order",
    ):
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
