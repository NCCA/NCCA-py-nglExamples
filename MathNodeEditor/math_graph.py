"""Calculation graph for the PyNGL maths node editor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from ncca.ngl import (
    Mat2,
    Mat3,
    Mat4,
    Obj,
    Quaternion,
    Transform,
    Vec2,
    Vec3,
    Vec4,
    frustum,
    look_at,
    ortho,
    perspective,
)
from ncca.ngl.opengl import Face


class MathType(Enum):
    """Maths value types accepted by value nodes."""

    FLOAT = "Float"
    VEC2 = "Vec2"
    VEC3 = "Vec3"
    VEC4 = "Vec4"
    MAT2 = "Mat2"
    MAT3 = "Mat3"
    MAT4 = "Mat4"
    QUATERNION = "Quaternion"


class Operation(Enum):
    """Operations which can be added to a graph."""

    ADD = "Add"
    SUBTRACT = "Subtract"
    MULTIPLY = "Multiply"
    MATRIX_MULTIPLY = "Matrix Multiply"
    DOT = "Dot Product"
    CROSS = "Cross Product"
    NORMALISE = "Normalise"
    TRANSPOSE = "Transpose"
    INVERSE = "Inverse"
    MAT4_TO_MAT3 = "Mat4 to Mat3"
    LOOK_AT = "Look At"
    PERSPECTIVE = "Perspective"
    ORTHO = "Orthographic"
    FRUSTUM = "Frustum"
    MAT4_TRANSLATE = "Mat4 Translate"
    MAT4_SCALE = "Mat4 Scale"
    MAT4_ROTATE_X = "Mat4 Rotate X"
    MAT4_ROTATE_Y = "Mat4 Rotate Y"
    MAT4_ROTATE_Z = "Mat4 Rotate Z"
    TRANSFORM = "Transform"
    QUATERNION_FROM_AXIS_ANGLE = "Quaternion from Axis Angle"
    QUATERNION_PRODUCT = "Quaternion Product"
    QUATERNION_ROTATE_VECTOR = "Quaternion Rotate Vector"
    QUATERNION_TO_MAT4 = "Quaternion to Mat4"
    MAT4_TO_QUATERNION = "Mat4 to Quaternion"
    QUATERNION_SLERP = "Quaternion Slerp"
    QUATERNION_CONJUGATE = "Quaternion Conjugate"
    QUATERNION_INVERSE = "Quaternion Inverse"
    TRANSFORM_VERTICES = "Transform Vertices"
    TRANSFORM_NORMALS = "Transform Normals"


OPERATION_INPUT_NAMES: dict[Operation, tuple[str, ...]] = {
    Operation.ADD: ("A", "B"),
    Operation.SUBTRACT: ("A", "B"),
    Operation.MULTIPLY: ("A", "B"),
    Operation.MATRIX_MULTIPLY: ("A", "B"),
    Operation.DOT: ("A", "B"),
    Operation.CROSS: ("A", "B"),
    Operation.NORMALISE: ("Value",),
    Operation.TRANSPOSE: ("Matrix",),
    Operation.INVERSE: ("Matrix",),
    Operation.MAT4_TO_MAT3: ("Matrix",),
    Operation.LOOK_AT: ("Eye", "Target", "Up"),
    Operation.PERSPECTIVE: ("FOV", "Aspect", "Near", "Far"),
    Operation.ORTHO: ("Left", "Right", "Bottom", "Top", "Near", "Far"),
    Operation.FRUSTUM: ("Left", "Right", "Bottom", "Top", "Near", "Far"),
    Operation.MAT4_TRANSLATE: ("X", "Y", "Z"),
    Operation.MAT4_SCALE: ("X", "Y", "Z"),
    Operation.MAT4_ROTATE_X: ("Angle",),
    Operation.MAT4_ROTATE_Y: ("Angle",),
    Operation.MAT4_ROTATE_Z: ("Angle",),
    Operation.TRANSFORM: ("Position", "Rotation", "Scale"),
    Operation.QUATERNION_FROM_AXIS_ANGLE: ("Axis", "Angle"),
    Operation.QUATERNION_PRODUCT: ("A", "B"),
    Operation.QUATERNION_ROTATE_VECTOR: ("Quaternion", "Vector"),
    Operation.QUATERNION_TO_MAT4: ("Quaternion",),
    Operation.MAT4_TO_QUATERNION: ("Matrix",),
    Operation.QUATERNION_SLERP: ("Start", "End", "T"),
    Operation.QUATERNION_CONJUGATE: ("Quaternion",),
    Operation.QUATERNION_INVERSE: ("Quaternion",),
    Operation.TRANSFORM_VERTICES: ("Matrix", "Vertices"),
    Operation.TRANSFORM_NORMALS: ("Matrix", "Normals"),
}

MESH_VIEWER_INPUT_NAMES: tuple[str, ...] = (
    "Vertices",
    "Faces",
    "UVs",
    "Normals",
    "Colour",
)

OPERATION_ARITY: dict[Operation, int] = {
    operation: len(input_names)
    for operation, input_names in OPERATION_INPUT_NAMES.items()
}

GENERATOR_OPERATIONS: frozenset[Operation] = frozenset(
    {
        Operation.LOOK_AT,
        Operation.PERSPECTIVE,
        Operation.ORTHO,
        Operation.FRUSTUM,
        Operation.MAT4_TRANSLATE,
        Operation.MAT4_SCALE,
        Operation.MAT4_ROTATE_X,
        Operation.MAT4_ROTATE_Y,
        Operation.MAT4_ROTATE_Z,
        Operation.TRANSFORM,
        Operation.QUATERNION_FROM_AXIS_ANGLE,
    }
)
"""Operations whose parameters are typed in directly, with no wired inputs."""

OPERATION_PARAMETER_TYPES: dict[Operation, tuple[MathType, ...]] = {
    Operation.LOOK_AT: (MathType.VEC3, MathType.VEC3, MathType.VEC3),
    Operation.PERSPECTIVE: (
        MathType.FLOAT,
        MathType.FLOAT,
        MathType.FLOAT,
        MathType.FLOAT,
    ),
    Operation.ORTHO: (MathType.FLOAT,) * 6,
    Operation.FRUSTUM: (MathType.FLOAT,) * 6,
    Operation.MAT4_TRANSLATE: (MathType.FLOAT, MathType.FLOAT, MathType.FLOAT),
    Operation.MAT4_SCALE: (MathType.FLOAT, MathType.FLOAT, MathType.FLOAT),
    Operation.MAT4_ROTATE_X: (MathType.FLOAT,),
    Operation.MAT4_ROTATE_Y: (MathType.FLOAT,),
    Operation.MAT4_ROTATE_Z: (MathType.FLOAT,),
    Operation.TRANSFORM: (MathType.VEC3, MathType.VEC3, MathType.VEC3),
    Operation.QUATERNION_FROM_AXIS_ANGLE: (MathType.VEC3, MathType.FLOAT),
}
"""The MathType of each named parameter, in OPERATION_INPUT_NAMES order."""

GENERATOR_OUTPUT_TYPE: dict[Operation, MathType] = {
    Operation.LOOK_AT: MathType.MAT4,
    Operation.PERSPECTIVE: MathType.MAT4,
    Operation.ORTHO: MathType.MAT4,
    Operation.FRUSTUM: MathType.MAT4,
    Operation.MAT4_TRANSLATE: MathType.MAT4,
    Operation.MAT4_SCALE: MathType.MAT4,
    Operation.MAT4_ROTATE_X: MathType.MAT4,
    Operation.MAT4_ROTATE_Y: MathType.MAT4,
    Operation.MAT4_ROTATE_Z: MathType.MAT4,
    Operation.TRANSFORM: MathType.MAT4,
    Operation.QUATERNION_FROM_AXIS_ANGLE: MathType.QUATERNION,
}
"""The MathType each generator operation's output socket should be coloured as."""


Corner: TypeAlias = tuple[int, "int | None", "int | None"]
"""One triangle corner: (vertex_index, uv_index, normal_index)."""


@dataclass(slots=True, frozen=True)
class VertexArray:
    """A mesh's vertex positions, in Obj file order."""

    values: tuple[Vec3, ...]


@dataclass(slots=True, frozen=True)
class NormalArray:
    """A mesh's vertex normals, in Obj file order."""

    values: tuple[Vec3, ...]


@dataclass(slots=True, frozen=True)
class UVArray:
    """A mesh's texture coordinates, in Obj file order."""

    values: tuple[Vec2, ...]


@dataclass(slots=True, frozen=True)
class FaceArray:
    """A triangulated mesh's per-corner vertex/uv/normal indices."""

    triangles: tuple[tuple[Corner, Corner, Corner], ...]


MathValue: TypeAlias = (
    Vec2
    | Vec3
    | Vec4
    | Mat2
    | Mat3
    | Mat4
    | Quaternion
    | float
    | VertexArray
    | NormalArray
    | UVArray
    | FaceArray
)

VALUE_CLASSES: dict[MathType, Callable[..., MathValue]] = {
    MathType.FLOAT: float,
    MathType.VEC2: Vec2,
    MathType.VEC3: Vec3,
    MathType.VEC4: Vec4,
    MathType.MAT2: Mat2,
    MathType.MAT3: Mat3,
    MathType.MAT4: Mat4,
    MathType.QUATERNION: Quaternion,
}

TYPE_SHAPES: dict[MathType, tuple[int, int]] = {
    MathType.FLOAT: (1, 1),
    MathType.VEC2: (1, 2),
    MathType.VEC3: (1, 3),
    MathType.VEC4: (1, 4),
    MathType.MAT2: (2, 2),
    MathType.MAT3: (3, 3),
    MathType.MAT4: (4, 4),
    MathType.QUATERNION: (1, 4),
}


class GraphError(ValueError):
    """An error which can be shown directly on an output node."""


def _validate_components(math_type: MathType, components: tuple[float, ...]) -> None:
    """Check that a value has the correct number of components."""
    rows, columns = TYPE_SHAPES[math_type]
    expected_count = rows * columns
    if len(components) != expected_count:
        raise GraphError(f"{math_type.value} needs {expected_count} components")


def _format_number(value: float) -> str:
    """Format a component without unnecessary trailing zeroes."""
    return f"{float(value):.5g}"


def format_value(value: MathValue) -> str:
    """Format a PyNGL maths value for an output node."""
    if isinstance(value, float):
        return _format_number(value)

    value_name = type(value).__name__
    data = value.to_numpy()
    if data.ndim == 1:
        components = ", ".join(_format_number(component) for component in data)
        return f"{value_name}({components})"

    rows = [
        "[" + "  ".join(_format_number(component) for component in row) + "]"
        for row in data
    ]
    return f"{value_name}\n" + "\n".join(rows)


def _normalise(value: MathValue) -> MathValue:
    """Return a unit-length copy of a vector or quaternion."""
    return value.normalized()


def _transpose(value: MathValue) -> MathValue:
    """Return the transpose of a matrix."""
    return value.transposed()


def _inverse(value: MathValue) -> MathValue:
    """Return the inverse of a Mat3 or Mat4 input."""
    if not isinstance(value, (Mat3, Mat4)):
        raise TypeError("Inverse needs a Mat3 or Mat4 input")
    return value.inverse()


def _mat4_to_mat3(value: MathValue) -> MathValue:
    """Extract the rotation/scale part of a Mat4 input, dropping translation."""
    if not isinstance(value, Mat4):
        raise TypeError("Mat4 to Mat3 needs a Mat4 input")
    return Mat3.from_mat4(value)


def _transform_vertices(matrix: MathValue, vertices: MathValue) -> MathValue:
    """Apply a Mat4 point transform, including translation, to a VertexArray."""
    if not isinstance(matrix, Mat4) or not isinstance(vertices, VertexArray):
        raise TypeError("Transform Vertices needs Mat4 and Vertices inputs")
    transformed = tuple(
        Vec3(*(Vec4(vertex.x, vertex.y, vertex.z, 1.0) @ matrix).to_list()[:3])
        for vertex in vertices.values
    )
    return VertexArray(transformed)


def _transform_normals(matrix: MathValue, normals: MathValue) -> MathValue:
    """Apply a Mat3 linear transform, ignoring translation, to a NormalArray."""
    if not isinstance(matrix, Mat3) or not isinstance(normals, NormalArray):
        raise TypeError("Transform Normals needs Mat3 and Normals inputs")
    return NormalArray(tuple(normal @ matrix for normal in normals.values))


def _quaternion_to_mat4(value: MathValue) -> MathValue:
    """Convert a Quaternion input to its Mat4 rotation."""
    if not isinstance(value, Quaternion):
        raise TypeError("Quaternion to Mat4 needs a Quaternion input")
    return value.to_mat4()


def _mat4_to_quaternion(value: MathValue) -> MathValue:
    """Convert a Mat4 input to the equivalent Quaternion."""
    if not isinstance(value, Mat4):
        raise TypeError("Mat4 to Quaternion needs a Mat4 input")
    return Quaternion.from_mat4(value)


def _quaternion_conjugate(value: MathValue) -> MathValue:
    """Return the conjugate of a Quaternion input."""
    if not isinstance(value, Quaternion):
        raise TypeError("Quaternion Conjugate needs a Quaternion input")
    return value.conjugate()


def _quaternion_inverse(value: MathValue) -> MathValue:
    """Return the inverse of a Quaternion input."""
    if not isinstance(value, Quaternion):
        raise TypeError("Quaternion Inverse needs a Quaternion input")
    return value.inverse()


def _quaternion_slerp(start: MathValue, end: MathValue, blend: MathValue) -> MathValue:
    """Spherically interpolate between two Quaternion inputs."""
    if (
        not isinstance(start, Quaternion)
        or not isinstance(end, Quaternion)
        or not isinstance(blend, float)
    ):
        raise TypeError(
            "Quaternion Slerp needs Quaternion, Quaternion and Float inputs"
        )
    return start.slerp(end, blend)


def _look_at(eye: MathValue, target: MathValue, up: MathValue) -> MathValue:
    """Build a view Mat4 from three Vec3 inputs."""
    if not all(isinstance(value, Vec3) for value in (eye, target, up)):
        raise TypeError("Look At needs three Vec3 inputs")
    return look_at(eye, target, up)


def _perspective(*inputs: MathValue) -> MathValue:
    """Build a perspective projection Mat4 from four Float inputs."""
    if not all(isinstance(value, float) for value in inputs):
        raise TypeError("Perspective needs four Float inputs")
    return perspective(*inputs)


def _ortho(*inputs: MathValue) -> MathValue:
    """Build an orthographic projection Mat4 from six Float inputs."""
    if not all(isinstance(value, float) for value in inputs):
        raise TypeError("Orthographic needs six Float inputs")
    return ortho(*inputs)


def _frustum(*inputs: MathValue) -> MathValue:
    """Build a frustum projection Mat4 from six Float inputs."""
    if not all(isinstance(value, float) for value in inputs):
        raise TypeError("Frustum needs six Float inputs")
    return frustum(*inputs)


def _mat4_translate(*inputs: MathValue) -> MathValue:
    """Build a translation Mat4 from three Float inputs."""
    if not all(isinstance(value, float) for value in inputs):
        raise TypeError("Mat4 Translate needs three Float inputs")
    return Mat4.translate(*inputs)


def _mat4_scale(*inputs: MathValue) -> MathValue:
    """Build a scale Mat4 from three Float inputs."""
    if not all(isinstance(value, float) for value in inputs):
        raise TypeError("Mat4 Scale needs three Float inputs")
    return Mat4.scale(*inputs)


def _mat4_rotate_x(angle: MathValue) -> MathValue:
    """Build a Mat4 rotation about X from a Float angle."""
    return Mat4.rotate_x(angle)


def _mat4_rotate_y(angle: MathValue) -> MathValue:
    """Build a Mat4 rotation about Y from a Float angle."""
    return Mat4.rotate_y(angle)


def _mat4_rotate_z(angle: MathValue) -> MathValue:
    """Build a Mat4 rotation about Z from a Float angle."""
    return Mat4.rotate_z(angle)


def _transform(position: MathValue, rotation: MathValue, scale: MathValue) -> MathValue:
    """Build a Model Mat4 from PyNGL's Transform (Position/Rotation/Scale)."""
    if not all(isinstance(value, Vec3) for value in (position, rotation, scale)):
        raise TypeError("Transform needs three Vec3 inputs")
    transform = Transform()
    transform.set_position(position)
    transform.set_rotation(rotation)
    transform.set_scale(scale)
    return transform.matrix()


def _quaternion_from_axis_angle(axis: MathValue, angle: MathValue) -> MathValue:
    """Build a Quaternion from a Vec3 axis and a Float angle."""
    if not isinstance(axis, Vec3) or not isinstance(angle, float):
        raise TypeError("Quaternion from Axis Angle needs Vec3 and Float inputs")
    return Quaternion.from_axis_angle(axis, angle)


def _add(left: MathValue, right: MathValue) -> MathValue:
    """Add two matching PyNGL values."""
    return left + right


def _subtract(left: MathValue, right: MathValue) -> MathValue:
    """Subtract one PyNGL value from another."""
    return left - right


def _matrix_multiply(left: MathValue, right: MathValue) -> MathValue:
    """Apply the PyNGL ``@`` product to two inputs."""
    return left @ right


def _quaternion_product(left: MathValue, right: MathValue) -> MathValue:
    """Return the Hamilton product of two Quaternion inputs."""
    if not isinstance(left, Quaternion) or not isinstance(right, Quaternion):
        raise TypeError("Quaternion Product needs two Quaternion inputs")
    return left @ right


def _quaternion_rotate_vector(left: MathValue, right: MathValue) -> MathValue:
    """Rotate a Vec3 input by a Quaternion input."""
    if not isinstance(left, Quaternion) or not isinstance(right, Vec3):
        raise TypeError("Quaternion Rotate Vector needs Quaternion and Vec3 inputs")
    return left * right


def _dot(left: MathValue, right: MathValue) -> MathValue:
    """Return the scalar dot product of two vector inputs."""
    return float(left.dot(right))


def _cross(left: MathValue, right: MathValue) -> MathValue:
    """Return the cross product of two vector inputs."""
    return left.cross(right)


def _multiply(left: MathValue, right: MathValue) -> MathValue:
    """Multiply two matching inputs component-wise.

    PyNGL reserves ``*`` for scalar multiplication, so this rebuilds the
    result from zipped components rather than using the operator.
    """
    if type(left) is not type(right):
        raise ValueError("component multiply needs matching input types")
    if isinstance(left, float):
        return left * right
    components = tuple(a * b for a, b in zip(left.to_list(), right.to_list()))
    return type(left)(*components)


_OPERATION_HANDLERS: dict[Operation, Callable[..., MathValue]] = {
    Operation.ADD: _add,
    Operation.SUBTRACT: _subtract,
    Operation.MULTIPLY: _multiply,
    Operation.MATRIX_MULTIPLY: _matrix_multiply,
    Operation.DOT: _dot,
    Operation.CROSS: _cross,
    Operation.NORMALISE: _normalise,
    Operation.TRANSPOSE: _transpose,
    Operation.INVERSE: _inverse,
    Operation.MAT4_TO_MAT3: _mat4_to_mat3,
    Operation.LOOK_AT: _look_at,
    Operation.PERSPECTIVE: _perspective,
    Operation.ORTHO: _ortho,
    Operation.FRUSTUM: _frustum,
    Operation.MAT4_TRANSLATE: _mat4_translate,
    Operation.MAT4_SCALE: _mat4_scale,
    Operation.MAT4_ROTATE_X: _mat4_rotate_x,
    Operation.MAT4_ROTATE_Y: _mat4_rotate_y,
    Operation.MAT4_ROTATE_Z: _mat4_rotate_z,
    Operation.TRANSFORM: _transform,
    Operation.QUATERNION_FROM_AXIS_ANGLE: _quaternion_from_axis_angle,
    Operation.QUATERNION_PRODUCT: _quaternion_product,
    Operation.QUATERNION_ROTATE_VECTOR: _quaternion_rotate_vector,
    Operation.QUATERNION_TO_MAT4: _quaternion_to_mat4,
    Operation.MAT4_TO_QUATERNION: _mat4_to_quaternion,
    Operation.QUATERNION_SLERP: _quaternion_slerp,
    Operation.QUATERNION_CONJUGATE: _quaternion_conjugate,
    Operation.QUATERNION_INVERSE: _quaternion_inverse,
    Operation.TRANSFORM_VERTICES: _transform_vertices,
    Operation.TRANSFORM_NORMALS: _transform_normals,
}


def apply_operation(
    operation: Operation,
    *inputs: MathValue,
) -> MathValue:
    """Apply an operation and translate PyNGL errors for the graph UI."""
    try:
        return _OPERATION_HANDLERS[operation](*inputs)
    except GraphError:
        raise
    except (AttributeError, TypeError, ValueError, ZeroDivisionError) as error:
        raise GraphError(f"{operation.value} failed: {error}") from error


def arrays_from_obj(obj: Obj) -> tuple[VertexArray, FaceArray, UVArray, NormalArray]:
    """Split a loaded Obj into the four array values an Obj Loader node outputs.

    Only triangulated meshes are supported, matching the rest of this
    codebase's Obj-handling demos (``ColourObj``, ``Obj2Numpy``).
    """
    if not obj.is_triangular():
        raise GraphError("Obj Loader needs a triangulated mesh")

    vertices = VertexArray(tuple(Vec3(v.x, v.y, v.z) for v in obj.vertex))
    normals = NormalArray(tuple(Vec3(n.x, n.y, n.z) for n in obj.normals))
    uvs = UVArray(tuple(Vec2(uv.x, uv.y) for uv in obj.uv))
    triangles = tuple(
        tuple(
            (
                face.vertex[corner],
                face.uv[corner] if face.uv else None,
                face.normal[corner] if face.normal else None,
            )
            for corner in range(3)
        )
        for face in obj.faces
    )
    return vertices, FaceArray(triangles), uvs, normals


def obj_from_arrays(
    vertices: VertexArray,
    faces: FaceArray,
    uvs: UVArray | None,
    normals: NormalArray | None,
) -> Obj:
    """Merge Obj Loader-style arrays back into a plain, VAO-less Obj mesh."""
    mesh = Obj()
    mesh.vertex = [Vec3(v.x, v.y, v.z) for v in vertices.values]
    mesh.normals = [Vec3(n.x, n.y, n.z) for n in normals.values] if normals else []
    mesh.uv = [Vec3(uv.x, uv.y, 0.0) for uv in uvs.values] if uvs else []

    built_faces = []
    for triangle in faces.triangles:
        face = Face()
        face.vertex = [corner[0] for corner in triangle]
        if mesh.uv and all(corner[1] is not None for corner in triangle):
            face.uv = [corner[1] for corner in triangle]
        if mesh.normals and all(corner[2] is not None for corner in triangle):
            face.normal = [corner[2] for corner in triangle]
        built_faces.append(face)
    mesh.faces = built_faces
    return mesh


@dataclass(slots=True)
class ValueNode:
    """A typed source value in the graph."""

    math_type: MathType
    components: tuple[float, ...]


@dataclass(slots=True)
class GeneratorNode:
    """An operation node whose parameters are typed in, not wired.

    Suits operations like Look At, Perspective and Transform, whose PyNGL
    inputs are just labelled Float/Vec3 numbers rather than other computed
    values, so there is nothing meaningful to connect a wire to.
    """

    operation: Operation
    parameters: list[tuple[float, ...]]


@dataclass(slots=True)
class OperationNode:
    """A mathematical operation and its input connections."""

    operation: Operation
    inputs: dict[int, str] = field(default_factory=dict)


@dataclass(slots=True)
class OutputNode:
    """A graph result with one input connection."""

    inputs: dict[int, str] = field(default_factory=dict)


@dataclass(slots=True)
class LiteralNode:
    """A pre-computed value with no inputs and no spin-box editing UI.

    Used for the Obj Loader node's four outputs, since their values come
    from a parsed file rather than user-typed components.
    """

    value: MathValue


@dataclass(slots=True)
class MeshViewerNode:
    """A mesh-rendering sink with Vertices/Faces/UVs/Normals/Colour inputs."""

    inputs: dict[int, str] = field(default_factory=dict)


@dataclass(slots=True)
class MeshViewerInputs:
    """The evaluated inputs of a Mesh Viewer node, ready to merge and render."""

    vertices: VertexArray
    faces: FaceArray
    uvs: UVArray | None
    normals: NormalArray | None
    colour: Vec4 | None


GraphNode: TypeAlias = (
    ValueNode
    | OperationNode
    | OutputNode
    | LiteralNode
    | MeshViewerNode
    | GeneratorNode
)


class MathGraph:
    """Own and evaluate value, operation and output nodes."""

    def __init__(self) -> None:
        """Create an empty graph."""
        self._nodes: dict[str, GraphNode] = {}
        self._next_id = 1

    def _add_node(self, node: GraphNode) -> str:
        """Add a graph node and return its stable identifier."""
        node_id = f"node-{self._next_id}"
        self._next_id += 1
        self._nodes[node_id] = node
        return node_id

    def add_value(self, math_type: MathType, components: tuple[float, ...]) -> str:
        """Add a typed value node to the graph."""
        _validate_components(math_type, components)
        return self._add_node(ValueNode(math_type, tuple(components)))

    def add_operation(self, operation: Operation) -> str:
        """Add a wired operation node to the graph."""
        if operation in GENERATOR_OPERATIONS:
            raise ValueError(
                f"{operation.value} has no wired inputs; use add_generator instead"
            )
        return self._add_node(OperationNode(operation))

    def add_generator(
        self, operation: Operation, parameters: tuple[tuple[float, ...], ...]
    ) -> str:
        """Add a parameter node (Look At, Perspective, ...) to the graph."""
        if operation not in GENERATOR_OPERATIONS:
            raise ValueError(
                f"{operation.value} has no typed-in parameters; use add_operation instead"
            )
        parameter_types = OPERATION_PARAMETER_TYPES[operation]
        if len(parameters) != len(parameter_types):
            raise ValueError(
                f"{operation.value} needs {len(parameter_types)} parameters, "
                f"got {len(parameters)}"
            )
        for parameter_type, components in zip(parameter_types, parameters, strict=True):
            _validate_components(parameter_type, components)
        return self._add_node(GeneratorNode(operation, [tuple(p) for p in parameters]))

    def set_generator_parameter(
        self, node_id: str, parameter_index: int, components: tuple[float, ...]
    ) -> None:
        """Replace one parameter's components on a generator node."""
        node = self._nodes[node_id]
        if not isinstance(node, GeneratorNode):
            raise ValueError("Only generator nodes store editable parameters")
        parameter_type = OPERATION_PARAMETER_TYPES[node.operation][parameter_index]
        _validate_components(parameter_type, components)
        node.parameters[parameter_index] = tuple(components)

    def set_value(self, node_id: str, components: tuple[float, ...]) -> None:
        """Replace the components stored by a value node."""
        node = self._nodes[node_id]
        if not isinstance(node, ValueNode):
            raise ValueError("Only value nodes store editable components")
        _validate_components(node.math_type, components)
        node.components = tuple(components)

    def add_output(self) -> str:
        """Add an output node to the graph."""
        return self._add_node(OutputNode())

    def add_literal(self, value: MathValue) -> str:
        """Add a pre-computed, non-editable value node to the graph."""
        return self._add_node(LiteralNode(value))

    def set_literal(self, node_id: str, value: MathValue) -> None:
        """Replace the value stored by a literal node."""
        node = self._nodes[node_id]
        if not isinstance(node, LiteralNode):
            raise ValueError("Only literal nodes store replaceable values")
        node.value = value

    def add_mesh_viewer(self) -> str:
        """Add a mesh viewer sink node to the graph."""
        return self._add_node(MeshViewerNode())

    def connect(self, source_id: str, target_id: str, input_index: int) -> None:
        """Connect a source node to an input on another node."""
        target = self._nodes[target_id]
        if isinstance(target, (ValueNode, LiteralNode, GeneratorNode)):
            raise ValueError("This node type does not have inputs")
        target.inputs[input_index] = source_id

    def disconnect(self, target_id: str, input_index: int) -> None:
        """Remove one input wire without deleting either node."""
        target = self._nodes[target_id]
        if isinstance(target, (ValueNode, LiteralNode, GeneratorNode)):
            raise ValueError("This node type does not have inputs")
        target.inputs.pop(input_index, None)

    def remove_node(self, node_id: str) -> None:
        """Delete a node and clear any downstream inputs that referenced it."""
        del self._nodes[node_id]
        for node in self._nodes.values():
            if isinstance(node, (ValueNode, LiteralNode, GeneratorNode)):
                continue
            for input_index, source_id in list(node.inputs.items()):
                if source_id == node_id:
                    del node.inputs[input_index]

    def evaluate(self, node_id: str) -> MathValue:
        """Evaluate a graph node and all the inputs below it."""
        return self._evaluate(node_id, set())

    def evaluate_mesh_viewer(self, node_id: str) -> MeshViewerInputs:
        """Evaluate a Mesh Viewer node's inputs, ready to merge and render."""
        node = self._nodes[node_id]
        if not isinstance(node, MeshViewerNode):
            raise ValueError("Only mesh viewer nodes can be evaluated this way")
        if 0 not in node.inputs:
            raise GraphError("Mesh Viewer needs input Vertices")
        if 1 not in node.inputs:
            raise GraphError("Mesh Viewer needs input Faces")

        vertices = self._evaluate(node.inputs[0], set())
        faces = self._evaluate(node.inputs[1], set())
        uvs = self._evaluate(node.inputs[2], set()) if 2 in node.inputs else None
        normals = self._evaluate(node.inputs[3], set()) if 3 in node.inputs else None
        colour = self._evaluate(node.inputs[4], set()) if 4 in node.inputs else None

        if not isinstance(vertices, VertexArray):
            raise GraphError("Mesh Viewer Vertices needs a Vertices input")
        if not isinstance(faces, FaceArray):
            raise GraphError("Mesh Viewer Faces needs a Faces input")
        if uvs is not None and not isinstance(uvs, UVArray):
            raise GraphError("Mesh Viewer UVs needs a UVs input")
        if normals is not None and not isinstance(normals, NormalArray):
            raise GraphError("Mesh Viewer Normals needs a Normals input")
        if colour is not None and not isinstance(colour, Vec4):
            raise GraphError("Mesh Viewer Colour needs a Vec4 input")

        return MeshViewerInputs(vertices, faces, uvs, normals, colour)

    def _evaluate(self, node_id: str, active_nodes: set[str]) -> MathValue:
        """Evaluate a node whilst tracking the current recursion path."""
        if node_id in active_nodes:
            raise GraphError("The graph contains a cycle")
        active_nodes.add(node_id)

        node = self._nodes[node_id]
        try:
            if isinstance(node, ValueNode):
                return VALUE_CLASSES[node.math_type](*node.components)
            if isinstance(node, LiteralNode):
                return node.value
            if isinstance(node, GeneratorNode):
                parameter_types = OPERATION_PARAMETER_TYPES[node.operation]
                values = tuple(
                    VALUE_CLASSES[parameter_type](*components)
                    for parameter_type, components in zip(
                        parameter_types, node.parameters, strict=True
                    )
                )
                return apply_operation(node.operation, *values)
            if isinstance(node, OutputNode):
                if 0 not in node.inputs:
                    raise GraphError("Output needs input Value")
                return self._evaluate(node.inputs[0], active_nodes)
            if isinstance(node, MeshViewerNode):
                raise GraphError("Mesh Viewer has no single output value")

            for input_index in range(OPERATION_ARITY[node.operation]):
                if input_index not in node.inputs:
                    input_name = OPERATION_INPUT_NAMES[node.operation][input_index]
                    raise GraphError(f"{node.operation.value} needs input {input_name}")

            values = tuple(
                self._evaluate(node.inputs[input_index], active_nodes)
                for input_index in range(OPERATION_ARITY[node.operation])
            )
            return apply_operation(node.operation, *values)
        finally:
            active_nodes.remove(node_id)
