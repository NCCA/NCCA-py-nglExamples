import enum
from pathlib import Path

from ncca.ngl import Mat4, Vec3


class GameControls(enum.IntFlag):
    UP = 1 << 0
    DOWN = 1 << 1
    LEFT = 1 << 2
    RIGHT = 1 << 3
    ROTATE = 1 << 4


MOTION_TABLE: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),  # 0
    (0.0, 1.0, 0.0),  # UP
    (0.0, -1.0, 0.0),  # DOWN
    (0.0, 0.0, 0.0),  # UP|DOWN (nonsense)
    (-1.0, 0.0, 0.0),  # LEFT
    (-0.707, 0.707, 0.0),  # UP|LEFT
    (-0.707, -0.707, 0.0),  # DOWN|LEFT
    (-1.0, 0.0, 0.0),  # UP|DOWN (nonsense) & LEFT
    (1.0, 0.0, 0.0),  # RIGHT
    (0.707, 0.707, 0.0),  # UP|RIGHT
    (0.707, -0.707, 0.0),  # DOWN|RIGHT
    (1.0, 0.0, 0.0),  # UP|DOWN (nonsense) & RIGHT
    (0.0, 0.0, 0.0),  # LEFT|RIGHT (nonsense)
    (0.0, 1.0, 0.0),  # UP & LEFT|RIGHT (nonsense)
    (0.0, -1.0, 0.0),  # DOWN & LEFT|RIGHT (nonsense)
    (0.0, 0.0, 0.0),  # UP|DOWN (nonsense) & LEFT|RIGHT (nonsense)
    # -- ROTATE held: same 16 entries again, rotation flag set to 1 --
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (0.0, -1.0, 1.0),
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 1.0),
    (-0.707, 0.707, 1.0),
    (-0.707, -0.707, 1.0),
    (-1.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (0.707, 0.707, 1.0),
    (0.707, -0.707, 1.0),
    (1.0, 0.0, 1.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (0.0, -1.0, 1.0),
    (0.0, 0.0, 1.0),
)
X_EXTENTS = 40.0
Y_EXTENTS = 30.0
ROTATION_UPDATE = 4.0


def move_ship(pos: Vec3, rotation: float, keys_pressed: int) -> tuple[Vec3, float]:
    dx, dy, drot = MOTION_TABLE[keys_pressed]
    new_pos = Vec3(pos.x + dx, pos.y + dy, pos.z)
    new_pos.x = max(-X_EXTENTS, min(X_EXTENTS, new_pos.x))
    new_pos.y = max(-Y_EXTENTS, min(Y_EXTENTS, new_pos.y))
    new_rotation = rotation + ROTATION_UPDATE * drot
    return new_pos, new_rotation


def ship_transform(pos: Vec3, rotation: float) -> Mat4:
    tx = Mat4().rotate_y(rotation)
    tx[3, 0] = pos.x
    tx[3, 1] = pos.y
    tx[3, 2] = pos.z
    return tx


class KeyRecorder:
    def __init__(self) -> None:
        self._frames: list[int] = []
        self._start_position: Vec3 = Vec3(0.0, 0.0, 0.0)

    def size(self) -> int:
        return len(self._frames)

    def __getitem__(self, index: int) -> int:
        return self._frames[index]

    def add_frame(self, control_vars: int) -> None:
        self._frames.append(control_vars)

    def set_start_position(self, pos: Vec3) -> None:
        self._start_position = Vec3(pos.x, pos.y, pos.z)

    def get_start_position(self) -> Vec3:
        return Vec3(
            self._start_position.x, self._start_position.y, self._start_position.z
        )

    def save(self, path: Path | str) -> None:
        lines = [str(len(self._frames))]
        lines.append(
            f"{self._start_position.x} {self._start_position.y} {self._start_position.z}"
        )
        lines.extend(str(frame) for frame in self._frames)
        Path(path).write_text("\n".join(lines) + "\n")

    def load(self, path: Path | str) -> None:
        lines = Path(path).read_text().split()
        count = int(lines[0])
        x, y, z = float(lines[1]), float(lines[2]), float(lines[3])
        self._start_position = Vec3(x, y, z)
        self._frames = [int(v) for v in lines[4 : 4 + count]]
