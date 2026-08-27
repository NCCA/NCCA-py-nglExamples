"""Shared image-maze layout and actor movement."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


class Direction(enum.Enum):
    NORTH = (0, -1, 0.0)
    SOUTH = (0, 1, 180.0)
    EAST = (1, 0, 270.0)
    WEST = (-1, 0, 90.0)


@dataclass(frozen=True)
class ActorState:
    x: int
    z: int
    rotation: float = 0.0


@dataclass(frozen=True)
class WallCell:
    x: float
    z: float
    colour: tuple[float, float, float, float]


class Maze:
    """A maze image where the white pixels are paths and everything else is a wall.

    Pixel (x, y) is drawn at world (width / 2 - x, height / 2 - y) in the x/z
    plane, and the actor's grid position is that same pixel, so one test covers
    both. Flip the sign of the z term and the actor walks a mirrored copy of the
    maze whilst standing inside the wall cubes.

    Attributes
    ----------
        pixels : np.ndarray
            the RGB image, indexed [y, x]
    """

    def __init__(self, pixels: np.ndarray) -> None:
        self.pixels = pixels

    @classmethod
    def from_file(cls, path: str | Path) -> Maze:
        with Image.open(path) as image:
            return cls.from_pixels(np.asarray(image.convert("RGB")))

    @classmethod
    def from_pixels(cls, pixels: np.ndarray) -> Maze:
        data = np.asarray(pixels)
        if data.ndim == 2:
            data = np.repeat(data[:, :, np.newaxis], 3, axis=2)
        if data.ndim != 3 or data.shape[2] < 3:
            raise ValueError("maze pixels must be a greyscale, RGB or RGBA image")
        return cls(np.ascontiguousarray(data[:, :, :3], dtype=np.uint8))

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    def is_open(self, image_x: int, image_y: int) -> bool:
        if not 0 <= image_x < self.width or not 0 <= image_y < self.height:
            return False
        return bool(np.all(self.pixels[image_y, image_x] == 255))

    def wall_cells(self) -> tuple[WallCell, ...]:
        half_width = self.width / 2.0
        half_height = self.height / 2.0
        cells = []
        for image_y, image_x in np.argwhere(np.any(self.pixels != 255, axis=2)):
            rgb = self.pixels[image_y, image_x].astype(np.float32) / 255.0
            cells.append(
                WallCell(
                    x=half_width - float(image_x),
                    z=half_height - float(image_y),
                    colour=(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0),
                )
            )
        return tuple(cells)


def move_actor(maze: Maze, actor: ActorState, direction: Direction) -> ActorState:
    dx, dz, rotation = direction.value
    next_x = actor.x + dx
    next_z = actor.z + dz
    if not maze.is_open(next_x, next_z):
        return actor
    return ActorState(next_x, next_z, rotation)


def actor_world_position(maze: Maze, actor: ActorState) -> tuple[float, float, float]:
    return (
        maze.width / 2.0 - actor.x,
        0.5,
        maze.height / 2.0 - actor.z,
    )


def actor_forward(actor: ActorState) -> tuple[float, float, float]:
    return {
        0.0: (0.0, 0.0, 1.0),
        180.0: (0.0, 0.0, -1.0),
        270.0: (-1.0, 0.0, 0.0),
        90.0: (1.0, 0.0, 0.0),
    }[actor.rotation]
