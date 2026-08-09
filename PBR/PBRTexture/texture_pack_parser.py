"""Shared texture-pack JSON parser for the PBR texture demos."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TextureInfo:
    location: int
    name: str
    path: str


@dataclass(frozen=True)
class TexturePackInfo:
    material: str
    textures: list[TextureInfo]


def parse_texture_packs(filename: str | Path) -> list[TexturePackInfo]:
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error opening or parsing json file: {e}")
        return []

    texture_packs = data.get("texture_packs")
    if not isinstance(texture_packs, list):
        print("This does not seem to be a valid Texture Pack json file")
        return []

    packs = []
    for pack_data in texture_packs:
        pack = _parse_pack(pack_data)
        if pack is not None:
            packs.append(pack)
    return packs


def _parse_pack(pack_data: Any) -> TexturePackInfo | None:
    if not isinstance(pack_data, dict):
        print("Skipping texture pack entry as it is not an object")
        return None

    material = pack_data.get("material")
    if not isinstance(material, str) or not material:
        print("Skipping entry as it has no material")
        return None

    textures_data = pack_data.get("textures")
    if not isinstance(textures_data, list):
        print(f"Skipping material '{material}' as it has no textures")
        return None

    textures = []
    for texture_data in textures_data:
        texture = _parse_texture(texture_data)
        if texture is not None:
            textures.append(texture)
    return TexturePackInfo(material=material, textures=textures)


def _parse_texture(texture_data: Any) -> TextureInfo | None:
    if not isinstance(texture_data, dict):
        return None

    location = texture_data.get("location")
    name = texture_data.get("name")
    path = texture_data.get("path")
    if (
        not isinstance(location, int)
        or not isinstance(name, str)
        or not isinstance(path, str)
    ):
        return None
    return TextureInfo(location=location, name=name, path=path)
