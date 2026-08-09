import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_DIR))

from texture_pack_parser import parse_texture_packs  # noqa: E402


def test_parse_texture_packs_reads_array_format(tmp_path):
    texture_file = tmp_path / "textures.json"
    texture_file.write_text(
        """{
  "texture_packs": [
    {
      "material": "wood",
      "textures": [
        {
          "location": 0,
          "name": "albedoMap",
          "path": "textures/wood/albedo.png"
        }
      ]
    }
  ]
}"""
    )

    packs = parse_texture_packs(texture_file)

    assert len(packs) == 1
    assert packs[0].material == "wood"
    assert len(packs[0].textures) == 1
    assert packs[0].textures[0].location == 0
    assert packs[0].textures[0].name == "albedoMap"
    assert packs[0].textures[0].path == "textures/wood/albedo.png"


def test_parse_texture_packs_rejects_legacy_duplicate_key_shape(tmp_path):
    texture_file = tmp_path / "textures.json"
    texture_file.write_text(
        """{
  "TexturePack": {
    "material": "wood",
    "Textures": []
  }
}"""
    )

    packs = parse_texture_packs(texture_file)

    assert packs == []


def test_parse_texture_packs_reads_demo_texture_file():
    texture_file = DEMO_DIR / "textures" / "textures.json"

    packs = parse_texture_packs(texture_file)

    assert [pack.material for pack in packs] == [
        "wood",
        "greasy",
        "copper",
        "rusty",
        "panel",
    ]
    assert all(len(pack.textures) == 5 for pack in packs)
