"""Headless tests for the UBOStorageBuffers std140 layout maths (numpy-only,
no GL/Qt/wgpu imports -- see ../layouts.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from layouts import (  # noqa: E402
    MATERIAL_BLOCK_NAIVE_DTYPE,
    MATERIAL_BLOCK_STD140_DTYPE,
    SCENE_BLOCK_DTYPE,
    naive_bytes_padded_to_std140,
    std140_offsets,
)


class TestSceneBlockDtype:
    def test_itemsize_matches_hand_computed_std140_size(self):
        # mat4 (64) + vec4 (16) + vec4 (16) = 96 bytes, no padding needed
        # because every member here is already 16-byte aligned.
        assert SCENE_BLOCK_DTYPE.itemsize == 96

    def test_field_offsets(self):
        offsets = {
            name: SCENE_BLOCK_DTYPE.fields[name][1] for name in SCENE_BLOCK_DTYPE.names
        }
        assert offsets == {"VP": 0, "lightPos": 64, "lightColour": 80}


class TestMaterialBlockStd140Dtype:
    def test_itemsize_matches_hand_computed_std140_size(self):
        # albedo (0..11) + specularColour pushed to 16 (vec3 base alignment
        # 16) + shininess packing into the tail slot at 28 -> exactly 32,
        # already a 16-byte multiple.
        assert MATERIAL_BLOCK_STD140_DTYPE.itemsize == 32

    def test_second_vec3_is_pushed_to_16_but_trailing_float_is_not(self):
        # THE std140 vec3 rule, stated correctly: a vec3 consumes 12 bytes
        # but forces the NEXT 16-byte-aligned member (here specularColour)
        # up to the next 16-byte boundary. A scalar float has base alignment
        # 4, so shininess packs straight after specularColour at 28 -- it is
        # NOT pushed to 32.
        offsets = {
            name: MATERIAL_BLOCK_STD140_DTYPE.fields[name][1]
            for name in MATERIAL_BLOCK_STD140_DTYPE.names
        }
        assert offsets == {"albedo": 0, "specularColour": 16, "shininess": 28}


class TestMaterialBlockNaiveDtype:
    def test_itemsize_is_tightly_packed(self):
        # 3 + 3 + 1 floats, no padding anywhere.
        assert MATERIAL_BLOCK_NAIVE_DTYPE.itemsize == 28

    def test_members_immediately_follow_each_other(self):
        offsets = {
            name: MATERIAL_BLOCK_NAIVE_DTYPE.fields[name][1]
            for name in MATERIAL_BLOCK_NAIVE_DTYPE.names
        }
        assert offsets == {"albedo": 0, "specularColour": 12, "shininess": 24}

    def test_layouts_agree_on_albedo_and_disagree_after_it(self):
        correct = {
            name: MATERIAL_BLOCK_STD140_DTYPE.fields[name][1]
            for name in MATERIAL_BLOCK_STD140_DTYPE.names
        }
        naive = {
            name: MATERIAL_BLOCK_NAIVE_DTYPE.fields[name][1]
            for name in MATERIAL_BLOCK_NAIVE_DTYPE.names
        }
        assert correct["albedo"] == naive["albedo"] == 0
        # every member after the first vec3 is displaced by its padding
        assert correct["specularColour"] - naive["specularColour"] == 4
        assert correct["shininess"] - naive["shininess"] == 4


class TestNaiveBytesPaddedToStd140:
    def test_output_length_matches_correct_block_size(self):
        payload = naive_bytes_padded_to_std140((1.0, 0.5, 0.25), (1.0, 1.0, 1.0), 32.0)
        assert len(payload) == MATERIAL_BLOCK_STD140_DTYPE.itemsize

    def test_shader_visible_reads_of_the_naive_payload_are_corrupted(self):
        albedo = (1.0, 0.5, 0.25)
        specular_colour = (0.9, 0.6, 0.3)
        shininess = 32.0
        payload = naive_bytes_padded_to_std140(albedo, specular_colour, shininess)
        as_floats = np.frombuffer(payload, dtype=np.float32)

        # albedo lives at offset 0 in both layouts -- the shader reads it
        # correctly, which is what makes this bug so easy to miss.
        assert as_floats[0:3] == pytest.approx(albedo)

        # the shader reads specularColour from std140 offset 16 (floats
        # 4..6), but the naive write put (specularColour, shininess) there:
        # it sees (specular.g, specular.b, shininess) -- scrambled.
        assert as_floats[4:7] == pytest.approx(
            (specular_colour[1], specular_colour[2], shininess)
        )

        # and the shader's real shininess slot (offset 28, float 7) was
        # never written -- it reads back the zero padding.
        assert as_floats[7] == pytest.approx(0.0)


class TestStd140Offsets:
    def test_reports_both_material_layouts_and_scene_block(self):
        table = std140_offsets()
        assert set(table) == {
            "SceneBlock",
            "MaterialBlock (std140-correct)",
            "MaterialBlock (naive/packed -- WRONG)",
        }

    def test_correct_and_naive_offsets_differ_exactly_as_documented(self):
        table = std140_offsets()
        correct = table["MaterialBlock (std140-correct)"]
        naive = table["MaterialBlock (naive/packed -- WRONG)"]
        assert correct["specularColour"] == (16, 12)
        assert naive["specularColour"] == (12, 12)
        assert correct["shininess"] == (28, 4)
        assert naive["shininess"] == (24, 4)

    def test_albedo_offset_is_identical_in_both_material_layouts(self):
        table = std140_offsets()
        assert (
            table["MaterialBlock (std140-correct)"]["albedo"]
            == table["MaterialBlock (naive/packed -- WRONG)"]["albedo"]
            == (0, 12)
        )
