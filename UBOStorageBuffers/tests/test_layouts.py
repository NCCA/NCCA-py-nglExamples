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
    def test_itemsize_is_padded_to_16_byte_multiple(self):
        # albedo (vec3, 16-byte aligned) + shininess (pushed to offset 16) +
        # tail padding -> whole block rounds up to 32 bytes.
        assert MATERIAL_BLOCK_STD140_DTYPE.itemsize == 32

    def test_shininess_is_pushed_past_vec3_padding(self):
        offsets = {
            name: MATERIAL_BLOCK_STD140_DTYPE.fields[name][1]
            for name in MATERIAL_BLOCK_STD140_DTYPE.names
        }
        assert offsets == {"albedo": 0, "shininess": 16}


class TestMaterialBlockNaiveDtype:
    def test_itemsize_is_tightly_packed(self):
        assert MATERIAL_BLOCK_NAIVE_DTYPE.itemsize == 16

    def test_shininess_immediately_follows_albedo(self):
        offsets = {
            name: MATERIAL_BLOCK_NAIVE_DTYPE.fields[name][1]
            for name in MATERIAL_BLOCK_NAIVE_DTYPE.names
        }
        assert offsets == {"albedo": 0, "shininess": 12}

    def test_naive_and_correct_layouts_differ_only_in_shininess_offset(self):
        correct = {
            name: MATERIAL_BLOCK_STD140_DTYPE.fields[name][1]
            for name in MATERIAL_BLOCK_STD140_DTYPE.names
        }
        naive = {
            name: MATERIAL_BLOCK_NAIVE_DTYPE.fields[name][1]
            for name in MATERIAL_BLOCK_NAIVE_DTYPE.names
        }
        assert correct["albedo"] == naive["albedo"]
        assert correct["shininess"] != naive["shininess"]
        assert correct["shininess"] - naive["shininess"] == 4


class TestNaiveBytesPaddedToStd140:
    def test_output_length_matches_correct_block_size(self):
        payload = naive_bytes_padded_to_std140((1.0, 0.5, 0.25), 32.0)
        assert len(payload) == MATERIAL_BLOCK_STD140_DTYPE.itemsize

    def test_shininess_lands_in_albedo_padding_not_the_real_offset(self):
        payload = naive_bytes_padded_to_std140((1.0, 0.5, 0.25), 32.0)
        # naive write put shininess at byte 12 (correct, non-zero value)...
        naive_slot = np.frombuffer(payload, dtype=np.float32, count=1, offset=12)[0]
        assert naive_slot == pytest.approx(32.0)
        # ...but the shader reads shininess from byte 16, which is untouched
        # padding (zero) -- this is the visible corruption.
        real_std140_slot = np.frombuffer(payload, dtype=np.float32, count=1, offset=16)[
            0
        ]
        assert real_std140_slot == pytest.approx(0.0)


class TestStd140Offsets:
    def test_reports_both_material_layouts_and_scene_block(self):
        table = std140_offsets()
        assert set(table) == {
            "SceneBlock",
            "MaterialBlock (std140-correct)",
            "MaterialBlock (naive/packed -- WRONG)",
        }

    def test_correct_and_naive_shininess_offsets_differ_exactly_as_documented(self):
        table = std140_offsets()
        correct = table["MaterialBlock (std140-correct)"]["shininess"]
        naive = table["MaterialBlock (naive/packed -- WRONG)"]["shininess"]
        assert correct == (16, 4)
        assert naive == (12, 4)

    def test_albedo_offset_is_identical_in_both_material_layouts(self):
        table = std140_offsets()
        assert (
            table["MaterialBlock (std140-correct)"]["albedo"]
            == table["MaterialBlock (naive/packed -- WRONG)"]["albedo"]
            == (0, 12)
        )
