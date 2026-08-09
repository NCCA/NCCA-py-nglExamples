import importlib.util
from pathlib import Path

import numpy as np


def load_webgpu_module():
    module_path = Path(__file__).resolve().parents[1] / "WebGPUmain.py"
    spec = importlib.util.spec_from_file_location("scifiui_webgpu", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_webgpu_terrain_updates_line_and_skirt_heights():
    module = load_webgpu_module()
    terrain = module.TerrainData()

    terrain.update(12.5)

    assert terrain.line_verts.shape == (
        module.TERRAIN_ROWS,
        module.TERRAIN_COLS,
        3,
    )
    assert terrain.skirt_verts.shape == (
        module.TERRAIN_ROWS,
        module.TERRAIN_COLS * 2,
        3,
    )
    np.testing.assert_allclose(
        terrain.line_verts[:, :, 1], terrain.skirt_verts[:, 0::2, 1]
    )
    np.testing.assert_allclose(terrain.skirt_verts[:, 1::2, 1], module.TERRAIN_FLOOR_Y)
    assert float(np.max(terrain.line_verts[:, :, 1])) > 0.1


def test_webgpu_ui_batch_records_ranges_with_float32_vertices():
    module = load_webgpu_module()
    batch = module.UIBatchData()

    batch.rect(10, 20, 30, 40, (0.1, 0.2, 0.3, 1.0))
    batch.outline(1, 2, 3, 4, (1.0, 1.0, 1.0, 1.0))
    batch.line(5, 6, 7, 8, (0.5, 0.5, 0.5, 1.0))

    assert batch.vertices.dtype == np.float32
    assert batch.vertices.shape == (16, 3)
    assert [entry.topology for entry in batch.ranges] == [
        "triangle-list",
        "line-list",
        "line-list",
    ]


def test_webgpu_shaders_are_present_with_entry_points():
    shader_dir = Path(__file__).resolve().parents[1] / "shaders"

    for name in ("UIShader.wgsl", "CRTShader.wgsl"):
        source = (shader_dir / name).read_text()
        assert "@vertex" in source
        assert "@fragment" in source
        assert "vertex_main" in source
        assert "fragment_main" in source
