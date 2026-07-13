"""std140-style uniform-block layouts shared by the OpenGL (UBO) and WebGPU
(uniform + storage buffer) halves of the UBOStorageBuffers demo.

Both GL's ``layout(std140)`` uniform blocks and WGSL's default "uniform
address space" layout follow the same packing rules (OpenGL 4.5 spec
7.6.2.2 / WGSL "Memory Layout" section), so a single set of numpy
structured dtypes can describe the CPU-side byte layout for both backends:

    - a scalar (float / int / bool) has a 4-byte base alignment.
    - a vec2 has an 8-byte base alignment.
    - a vec3 *or* vec4 has a 16-byte base alignment -- vec3 is stored as if
      it were a vec4, the 4th component is simply unused padding.
    - mat4 is stored as 4 columns, each a 16-byte-aligned vec4 (64 bytes).
    - every member's offset is rounded up to its own base alignment; a
      struct/block's total size is rounded up to a multiple of 16.

The classic "gotcha" this demo exists to teach lives in ``MaterialBlock``:

    struct MaterialBlock { vec3 albedo; float shininess; };

Because ``vec3``'s base alignment is 16 (not 12), ``shininess`` is *not*
packed immediately after ``albedo`` at byte offset 12. It is pushed up to
byte offset 16, leaving 4 bytes of padding between ``albedo.z`` and
``shininess``. A CPU-side struct that assumes tight packing (offset 12)
will write ``shininess`` into what the GPU actually treats as padding --
and the GPU will read *its* offset-16 bytes (whatever is there -- zero,
in this demo, since the upload buffer is zero-initialised) as
``shininess``. That is the "naive" layout below; toggling between the two
dtypes while uploading to the *same* correctly-declared shader block is
exactly how both demos reproduce the bug on screen.
"""

import numpy as np

# GL uniform-block binding points / WGSL @group(0) bind indices used by both
# demos for the blocks that exist on both backends.
SCENE_BLOCK_BINDING = 0
MATERIAL_BLOCK_BINDING = 1

# SceneBlock { mat4 VP; vec4 lightPos; vec4 lightColour; } -- one block, fed
# once per frame, read by every shader program/pipeline that declares it.
SCENE_BLOCK_DTYPE = np.dtype(
    {
        "names": ["VP", "lightPos", "lightColour"],
        "formats": [(np.float32, (4, 4)), (np.float32, 4), (np.float32, 4)],
        "offsets": [0, 64, 80],
        "itemsize": 96,
    }
)

# MaterialBlock { vec3 albedo; float shininess; } -- the CORRECT std140/WGSL
# layout: shininess is pushed to offset 16 by vec3's 16-byte alignment, and
# the block as a whole is padded out to 32 bytes (next multiple of 16).
MATERIAL_BLOCK_STD140_DTYPE = np.dtype(
    {
        "names": ["albedo", "shininess"],
        "formats": [(np.float32, 3), np.float32],
        "offsets": [0, 16],
        "itemsize": 32,
    }
)

# The NAIVE (wrong) layout a programmer gets by assuming a plain, tightly
# packed C struct: shininess immediately follows the 3 floats of albedo at
# byte offset 12. Uploading this to a shader block declared as above lands
# shininess in albedo's padding bytes, and the shader reads zero (or stale
# data) from the true offset-16 location instead.
MATERIAL_BLOCK_NAIVE_DTYPE = np.dtype(
    {
        "names": ["albedo", "shininess"],
        "formats": [(np.float32, 3), np.float32],
        "offsets": [0, 12],
        "itemsize": 16,
    }
)


def naive_bytes_padded_to_std140(albedo, shininess) -> bytes:
    """Build the NAIVE (wrong) MaterialBlock payload, zero-padded up to the
    STD140-correct block size (32 bytes) so it can be uploaded with a single
    ``glBufferSubData`` / ``write_buffer`` call of the buffer's real size.

    The result deliberately writes ``shininess`` at byte offset 12 -- which
    the shader (compiled against the correct std140/WGSL layout) treats as
    ``albedo``'s padding -- and leaves the shader's real offset-16
    ``shininess`` bytes as zero, so the corruption is a *visible, repeatable*
    "shininess collapses to 0" rather than uninitialised GPU memory.
    """
    naive = np.zeros((), dtype=MATERIAL_BLOCK_NAIVE_DTYPE)
    naive["albedo"] = albedo
    naive["shininess"] = shininess
    payload = bytearray(MATERIAL_BLOCK_STD140_DTYPE.itemsize)
    payload[: MATERIAL_BLOCK_NAIVE_DTYPE.itemsize] = naive.tobytes()
    return bytes(payload)


def std140_offsets() -> dict:
    """Return a small (block name -> {field: (offset, size)}) table for the
    HUD / README -- the ground truth this demo renders on screen.
    """
    return {
        "SceneBlock": {
            "VP": (0, 64),
            "lightPos": (64, 16),
            "lightColour": (80, 16),
            "size": SCENE_BLOCK_DTYPE.itemsize,
        },
        "MaterialBlock (std140-correct)": {
            "albedo": (0, 12),
            "shininess": (16, 4),
            "size": MATERIAL_BLOCK_STD140_DTYPE.itemsize,
        },
        "MaterialBlock (naive/packed -- WRONG)": {
            "albedo": (0, 12),
            "shininess": (12, 4),
            "size": MATERIAL_BLOCK_NAIVE_DTYPE.itemsize,
        },
    }
