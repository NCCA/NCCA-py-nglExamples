"""std140-style uniform-block layouts shared by the OpenGL (UBO) and WebGPU
(uniform + storage buffer) halves of the UBOStorageBuffers demo.

Both GL's ``layout(std140)`` uniform blocks and WGSL's default "uniform
address space" layout follow the same packing rules (OpenGL 4.5 spec
7.6.2.2 / WGSL "Memory Layout" section), so a single set of numpy
structured dtypes can describe the CPU-side byte layout for both backends:

    - a scalar (float / int / bool) has a 4-byte base alignment.
    - a vec2 has an 8-byte base alignment.
    - a vec3 *or* vec4 has a 16-byte base alignment, but a vec3 only
      *consumes* 12 bytes -- a following member whose own base alignment
      is <= 4 (a lone float, say) packs straight into the leftover 4-byte
      slot. The padding only appears when the NEXT member itself needs
      16-byte alignment.
    - mat4 is stored as 4 columns, each a 16-byte-aligned vec4 (64 bytes).
    - every member's offset is rounded up to *that member's own* base
      alignment; a struct/block's total size is rounded up to a multiple
      of 16.

The classic "gotcha" this demo exists to teach lives in ``MaterialBlock``:

    struct MaterialBlock { vec3 albedo; vec3 specularColour; float shininess; };

``albedo`` occupies bytes 0..11. A naively tight C-struct mindset puts
``specularColour`` at 12 -- but its base alignment is 16, so the compiler
pushes it to 16, leaving bytes 12..15 as padding. ``shininess`` (alignment
4) then packs into the slot straight after it, at 28 -- note it is NOT
pushed to 32; that is the half of the vec3 rule people over-correct on.

A CPU-side struct that assumes tight packing writes ``specularColour``
starting in the GPU's padding bytes and ``shininess`` inside what the GPU
reads as ``specularColour`` -- so the shader sees a scrambled specular
colour and a shininess of 0 (this demo's upload buffer is
zero-initialised). That is the "naive" layout below; toggling between the
two dtypes while uploading to the *same* correctly-declared shader block
is exactly how both demos reproduce the bug on screen. The GL demo also
queries ``GL_UNIFORM_OFFSET`` at startup so the driver's own answer is on
the HUD next to these hand-computed numbers.
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

# MaterialBlock { vec3 albedo; vec3 specularColour; float shininess; } --
# the CORRECT std140/WGSL layout: specularColour is pushed from 12 to 16 by
# its own 16-byte vec3 alignment, shininess (alignment 4) packs into the
# tail slot at 28, and 28 + 4 = 32 is already a 16-byte multiple.
MATERIAL_BLOCK_STD140_DTYPE = np.dtype(
    {
        "names": ["albedo", "specularColour", "shininess"],
        "formats": [(np.float32, 3), (np.float32, 3), np.float32],
        "offsets": [0, 16, 28],
        "itemsize": 32,
    }
)

# The NAIVE (wrong) layout a programmer gets by assuming a plain, tightly
# packed C struct: specularColour immediately follows albedo's 3 floats at
# byte offset 12, shininess at 24. Uploading this to a shader block declared
# as above starts specularColour in albedo's padding, so the shader reads a
# scrambled colour from its true offset 16 and zero shininess from 28.
MATERIAL_BLOCK_NAIVE_DTYPE = np.dtype(
    {
        "names": ["albedo", "specularColour", "shininess"],
        "formats": [(np.float32, 3), (np.float32, 3), np.float32],
        "offsets": [0, 12, 24],
        "itemsize": 28,
    }
)


def naive_bytes_padded_to_std140(albedo, specular_colour, shininess) -> bytes:
    """Build the NAIVE (wrong) MaterialBlock payload, zero-padded up to the
    STD140-correct block size (32 bytes) so it can be uploaded with a single
    ``glBufferSubData`` / ``write_buffer`` call of the buffer's real size.

    The result deliberately writes ``specularColour`` at byte offset 12 and
    ``shininess`` at 24. The shader (compiled against the correct
    std140/WGSL layout) reads specularColour from 16 -- landing on
    ``(specular.g, specular.b, shininess)``, a scrambled colour -- and
    shininess from 28, which was never written and reads back zero. Both
    corruptions are *visible and repeatable* rather than uninitialised GPU
    memory: the highlight loses its tight falloff AND its colour goes wrong,
    while albedo (offset 0 in both layouts) stays perfect.
    """
    naive = np.zeros((), dtype=MATERIAL_BLOCK_NAIVE_DTYPE)
    naive["albedo"] = albedo
    naive["specularColour"] = specular_colour
    naive["shininess"] = shininess
    payload = bytearray(MATERIAL_BLOCK_STD140_DTYPE.itemsize)
    payload[: MATERIAL_BLOCK_NAIVE_DTYPE.itemsize] = naive.tobytes()
    return bytes(payload)


def std140_offsets() -> dict:
    """Return a small (block name -> {field: (offset, size)}) table for the
    HUD / README -- hand-computed values the GL demo cross-checks against
    the driver's GL_UNIFORM_OFFSET answers at startup.
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
            "specularColour": (16, 12),
            "shininess": (28, 4),
            "size": MATERIAL_BLOCK_STD140_DTYPE.itemsize,
        },
        "MaterialBlock (naive/packed -- WRONG)": {
            "albedo": (0, 12),
            "specularColour": (12, 12),
            "shininess": (24, 4),
            "size": MATERIAL_BLOCK_NAIVE_DTYPE.itemsize,
        },
    }
