// Compute-shader pick resolve.
//
// The ID render pass has written an integer object ID per pixel into an
// r32uint texture. Instead of copying that whole texture back to the CPU
// (as the colour-ID demos do), one 9x9 workgroup of this kernel inspects the
// block of pixels around the mouse *on the GPU* and reduces it to a single
// u32 answer; the CPU then reads back exactly 4 bytes.
//
// The reduction is a textbook parallel argmin done with atomicMin: each
// thread packs (squared distance to the click << ID_BITS) | object id into
// one u32 and atomicMin keeps the winner. Packing works because the ID is
// in the low bits: of two candidates the smaller distance always wins, and
// ties on distance resolve to the smaller ID deterministically. The block
// gives a few pixels of slop so near-misses on thin geometry still pick.
//
// result must be initialised to NO_HIT (0xffffffff) before every dispatch.

const BLOCK : i32 = 9;          // must match PICK_BLOCK on the CPU side
const HALF : i32 = BLOCK / 2;
const ID_BITS : u32 = 20u;      // low bits hold the id -> up to ~1M objects
const ID_MASK : u32 = (1u << ID_BITS) - 1u;

struct PickParams {
    pos : vec2<i32>,            // mouse position in texture pixels
    _pad : vec2<i32>,
};

@group(0) @binding(0) var id_texture : texture_2d<u32>;
@group(0) @binding(1) var<uniform> params : PickParams;
@group(0) @binding(2) var<storage, read_write> result : atomic<u32>;

@compute @workgroup_size(BLOCK, BLOCK)
fn pick_main(@builtin(local_invocation_id) lid : vec3<u32>) {
    let offset = vec2<i32>(i32(lid.x) - HALF, i32(lid.y) - HALF);
    let dims = vec2<i32>(textureDimensions(id_texture));
    let p = clamp(params.pos + offset, vec2<i32>(0, 0), dims - vec2<i32>(1, 1));

    let id = textureLoad(id_texture, p, 0).r;
    if (id == 0u) {             // 0 is background
        return;
    }
    let d2 = u32(offset.x * offset.x + offset.y * offset.y);
    atomicMin(&result, (d2 << ID_BITS) | (id & ID_MASK));
}
