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
// Gizmo handles carry reserved IDs at the top of the range (>= PRIORITY_BASE)
// and are treated as distance zero, so a handle anywhere in the block beats
// every object - the integer equivalent of the colour-ID demo scanning for
// gizmo colours before object colours.
//
// result must be initialised to NO_HIT (0xffffffff) before every dispatch.

const BLOCK : i32 = 9;          // must match PICK_BLOCK on the CPU side
const HALF : i32 = BLOCK / 2;
const ID_BITS : u32 = 20u;      // low bits hold the id -> up to ~1M objects
const ID_MASK : u32 = (1u << ID_BITS) - 1u;
const PRIORITY_BASE : u32 = 0xFFF00u;  // must match GIZMO_ID_BASE in Manipulator.py

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
    // objects pack with distance + 1 so their smallest possible key
    // (1 << ID_BITS) is still larger than any gizmo key (distance 0):
    // a handle anywhere in the block wins even against an object under
    // the exact click pixel
    var d2 = u32(offset.x * offset.x + offset.y * offset.y) + 1u;
    if (id >= PRIORITY_BASE) {  // gizmo handles always beat objects
        d2 = 0u;
    }
    atomicMin(&result, (d2 << ID_BITS) | (id & ID_MASK));
}
