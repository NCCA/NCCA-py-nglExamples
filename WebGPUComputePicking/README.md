# Compute-Shader Picking (WebGPU)

An alternative to the colour-ID picking used in
[`SelectionManipulatorWebGPU`](../SelectionManipulatorWebGPU). Objects still
render into an offscreen ID target on click, but the ID is a **real integer
in an `r32uint` texture** rather than a float colour, and the whole-image
readback is replaced by a **compute-shader reduction** that hands the CPU
exactly 4 bytes.

```bash
uv run main.py          # or ./main.py
```

## Controls

| Input | Action |
|---|---|
| Left click | Select the object under the cursor (replaces selection) |
| `Ctrl` + click | Toggle an object in / out of the selection (multi-select) |
| `Alt` + LMB drag | Tumble the camera |
| `Alt` + RMB drag | Pan the camera |
| Mouse wheel | Dolly in / out |
| `Space` | Reset the camera |
| `Escape` | Quit |

## How it works

```
click ──► ID render pass ──► compute reduce ──► 4-byte readback
          (r32uint target)   (9x9 block,        (packed dist|id)
                              atomicMin)
```

### 1. Integer ID pass (`ObjectShader.wgsl`, `ObjectPipeline.py`)

The object shader has a second fragment entry point, `fragment_pick`, that
writes the object's `u32` pick ID straight to an `r32uint` attachment
(0 is reserved for the background clear). Compared with colour-ID picking
this removes the float→byte encoding, the 24-bit / 16.7M-object ceiling and
the reserved-colour bookkeeping — an ID is just the next integer.

Uint formats can't be multisampled, so the ID pipeline is a single-sampled
sibling of the shaded MSAA pipeline (built from the same shader module and
bind group). That's no loss: "antialiasing" object IDs would be meaningless.
The ID texture has `TEXTURE_BINDING` usage and **no `COPY_SRC`** — it never
leaves the GPU.

### 2. Compute reduction (`PickCompute.wgsl`, `PickResolver`)

One dispatch of a single `9x9` workgroup (matching the pick-block slop the
colour demos used) runs over the pixels around the click. Each thread loads
one ID and, if it's non-zero, packs

```
(squared distance to click) << 20  |  object id
```

into a `u32` and `atomicMin`s it into a storage buffer — a textbook
parallel argmin. Because the ID sits in the low bits, the nearest hit
always wins and distance ties resolve deterministically to the lowest ID.
The buffer is seeded with `0xffffffff` ("no hit") before each dispatch.

### 3. Readback

The CPU copies the single `u32` result to a `MAP_READ` buffer and unpacks
it. The colour-ID demo maps the entire resolved frame (a ~2.9 MB copy at
1024x720 on a 2x display) to inspect 81 pixels; this demo maps 4 bytes, and
the block scan happened in parallel on the GPU.

### Where this scales

The win isn't just the smaller copy — it's that the technique stays on the
GPU. The same pattern extends to marquee/lasso selection (dispatch over the
rectangle, accumulate unique IDs into a bitset), hover highlighting every
frame without stalls, or writing depth alongside the ID for a full
world-space hit point, none of which are practical when every query means
mapping the framebuffer.
