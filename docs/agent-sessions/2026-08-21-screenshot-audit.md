# Screenshot audit

I checked the screenshots used by the root README and the individual demo READMEs. The existing preview images have all been made 400 by 300 pixels. I did not change textures, maps or the Easing Functions graph as these are not demo screenshots.

The following demos were missing a preview, so I ran them locally and captured a new image:

- `LookAtDemos/LookAtDemos.png`
- `MatrixStack/MatrixStack.png`
- `ShadedGrid/ShadedGrid.png`
- `ViewToWorldTransform/ViewToWorldTransform.png`

The other changed PNG files are the README preview screenshots normalised to the same size.

Commands run:

```bash
sips -g pixelWidth -g pixelHeight <preview files>
uv run python capture_preview.py <demo> <preview>
sips --resampleHeight --cropToHeightWidth --padToHeightWidth <preview files>
uv run --with ruff ruff check .
uv build
```
