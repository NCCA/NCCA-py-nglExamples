import base64
import time
from io import BytesIO

import marimo as mo
import numpy as np
from PIL import Image

app = mo.App()


# Cell 1 — create initial frame and helper
@app.cell
def _():
    import numpy as np

    def buffer_to_base64(buffer: np.ndarray) -> str:
        """Convert a numpy RGB buffer to a base64 PNG."""
        img = Image.fromarray(buffer.astype(np.uint8), "RGB")
        bio = BytesIO()
        img.save(bio, format="PNG")
        return base64.b64encode(bio.getvalue()).decode("utf-8")

    return buffer_to_base64


# end cell


# Cell 2 — make a buffer that updates every time this cell runs
@app.cell
def _(buffer_to_base64):
    t = time.time()
    w, h = 256, 256
    buffer = np.zeros((h, w, 3), dtype=np.uint8)
    # Animate colors using time
    buffer[..., 0] = (np.sin(t * 2.0) * 127 + 128).astype(np.uint8)
    buffer[..., 1] = (np.cos(t * 3.0) * 127 + 128).astype(np.uint8)
    buffer[..., 2] = ((np.sin(t * 1.5) + np.cos(t * 2.5)) * 63 + 128).astype(np.uint8)
    encoded = buffer_to_base64(buffer)
    return encoded


# end cell


# Cell 3 — render it to a live-updating canvas via JS
@app.cell
def _(encoded):
    html = f"""
    <canvas id="livecanvas" width="256" height="256"></canvas>
    <script>
      const imgData = "data:image/png;base64,{encoded}";
      const canvas = document.getElementById('livecanvas');
      const ctx = canvas.getContext('2d');
      const img = new Image();
      img.onload = () => {{
        ctx.drawImage(img, 0, 0);
      }};
      img.src = imgData;
    </script>
    """
    display = mo.Html(html)
    display
    return display


# end cell


# Optional — Cell 4: re-run automatically every second
@app.cell
def _():
    import marimo as mo

    async def auto_refresh(interval=1.0):
        while True:
            await mo.stop()  # yields to Marimo scheduler
            time.sleep(interval)

    # This won't run automatically, but you can rerun cells manually or with an external trigger.


# end cell


if __name__ == "__main__":
    app.run()
