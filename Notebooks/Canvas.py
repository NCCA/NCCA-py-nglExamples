import asyncio
import base64
import io

import marimo as mo
import numpy as np
from PIL import Image

app = mo.App()


@app.cell
def _():
    width, height = 256, 256
    html = f"""
    <canvas id="myCanvas" width="{width}" height="{height}" style="border:1px solid black;"></canvas>
    <script type="module">
      const canvas = document.getElementById('myCanvas');
      const ctx = canvas.getContext('2d');
      async function updateFrame(b64) {{
        const img = new Image();
        img.src = 'data:image/png;base64,' + b64;
        await img.decode();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);
      }}
      window.updateFrame = updateFrame;
    </script>
    """
    container = mo.ui.html(html)

    async def loop():
        for frame in range(200):
            x = np.linspace(0, 2 * np.pi, width)
            y = np.linspace(0, 2 * np.pi, height)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(X + frame / 10) * np.cos(Y + frame / 15)
            img_data = ((Z - Z.min()) / (Z.max() - Z.min()) * 255).astype(np.uint8)
            img = Image.fromarray(img_data, "L")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            # call the JS function
            await mo.run_js(f"window.updateFrame('{b64}')")
            await asyncio.sleep(0.05)

    mo.run(loop)
    return container
