import sys
import os

from tqdm import tqdm

from sketchkit.utils.file import save_seq_gif

sys.path.append(os.getcwd())

import numpy as np
from PIL import Image
from sketchkit.ordering import Orderer
from sketchkit.core.sketch import Sketch


if __name__ == "__main__":
    orderer = Orderer.create(method="LineDrawer")

    # Load image
    img_path = "a1-2.png"
    img_np = np.array(Image.open(img_path).convert("L"))

    # Video path
    video_path = "a1-2.mp4"

    if not os.path.exists(video_path):
        print(f"⚠ Warning: Video file {video_path} not found")
        print("Test will fail - please provide a video file")
    else:
        print(f"Running ordering with image path: {img_path}")
        print(f"Video path: {video_path}")

        # Orderer expects paths
        sketch = orderer.run(img_np, video_path)

        print(f"\n✓ Ordered sketch generated.")

        # Verify it is a Sketch object
        if isinstance(sketch, Sketch):
            print(f"  ✓ Output confirmed as Sketch object")
            print(f"  path_num: {sketch.path_num}")
            print(f"  curve_num: {sketch.curve_num}")
        else:
            print(f"  ⚠ Error: Output is {type(sketch)}, expected Sketch")

        print("Done.")

        from sketchkit.renderer.cairo_renderer import CairoRenderer

        renderer = CairoRenderer()

        sketch_frames = sketch.get_progressive_frames(sample_rate=10)
        raster_images = [
            renderer.render(frame, {"canvas_size": (1024, 1024)})
            for frame in tqdm(sketch_frames)
        ]
        raster_images[-1].save("linedrawer.jpg")
        save_seq_gif(raster_images, "animated_drawing.gif", fps=30)
