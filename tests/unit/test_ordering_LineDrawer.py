import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
from sketchkit.ordering import Orderer
from sketchkit.core.sketch import Sketch
from sketchkit.utils.file import save_seq_gif
from sketchkit.renderer.cairo_renderer import CairoRenderer

if __name__ == "__main__":
    orderer = Orderer.create(method="LineDrawer")

    # Load image
    img_path = "test_images/a1-2.png"
    img_np = np.array(Image.open(img_path).convert("L"))

    # Video path
    video_path = "test_images/a1-2.mp4"
    output_path = "outputs/ordering/LineDrawer"
    os.makedirs(output_path, exist_ok=True)

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

        renderer = CairoRenderer()
        sketch_frames = sketch.get_progressive_frames(length_per_frame=10)
        raster_images = [renderer.render(frame) for frame in tqdm(sketch_frames)]
        raster_images[-1].save(f"{output_path}/sketch.jpg")
        save_seq_gif(raster_images, f"{output_path}/animated_drawing.gif", fps=30)
