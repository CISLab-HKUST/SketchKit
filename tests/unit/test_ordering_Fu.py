from tqdm import tqdm
import pandas as pd
import os

from sketchkit.core import Sketch
from sketchkit.ordering import Orderer
from sketchkit.renderer.cairo_renderer import CairoRenderer
from sketchkit.utils.file import save_seq_gif

if __name__ == "__main__":
    pd.set_option("display.max_colwidth", None)
    renderer = CairoRenderer()

    sketch = Sketch.from_svg("test_files/angel.svg")
    outpath = "outputs/ordering/Fu/sketch.png"
    svg_outpath = "outputs/ordering/Fu/sketch.svg"
    output_gif_path = "outputs/ordering/Fu/sketch.gif"
    output_images_path = "outputs/ordering/Fu/frames/"
    os.makedirs(output_images_path, exist_ok=True)

    animated_ordering = Orderer(method="Fu", device="cpu")
    sketch = animated_ordering.run(sketch)

    sketch_frames = sketch.get_progressive_frames(length_per_frame=10)

    raster_images = [renderer.render(frame) for frame in tqdm(sketch_frames)]
    for i, raster_image in enumerate(raster_images):
        raster_image.save(f"{output_images_path}/frame_{i:04d}.png", "PNG")
    
    save_seq_gif(raster_images, output_gif_path)

    raster_image = renderer.render(sketch)
    if outpath is not None:
        sketch.to_svg(filename=svg_outpath)
        raster_image.save(outpath, "PNG")
