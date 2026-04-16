from sketchkit.datasets import DifferSketching
from sketchkit.renderer import CairoRenderer, CairoRenderOptions
import os

if __name__ == "__main__":
    # Load a sketch dataset
    my_data = DifferSketching(skip_integrity_check=True)
    sketch = my_data[0]

    # Initialize the renderer
    # Set up rendering options(optional). If canvas_size is not specified, it will use the original sketch size.
    options = CairoRenderOptions(
        canvas_size=(1024, 1024),
        background_color=(255, 255, 255),
        stroke_color=(0, 0, 0),
        stroke_width=2,
    )
    renderer = CairoRenderer(options)

    # Render a sketch into a raster image
    raster_image = renderer.render(sketch)

    # Save the rendered result
    output_path = "outputs/cairo_renderer"
    os.makedirs(output_path, exist_ok=True)
    raster_image.save(f"{output_path}/cairo_rendering.png")
