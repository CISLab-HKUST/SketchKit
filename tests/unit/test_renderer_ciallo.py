import os
from sketchkit.datasets import DifferSketching
from sketchkit.renderer import CialloRenderer, CialloRenderOptions

if __name__ == "__main__":
    my_data = DifferSketching(skip_integrity_check=True)
    sketch = my_data[0]

    render_options = CialloRenderOptions()
    renderer = CialloRenderer()

    render_options.canvas_size = (1024, 1024)
    render_options.brush_texture_path = "test_files/stamp1.png"
    render_options.background_color = (2, 0, 0, 1.0)
    render_options.stroke_color = (1, 0, 0, 255)

    raster_image = renderer.render(sketch, render_options)

    output_path = "outputs/renderer/ciallo_renderer"
    os.makedirs(output_path, exist_ok=True)
    raster_image.save(f"{output_path}/ciallo_rendering.png")
