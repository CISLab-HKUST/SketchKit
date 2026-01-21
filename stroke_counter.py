from sketchkit.core import Sketch, Point, Curve, Vertex, Path
import numpy as np
from sketchkit.renderer.cairo_renderer import CairoRenderer, CairoRenderOptions
import os

svg_path = "media/motorcycle_757/motorcycle_757_80_strokes"


render_options = CairoRenderOptions(background_color=(1.0, 1.0, 1.0), stroke_width=5.0)
renderer = CairoRenderer(render_options)

if __name__ == '__main__':
    for path in os.listdir(os.path.join(svg_path, "svg_logs")):
        if path.endswith('.svg'):
            sketch = Sketch.from_svg(os.path.join(svg_path, "svg_logs", path))
            print(len(sketch.paths))
            raster_image = renderer.render(sketch)
            raster_image.save(os.path.join(svg_path, "svg_to_png", path.replace('.svg', '.png')), 'PNG')
    sketch = Sketch.from_svg(os.path.join(svg_path, "final_svg.svg"))
    print("Final stroke count:", len(sketch.paths))