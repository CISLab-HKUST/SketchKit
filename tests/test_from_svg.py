from sketchkit.renderer.cairo_renderer import CairoRenderer
from sketchkit.core import Sketch

sketch = Sketch.from_svg(
    "1.svg",
    default_thickness=2.0,
    default_pressure=None,
    cast_canvas_to_int=True,
)

print(
    "canvas:",
    (sketch.width, sketch.height),
    "paths:",
    sketch.path_num,
    "curves:",
    sketch.curve_num,
)

renderer = CairoRenderer()
img = renderer.render(sketch)
outpath = "DifferSketching.png"
img.save(outpath, "PNG")
