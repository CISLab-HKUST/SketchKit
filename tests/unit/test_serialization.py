from sketchkit.core.sketch import Sketch
from sketchkit.datasets import LineDrawer
from sketchkit.renderer.cairo_renderer import CairoRenderer
import os

if __name__ == "__main__":
    output_dir = "outputs/serialization"
    os.makedirs(output_dir, exist_ok=True)

    dataset = LineDrawer()
    print(len(dataset))

    renderer = CairoRenderer()
    s = dataset[0]
    raster_image = renderer.render(s)
    raster_image.save(os.path.join(output_dir, "LineDrawer_test1.png"))

    s.to_json(os.path.join(output_dir, "LineDrawer_test.json"))
    s.to_svg(filename=os.path.join(output_dir, "LineDrawer_test.svg"))

    r = Sketch.from_json(
        os.path.join(output_dir, "LineDrawer_test.json"),
    )
    raster_image = renderer.render(r)
    raster_image.save(os.path.join(output_dir, "LineDrawer_test2.png"))

    svg = Sketch.from_svg(
        os.path.join(output_dir, "LineDrawer_test.svg"),
        default_thickness=2.0,
        default_pressure=None,
        cast_canvas_to_int=True,
    )
    raster_image = renderer.render(svg)
    raster_image.save(os.path.join(output_dir, "LineDrawer_test3.png"))