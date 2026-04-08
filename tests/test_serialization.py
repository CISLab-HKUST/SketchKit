from sketchkit.core.sketch import Sketch
from sketchkit.datasets import SmartTracer

if __name__ == "__main__":
    dataset = SmartTracer(skip_integrity_check=True)
    print(len(dataset))

    from PIL import Image
    from sketchkit.renderer.cairo_renderer import CairoRenderer

    renderer = CairoRenderer()
    s = dataset[0]
    raster_image = renderer.render(s)
    raster_image.save("SmartTracer_test1.png")

    s.to_json("SmartTracer_test.json")
    s.to_svg(filename="SmartTracer_test.svg")

    r = Sketch.from_json("SmartTracer_test.json")
    raster_image = renderer.render(r)
    raster_image.save("SmartTracer_test2.png")
