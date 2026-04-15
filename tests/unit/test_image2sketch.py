import os

from PIL import Image

from sketchkit.core import Sketch
from sketchkit.image2sketch import SketchGenerator
from sketchkit.renderer.cairo_renderer import CairoRenderer


if __name__ == "__main__":
    image_path = "test_images/camel.jpg"
    # Load an input image
    with Image.open(image_path) as pil_image:
        input_image = pil_image.convert("RGB")

    output_dir = "outputs/image2sketch"
    os.makedirs(output_dir, exist_ok=True)

    # 1) HED: raster edge map
    hed_generator = SketchGenerator(method="HED")
    hed_result = hed_generator.run(input_image, size=512)
    assert isinstance(hed_result, Image.Image)
    hed_path = os.path.join(output_dir, "angel_hed.png")
    hed_result.save(hed_path)

    # 2) PhotoSketch: raster contour sketch
    photosketch_generator = SketchGenerator(method="PhotoSketch")
    photosketch_result = photosketch_generator.run(input_image, size=512)
    assert isinstance(photosketch_result, Image.Image)
    photosketch_path = os.path.join(output_dir, "angel_photosketch.png")
    photosketch_result.save(photosketch_path)

    # 3) SwiftSketch: vector sketch
    swiftsketch_generator = SketchGenerator(method="SwiftSketch")
    swiftsketch_result = swiftsketch_generator.run(input_image, size=512)
    assert isinstance(swiftsketch_result, Sketch)

    renderer = CairoRenderer()
    rendered_image = renderer.render(swiftsketch_result)
    assert isinstance(rendered_image, Image.Image)
    swiftsketch_path = os.path.join(output_dir, "angel_swiftsketch.png")
    rendered_image.save(swiftsketch_path)
