import os

from PIL import Image

from sketchkit.core import Sketch
from sketchkit.image2sketch import SketchGenerator
from sketchkit.renderer.cairo_renderer import CairoRenderer
from sketchkit.image2sketch.anilines.generate import AniLinesModel


if __name__ == "__main__":
    image_path = "test_files/camel.jpg"
    # Load an input image
    with Image.open(image_path) as pil_image:
        input_image = pil_image.convert("RGB")

    output_dir = "outputs/image2sketch"
    os.makedirs(output_dir, exist_ok=True)

    # 1) HED: raster edge map
    hed_generator = SketchGenerator(method="HED")
    hed_result = hed_generator.run(input_image, size=512)
    assert isinstance(hed_result, Image.Image)
    hed_path = os.path.join(output_dir, "camel_hed.png")
    hed_result.save(hed_path)

    # 2) PhotoSketch: raster contour sketch
    photosketch_generator = SketchGenerator(method="PhotoSketch")
    photosketch_result = photosketch_generator.run(input_image, size=512)
    assert isinstance(photosketch_result, Image.Image)
    photosketch_path = os.path.join(output_dir, "camel_photosketch.png")
    photosketch_result.save(photosketch_path)

    # 3) SwiftSketch: vector sketch
    swiftsketch_generator = SketchGenerator(method="SwiftSketch")
    swiftsketch_result = swiftsketch_generator.run(input_image, size=512)
    assert isinstance(swiftsketch_result, Sketch)

    renderer = CairoRenderer()
    rendered_image = renderer.render(swiftsketch_result)
    assert isinstance(rendered_image, Image.Image)
    swiftsketch_path = os.path.join(output_dir, "camel_swiftsketch.png")
    rendered_image.save(swiftsketch_path)

    # 4) AniLinesModel
    aniLinesModel_generator = SketchGenerator(method="anilines")
    anilines_result = aniLinesModel_generator.run(input_image)
    assert isinstance(anilines_result, Image.Image)
    anilines_path = os.path.join(output_dir, "camel_anilines_basic.png")
    anilines_result.save(str(anilines_path))

    anilines_detail_model = AniLinesModel(mode="detail")
    anilines_detail_result = anilines_detail_model.generate(input_image)
    assert isinstance(anilines_detail_result, Image.Image)
    anilines_detail_path = os.path.join(output_dir, "camel_anilines_detail.png")
    anilines_detail_result.save(str(anilines_detail_path))
