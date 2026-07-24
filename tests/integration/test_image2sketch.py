import pytest
from PIL import Image
from sketchkit.core import Sketch
from sketchkit.image2sketch import SketchGenerator
from sketchkit.renderer.cairo_renderer import CairoRenderer
from sketchkit.image2sketch.anilines.generate import AniLinesModel


def test_image2sketch_generators(tmp_path, data_dir):
    image_path = data_dir / "image_to_sketch" / "photo.jpg"
    if not image_path.exists():
        pytest.skip("Required image-to-sketch test image is missing")

    with Image.open(image_path) as pil_image:
        input_image = pil_image.convert("RGB")

    output_dir = tmp_path / "image2sketch"
    output_dir.mkdir(parents=True, exist_ok=True)

    hed_generator = SketchGenerator(method="HED")
    hed_result = hed_generator.run(input_image, size=512)
    assert isinstance(hed_result, Image.Image)
    hed_path = output_dir / "camel_hed.png"
    hed_result.save(str(hed_path))
    assert hed_path.exists()

    photosketch_generator = SketchGenerator(method="PhotoSketch")
    photosketch_result = photosketch_generator.run(input_image, size=512)
    assert isinstance(photosketch_result, Image.Image)
    photosketch_path = output_dir / "camel_photosketch.png"
    photosketch_result.save(str(photosketch_path))
    assert photosketch_path.exists()

    swiftsketch_generator = SketchGenerator(method="SwiftSketch")
    swiftsketch_result = swiftsketch_generator.run(input_image, size=512)
    assert isinstance(swiftsketch_result, Sketch)

    renderer = CairoRenderer()
    rendered_image = renderer.render(swiftsketch_result)
    assert isinstance(rendered_image, Image.Image)
    swiftsketch_path = output_dir / "camel_swiftsketch.png"
    rendered_image.save(str(swiftsketch_path))
    assert swiftsketch_path.exists()

    aniLinesModel_generator = SketchGenerator(method="anilines")
    anilines_result = aniLinesModel_generator.run(input_image, size=512)
    assert isinstance(anilines_result, Image.Image)
    anilines_path = output_dir / "camel_anilines_basic.png"
    anilines_result.save(str(anilines_path))
    assert anilines_path.exists()

    anilines_detail_model = AniLinesModel(mode="detail")
    anilines_detail_result = anilines_detail_model.generate(input_image)
    assert isinstance(anilines_detail_result, Image.Image)
    anilines_detail_path = output_dir / "camel_anilines_detail.png"
    anilines_detail_result.save(str(anilines_detail_path))
    assert anilines_detail_path.exists()
