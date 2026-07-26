import pytest
from PIL import Image

from sketchkit.core import Sketch
from sketchkit.image2sketch import SketchGenerator
from sketchkit.renderer.cairo_renderer import CairoRenderer


@pytest.fixture
def input_image(data_dir):
    image_path = data_dir / "image_to_sketch" / "photo.jpg"
    if not image_path.exists():
        pytest.skip("Required image-to-sketch test image is missing")

    with Image.open(image_path) as image:
        return image.convert("RGB")


@pytest.fixture
def output_dir(tmp_path):
    path = tmp_path / "image2sketch"
    path.mkdir(parents=True)
    return path


def _save_image(result, output_path):
    assert isinstance(result, Image.Image)
    result.save(output_path)
    assert output_path.exists()


def test_hed(input_image, output_dir):
    generator = SketchGenerator(method="HED")
    result = generator.run(input_image, size=512)

    _save_image(result, output_dir / "camel_hed.png")


def test_photosketch(input_image, output_dir):
    generator = SketchGenerator(method="PhotoSketch")
    result = generator.run(input_image, size=512)

    _save_image(result, output_dir / "camel_photosketch.png")


def test_swiftsketch(input_image, output_dir):
    generator = SketchGenerator(method="SwiftSketch")
    result = generator.run(input_image, size=512)
    assert isinstance(result, Sketch)

    rendered_image = CairoRenderer().render(result)
    _save_image(rendered_image, output_dir / "camel_swiftsketch.png")


def test_anilines_basic(input_image, output_dir):
    generator = SketchGenerator(method="AniLines")
    result = generator.run(input_image, size=512)

    _save_image(result, output_dir / "camel_anilines_basic.png")


def test_anilines_detail(input_image, output_dir):
    generator = SketchGenerator(method="AniLines", mode="detail")
    result = generator.run(input_image)

    _save_image(result, output_dir / "camel_anilines_detail.png")
