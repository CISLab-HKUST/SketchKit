import os
import pytest
from PIL import Image

from sketchkit.core.sketch import Sketch
from sketchkit.colorization import Colorizer
from sketchkit.renderer.cairo_renderer import CairoRenderer, CairoRenderOptions


def convert_sketch_to_image(sketch, size=(1024, 1024)):
    original_w, original_h = sketch.width, sketch.height
    sketch.resize(size[0], size[1])

    render_options = CairoRenderOptions(
        canvas_size=size,
        background_color=(1, 1, 1),
        stroke_width=3,
    )
    renderer = CairoRenderer(render_options=render_options)
    pil_image = renderer.render(sketch, render_options=render_options)

    sketch.resize(original_w, original_h)
    return pil_image


@pytest.fixture(scope="module")
def sample_data():
    svg_path = os.path.join("test_images", "plane.svg")
    if not os.path.exists(svg_path):
        pytest.skip("Required test file test_images/plane.svg is missing")

    sketch = Sketch.from_svg(svg_path)
    sketch_image = convert_sketch_to_image(sketch, size=(1024, 1024))
    return sketch, sketch_image


COLORIZATION_TEST_PARAMS = [
    (
        "controlnet_lineart",
        "cagliostrolab/animagine-xl-3.1",
        "one blue plane, masterpiece, best quality, highly detailed, vibrant colors, flat shading, anime illustration",
        {"num_inference_steps": 15, "guidance_scale": 7.5},
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize("method, sd_model, prompt, kwargs", COLORIZATION_TEST_PARAMS)
def test_colorizer_methods_on_sketch(sample_data, method, sd_model, prompt, kwargs, tmp_path):
    sketch, _ = sample_data

    colorizer = Colorizer(method=method, sd_model=sd_model)

    img = colorizer.run(
        sketch,
        prompt=prompt,
        size=1024,
        input_size=None,
        **kwargs,
    )

    assert isinstance(img, Image.Image)

    out_file = tmp_path / f"colorization_{method}_sketch.png"
    img.save(out_file)
    assert out_file.exists()


@pytest.mark.slow
def test_manga_ninja_colorization(tmp_path):
    ref_path = os.path.join("test_images", "manga_ninja", "ref_hz0.png")
    lineart_path = os.path.join("test_images", "manga_ninja", "lineart_hz1.png")
    if not os.path.exists(ref_path) or not os.path.exists(lineart_path):
        pytest.skip("Required MangaNinja test images are missing")

    reference_image = Image.open(ref_path).convert("RGB")
    lineart_image = Image.open(lineart_path).convert("RGB")

    colorizer = Colorizer(method="manga_ninja")

    img = colorizer.run(
        lineart_image,
        reference_image=reference_image,
        size=512,
        is_lineart=True,
        num_inference_steps=15,
        guidance_scale_ref=9.0,
    )

    assert isinstance(img, Image.Image)

    out_file = tmp_path / "colorization_manga_ninja.png"
    img.save(out_file)
    assert out_file.exists()


@pytest.mark.slow
def test_manga_ninja_with_point_control(tmp_path):
    ref_path = os.path.join("test_images", "manga_ninja", "ref_hz0.png")
    lineart_path = os.path.join("test_images", "manga_ninja", "lineart_hz1.png")
    point_ref_path = os.path.join("test_images", "manga_ninja", "point_ref_hz01.npy")
    point_main_path = os.path.join("test_images", "manga_ninja", "point_main_hz01.npy")
    if not os.path.exists(ref_path) or not os.path.exists(lineart_path):
        pytest.skip("Required MangaNinja test images are missing")

    reference_image = Image.open(ref_path).convert("RGB")
    lineart_image = Image.open(lineart_path).convert("RGB")

    import torch
    import numpy as np
    point_ref = None
    point_main = None
    if os.path.exists(point_ref_path):
        point_ref = torch.from_numpy(np.load(point_ref_path)).unsqueeze(0).unsqueeze(0)
    if os.path.exists(point_main_path):
        point_main = torch.from_numpy(np.load(point_main_path)).unsqueeze(0).unsqueeze(0)

    colorizer = Colorizer(method="manga_ninja")

    img = colorizer.run(
        lineart_image,
        reference_image=reference_image,
        size=512,
        is_lineart=True,
        num_inference_steps=15,
        guidance_scale_ref=9.0,
        guidance_scale_point=15.0,
        point_ref=point_ref,
        point_main=point_main,
    )

    assert isinstance(img, Image.Image)

    out_file = tmp_path / "colorization_manga_ninja_point.png"
    img.save(out_file)
    assert out_file.exists()


def test_manga_ninja_requires_reference_image():
    colorizer = Colorizer.__new__(Colorizer)
    colorizer.method = "manga_ninja"
    colorizer.device = "cuda"
    colorizer._model = None

    with pytest.raises(ValueError, match="reference_image is required"):
        colorizer.run(input_data=Image.new("RGB", (512, 512)))
