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