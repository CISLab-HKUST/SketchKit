import pytest
from PIL import Image

from sketchkit.core.sketch import Sketch
from sketchkit.sketch2image import ImageGenerator
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
def sample_data(data_dir):
    svg_path = data_dir / "generation" / "sketch.svg"
    if not svg_path.exists():
        pytest.skip("Required sketch-to-image test SVG is missing")

    sketch = Sketch.from_svg(str(svg_path))
    sketch_image = convert_sketch_to_image(sketch, size=(1024, 1024))
    return sketch, sketch_image


SKETCH2IMG_TEST_PARAMS = [
    (
        "controlnet",
        "SG161222/RealVisXL_V4.0",
        "a highly photorealistic blue plane flying in the sky, 8k resolution, award winning photography",
        {"num_inference_steps": 15, "guidance_scale": 7.0},
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize("method, sd_model, prompt, kwargs", SKETCH2IMG_TEST_PARAMS)
def test_image_generator_methods_on_sketch(sample_data, method, sd_model, prompt, kwargs, tmp_path):
    sketch, _ = sample_data

    if sd_model is not None:
        generator = ImageGenerator(method=method, sd_model=sd_model)
    else:
        generator = ImageGenerator(method=method)

    img = generator.run(
        sketch,
        prompt=prompt,
        size=1024,
        input_size=None,
        **kwargs,
    )

    assert isinstance(img, Image.Image)

    out_file = tmp_path / f"sketch2img_{method}_sketch.png"
    img.save(out_file)
    assert out_file.exists()