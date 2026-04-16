import os
import traceback

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


def test_sketch_input(sketch, output_path, prompt):
    method = "controlnet"
    sd_model = "SG161222/RealVisXL_V4.0"

    kwargs = {
        "num_inference_steps": 15,
        "guidance_scale": 7.0,
    }

    print(f"Initializing ImageGenerator: method={method}, model={sd_model}")
    generator = ImageGenerator(method=method, sd_model=sd_model)

    try:
        print("Running sketch2image with sketch input...")
        img = generator.run(
            sketch,
            prompt=prompt,
            size=1024,
            input_size=None,  # or input_size=(768, 768)
            **kwargs,
        )

        output_file = f"{output_path}/sketch_input.png"
        img.save(output_file)
        print(f"Done! Image saved to: {output_file}")

    except Exception as e:
        print(f"Run failed: {e}")
        traceback.print_exc()


def test_image_input(image, output_path, prompt):
    method = "controlnet"
    sd_model = "SG161222/RealVisXL_V4.0"

    kwargs = {
        "num_inference_steps": 15,
        "guidance_scale": 7.0,
    }

    print(f"Initializing ImageGenerator: method={method}, model={sd_model}")
    generator = ImageGenerator(method=method, sd_model=sd_model)

    try:
        print("Running sketch2image with image input...")
        img = generator.run(
            image,
            prompt=prompt,
            size=1024,
            input_size=None,  # or input_size=(768, 768)
            **kwargs,
        )

        output_file = f"{output_path}/image_input.png"
        img.save(output_file)
        print(f"Done! Image saved to: {output_file}")

    except Exception as e:
        print(f"Run failed: {e}")
        traceback.print_exc()


def main():
    sketch = Sketch.from_svg("test_images/plane.svg")
    sketch_image = convert_sketch_to_image(sketch, size=(1024, 1024))

    output_path = "outputs/sketch2image"
    os.makedirs(output_path, exist_ok=True)

    prompt = (
        "a highly photorealistic blue plane flying in the sky, "
        "8k resolution, award winning photography"
    )

    test_sketch_input(sketch, output_path, prompt)
    test_image_input(sketch_image, output_path, prompt)


if __name__ == "__main__":
    main()