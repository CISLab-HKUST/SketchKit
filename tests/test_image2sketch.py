"image2sketch test"
from __future__ import annotations

from pathlib import Path

from PIL import Image


from sketchkit.core import Sketch
from sketchkit.image2sketch import SketchGenerator
from sketchkit.renderer.cairo_renderer import CairoRenderer, CairoRenderOptions

def _save_png(image: Image.Image, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    image.save(destination, format="PNG")
    assert destination.exists()
    assert destination.stat().st_size > 0

if __name__ == "__main__":
    image_path = Path(__file__).parent / "angel.png"
    if not image_path.exists():
        fallback_path = Path(__file__).parent.parent / "camel.jpg"
        if fallback_path.exists():
            image_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Input image not found: {image_path}. "
                f"Also missing fallback image: {fallback_path}."
            )

    with Image.open(image_path) as pil_image:
        input_image = pil_image.convert("RGB")

    output_dir = Path(__file__).parent


    hed_generator = SketchGenerator(method="HED")
    hed_result = hed_generator.run(input_image, size=512)
    assert isinstance(hed_result, Image.Image)
    hed_path = output_dir / "angel_hed.png"
    _save_png(hed_result, hed_path)
    
    photosketch_generator = SketchGenerator(method="PhotoSketch")
    photosketch_result = photosketch_generator.run(input_image, size=512)
    assert isinstance(photosketch_result, Image.Image)
    photosketch_path = output_dir / "angel_photosketch.png"
    _save_png(photosketch_result, photosketch_path)

    swiftsketch_generator = SketchGenerator(
        method="SwiftSketch",
        use_refine=True,
        guidance_param=2.5,
    )
    swiftsketch_result = swiftsketch_generator.run(input_image, size=512)
    assert isinstance(swiftsketch_result, Sketch)

    render_options = CairoRenderOptions(canvas_size=(512, 512))
    renderer = CairoRenderer(render_options=render_options)
    rendered_image = renderer.render(swiftsketch_result, render_options=render_options)
    assert isinstance(rendered_image, Image.Image)
    swiftsketch_path = output_dir / "angel_swiftsketch.png"
    _save_png(rendered_image, swiftsketch_path)
