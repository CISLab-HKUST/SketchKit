"image2sketch test"
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image


from sketchkit.core import Sketch
from sketchkit.image2sketch import hed, image2sketch, photosketch, swiftsketch
from sketchkit.renderer.cairo_renderer import CairoRenderer

def _save_png(image: Image.Image, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    image.save(destination, format="PNG")
    assert destination.exists()
    assert destination.stat().st_size > 0

if __name__ == "__main__":
    image_path = Path(__file__).parent / "ant_357.png"

    with Image.open(image_path) as pil_image:
        input_image = pil_image.convert("RGB")

    output_dir = Path(__file__).parent


    hed_result = image2sketch(input_image, method="hed", size=512)
    assert isinstance(hed_result, Image.Image)
    hed_path = output_dir / "angel_hed.png"
    _save_png(hed_result, hed_path)
    
    photosketch_result = image2sketch(input_image, method="photosketch", size=512)
    assert isinstance(photosketch_result, Image.Image)
    photosketch_path = output_dir / "angel_photosketch.png"
    _save_png(photosketch_result, photosketch_path)

    swiftsketch_result = image2sketch(input_image, method="swiftsketch", size=512)
    assert isinstance(swiftsketch_result, Sketch)

    renderer = CairoRenderer(canvas_size=512)
    rendered_array = renderer.render(swiftsketch_result, fit_canvas=True)
    rendered_image = Image.fromarray(rendered_array.astype("uint8")).convert("RGB")
    swiftsketch_path = output_dir / "angel_swiftsketch.png"
    _save_png(rendered_image, swiftsketch_path)
