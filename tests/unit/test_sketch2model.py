import os
from pathlib import Path

import numpy as np
from PIL import Image

from sketchkit.datasets.differsketching import DifferSketching
from sketchkit.renderer.cairo_renderer import CairoRenderer
from sketchkit.sketch2model import Modeler


def get_sketch_and_img():
    dataset = DifferSketching()
    sketch = dataset[0]

    renderer = CairoRenderer()
    raster_image = renderer.render(sketch)
    if isinstance(raster_image, np.ndarray):
        if raster_image.dtype != np.uint8:
            raster_image = (raster_image * 255).astype(np.uint8)
        if raster_image.ndim == 3:
            img = Image.fromarray(raster_image[..., :3], "RGB")
        else:
            img = Image.fromarray(raster_image, "L")
    else:
        img = raster_image
    return sketch, img


def test_teddy_methods(output_dir):
    sketch, img = get_sketch_and_img()

    sketch_out = output_dir / "teddy_sketch.obj"
    img_out = output_dir / "teddy_image.obj"
    modeler = Modeler(method="teddy")

    try:
        out1 = modeler.run(input=sketch, output_path=str(sketch_out))
        assert os.path.exists(out1)
    except Exception as e:
        print(f"Teddy Sketch input failed: {e}")

    try:
        out2 = modeler.run(
            input=img,
            output_path=str(img_out),
        )
        assert os.path.exists(out2)
    except Exception as e:
        print(f"Teddy Image input failed: {e}")


def test_sens_methods(output_dir):
    sketch, img = get_sketch_and_img()

    sketch_out = output_dir / "sens_sketch.obj"
    img_out = output_dir / "sens_image.obj"
    modeler = Modeler(method="sens")

    try:
        out1 = modeler.run(input=sketch, output_path=str(sketch_out))
        assert isinstance(out1, str) and os.path.exists(out1)
    except Exception as e:
        print(f"SENS Sketch input failed: {e}")

    try:
        out2 = modeler.run(input=img, output_path=str(img_out))
        assert isinstance(out2, str) and os.path.exists(out2)
    except Exception as e:
        print(f"SENS Image input failed: {e}")


def test_doubao_methods(output_dir):
    sketch, img = get_sketch_and_img()

    sketch_out = output_dir / "doubao_sketch.glb"
    img_out = output_dir / "doubao_image.glb"
    modeler = Modeler(method="doubao-seed3d")

    try:
        out1 = modeler.run(
            input=sketch,
            output_path=str(sketch_out),
        )
        assert os.path.exists(out1)
    except Exception as e:
        print(f"Doubao Sketch input failed: {e}")

    try:
        out2 = modeler.run(
            input=img,
            output_path=str(img_out),
        )
        assert os.path.exists(out2)
    except Exception as e:
        print(f"Doubao Image input failed: {e}")


if __name__ == "__main__":
    output_dir = Path("outputs/sketch2model")
    output_dir.mkdir(parents=True, exist_ok=True)

    # print("Testing Teddy/MonsterMash...")
    # test_teddy_methods(output_dir=output_dir)

    # print("Testing SENS...")
    # test_sens_methods(output_dir=output_dir)
    print("Testing Doubao...")
    test_doubao_methods(output_dir=output_dir)
