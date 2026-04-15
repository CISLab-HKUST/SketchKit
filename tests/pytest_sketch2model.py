import os
import numpy as np
import pytest
from PIL import Image

from sketchkit.datasets.differsketching import DifferSketching
from sketchkit.renderer.cairo_renderer import CairoRenderer
from sketchkit.sketch2model import Modeler


def get_sketch_and_img():
    try:
        dataset = DifferSketching()
    except Exception as exc:
        pytest.skip(f"DifferSketching dataset unavailable: {exc}")

    assert len(dataset) > 0
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


def test_teddy_methods(tmp_path):
    sketch, img = get_sketch_and_img()

    sketch_out = tmp_path / "teddy_sketch.obj"
    img_out = tmp_path / "teddy_image.obj"
    modeler = Modeler(method="teddy")

    out1 = modeler.run(input=sketch, output_path=str(sketch_out))
    assert isinstance(out1, str)
    assert os.path.exists(out1)

    out2 = modeler.run(input=img, output_path=str(img_out))
    assert isinstance(out2, str)
    assert os.path.exists(out2)


def test_sens_methods(tmp_path):
    sketch, img = get_sketch_and_img()

    sketch_out = tmp_path / "sens_sketch.obj"
    img_out = tmp_path / "sens_image.obj"
    modeler = Modeler(method="sens")

    out1 = modeler.run(input=sketch, output_path=str(sketch_out))
    assert isinstance(out1, str)
    assert os.path.exists(out1)

    out2 = modeler.run(input=img, output_path=str(img_out))
    assert isinstance(out2, str)
    assert os.path.exists(out2)


@pytest.mark.api
@pytest.mark.skipif("DOUBAO_API_KEY" not in os.environ, reason="DOUBAO_API_KEY not set")
def test_doubao_methods(tmp_path):
    sketch, img = get_sketch_and_img()

    sketch_out = tmp_path / "doubao_sketch.glb"
    img_out = tmp_path / "doubao_image.glb"
    modeler = Modeler(method="doubao-seed3d")

    out1 = modeler.run(input=sketch, output_path=str(sketch_out))
    assert isinstance(out1, str)
    assert os.path.exists(out1)

    out2 = modeler.run(input=img, output_path=str(img_out))
    assert isinstance(out2, str)
    assert os.path.exists(out2)
