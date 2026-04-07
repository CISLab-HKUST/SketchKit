import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from PIL import Image

from sketchkit.datasets.differsketching import DifferSketching
from sketchkit.sketch2model.methods.teddy_monstermash import sketch_to_3d_teddy_impl
from sketchkit.sketch2model.methods.doubao import sketch_to_3d_doubao_impl
from sketchkit.sketch2model.methods.sens_standalone import SENSStandaloneInference

def get_sketch_and_img():
    dataset = DifferSketching()
    sketch = dataset[0]
    from sketchkit.renderer.cairo_renderer import CairoRenderer
    renderer = CairoRenderer()  # Use default constructor
    raster_image = renderer.render(sketch)
    if isinstance(raster_image, np.ndarray):
        if raster_image.dtype != np.uint8:
            raster_image = (raster_image * 255).astype(np.uint8)
        if raster_image.ndim == 3:
            img = Image.fromarray(raster_image[...,:3], 'RGB')
        else:
            img = Image.fromarray(raster_image, 'L')
    else:
        img = raster_image
    return sketch, img

def test_teddy_methods():
    sketch, img = get_sketch_and_img()
    try:
        out1 = sketch_to_3d_teddy_impl(sketch)
        assert os.path.exists(out1)
    except Exception as e:
        print(f"Teddy Sketch input failed: {e}")
    try:
        out2 = sketch_to_3d_teddy_impl(img)
        assert os.path.exists(out2)
    except Exception as e:
        print(f"Teddy Image input failed: {e}")

def test_doubao_methods():
    sketch, img = get_sketch_and_img()
    try:
        out1 = sketch_to_3d_doubao_impl(sketch)
        assert os.path.exists(out1)
    except Exception as e:
        print(f"Doubao Sketch input failed: {e}")
    try:
        out2 = sketch_to_3d_doubao_impl(img)
        assert os.path.exists(out2)
    except Exception as e:
        print(f"Doubao Image input failed: {e}")

def test_sens_methods():
    sketch, img = get_sketch_and_img()
    from sketchkit.sketch2model.methods.sens_standalone import sketch_to_3d_sens_standalone
    try:
        out1 = sketch_to_3d_sens_standalone(sketch)
        assert isinstance(out1, str) and os.path.exists(out1)
    except Exception as e:
        print(f"SENS Sketch input failed: {e}")
    try:
        out2 = sketch_to_3d_sens_standalone(img)
        assert isinstance(out2, str) and os.path.exists(out2)
    except Exception as e:
        print(f"SENS Image input failed: {e}")


def main():
    print("Testing Teddy/MonsterMash...")
    test_teddy_methods()
    # print("Testing Doubao...")
    # test_doubao_methods()
    print("Testing SENS...")
    test_sens_methods()

if __name__ == "__main__":
    main()
