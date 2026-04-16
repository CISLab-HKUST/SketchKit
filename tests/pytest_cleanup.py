import os
import numpy as np
import pytest
from PIL import Image
from sketchkit.cleanup import Cleaner


def test_cleaner_master_sketch(tmp_path):
    image_path = os.path.join("test_images", "cleanup.png")
    if not os.path.exists(image_path):
        pytest.skip("Cleanup test image is missing")

    img_np = np.array(Image.open(image_path).convert("L"))
    cleaner = Cleaner(method="MasterSketch")
    output = cleaner.run(img_np, output_folder=str(tmp_path))

    assert output is not None
    assert isinstance(output, np.ndarray)
    assert output.ndim == 2
    assert output.dtype == np.uint8
    assert output.shape[0] >= img_np.shape[0] - 16
    assert output.shape[1] >= img_np.shape[1] - 16
    assert output.min() >= 0
    assert output.max() <= 255