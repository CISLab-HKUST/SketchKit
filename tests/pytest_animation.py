import os
import numpy as np
import pytest
from PIL import Image
from sketchkit.animation import Animator


def test_animator_runs_on_example_images(tmp_path):
    img0_path = os.path.join("test_images", "0-0.png")
    img1_path = os.path.join("test_images", "0-1.png")
    if not os.path.exists(img0_path) or not os.path.exists(img1_path):
        pytest.skip("Required test image files are missing")

    img0_np = np.array(Image.open(img0_path).convert("RGB"))
    img1_np = np.array(Image.open(img1_path).convert("RGB"))

    animator = Animator()
    frames = animator.run(
        img0_np,
        img1_np,
        inner_frames=24,
        output_folder=str(tmp_path),
    )

    assert isinstance(frames, list)
    assert len(frames) > 0
    for frame in frames:
        assert isinstance(frame, np.ndarray)
