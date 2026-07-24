import numpy as np
import pytest
from PIL import Image
from sketchkit.animation import Animator


def test_animator_runs_on_example_images(tmp_path, data_dir):
    img0_path = data_dir / "animation" / "start.png"
    img1_path = data_dir / "animation" / "end.png"
    if not img0_path.exists() or not img1_path.exists():
        pytest.skip("Required animation test images are missing")

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
