import math
import shutil
import numpy as np
import pytest
from sketchkit.renderer.npr_renderer import NPRRenderer, NPRRenderOptions
from sketchkit.utils.file import save_seq_gif
from sketchkit.core.camera import Camera


@pytest.mark.parametrize("method", ["SuggestiveContour", "RidgesAndValleys", "AppearantRidges"])
def test_npr_renderer_renders_example_obj(tmp_path, data_dir, method):
    obj_path = data_dir / "rendering" / "head.obj"
    if not obj_path.exists():
        pytest.skip("NPR renderer test object file is missing")

    # Normalization writes beside the input; exercise spaces without editing fixtures.
    scene_dir = tmp_path / "scene with spaces"
    shutil.copytree(data_dir / "rendering", scene_dir)
    renderer = NPRRenderer(obj_path=str(scene_dir / "head.obj"))
    cameras = []
    r = 2.5
    for angle in range(0, 360, 120):
        rad = math.radians(angle)
        camera = Camera()
        camera.set_look_at([0, 0, 0])
        camera.set_xyz([r * math.cos(rad), r * math.sin(rad), 0])
        cameras.append(camera)

    options = NPRRenderOptions(canvas_size=(128, 128), method=method, threshold=0.005)
    images = renderer._render(cameras, options)

    assert isinstance(images, list)
    assert len(images) == len(cameras)
    for image in images:
        pixels = np.asarray(image)
        assert image.size == (128, 128)
        assert pixels.min() < 100 and pixels.max() > 200
    out_gif = tmp_path / f"{method}.gif"
    save_seq_gif(images, str(out_gif))
    assert out_gif.exists()
