import math
import pytest
from sketchkit.renderer.npr_renderer import NPRRenderer, NPRRenderOptions
from sketchkit.utils.file import save_seq_gif
from sketchkit.core.camera import Camera


def test_npr_renderer_renders_example_obj(tmp_path, data_dir):
    obj_path = data_dir / "rendering" / "head.obj"
    if not obj_path.exists():
        pytest.skip("NPR renderer test object file is missing")

    renderer = NPRRenderer(obj_path=str(obj_path))
    cameras = []
    r = 2.5
    for angle in range(0, 360, 120):
        rad = math.radians(angle)
        camera = Camera()
        camera.set_look_at([0, 0, 0])
        camera.set_xyz([r * math.cos(rad), r * math.sin(rad), 0])
        cameras.append(camera)

    options = NPRRenderOptions(method="SuggestiveContour", threshold=0.005)
    images = renderer._render(cameras, options)

    assert isinstance(images, list)
    assert len(images) > 0
    out_gif = tmp_path / "SuggestiveContour.gif"
    save_seq_gif(images, str(out_gif))
    assert out_gif.exists()
