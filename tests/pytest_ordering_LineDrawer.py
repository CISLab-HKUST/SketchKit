import os
import numpy as np
import pytest
from PIL import Image
from sketchkit.ordering import Orderer
from sketchkit.core.sketch import Sketch
from sketchkit.renderer.cairo_renderer import CairoRenderer
from sketchkit.utils.file import save_seq_gif


def test_ordering_linedrawer_with_video(tmp_path):
    img_path = os.path.join("test_files", "a1-2.png")
    video_path = os.path.join("test_files", "a1-2.mp4")
    if not os.path.exists(img_path) or not os.path.exists(video_path):
        pytest.skip("LineDrawer ordering test image or video is missing")

    img_np = np.array(Image.open(img_path).convert("L"))
    orderer = Orderer.create(method="LineDrawer")
    sketch = orderer.run(img_np, video_path)

    assert isinstance(sketch, Sketch)
    assert hasattr(sketch, "path_num")
    assert hasattr(sketch, "curve_num")
    assert sketch.path_num > 0 or sketch.curve_num > 0

    frames = sketch.get_progressive_frames(length_per_frame=10)
    assert isinstance(frames, list)
    assert len(frames) > 0

    renderer = CairoRenderer()
    raster_images = [renderer.render(frame) for frame in frames]
    assert len(raster_images) == len(frames)

    out_gif = tmp_path / "linedrawer.gif"
    save_seq_gif(raster_images, str(out_gif), fps=30)
    assert out_gif.exists()