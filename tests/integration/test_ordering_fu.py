import pytest
from sketchkit.core import Sketch
from sketchkit.ordering import Orderer
from sketchkit.renderer.cairo_renderer import CairoRenderer
from sketchkit.utils.file import save_seq_gif


def test_ordering_fu_progressive_frames(tmp_path, data_dir):
    svg_path = data_dir / "ordering" / "strokes.svg"
    if not svg_path.exists():
        pytest.skip("Ordering Fu test SVG is missing")

    sketch = Sketch.from_svg(str(svg_path))
    orderer = Orderer(method="Fu", device="cpu")
    animated_sketch = orderer.run(sketch)

    frames = animated_sketch.get_progressive_frames(length_per_frame=10)
    assert isinstance(frames, list)
    assert len(frames) > 0

    renderer = CairoRenderer()
    raster_images = [renderer.render(frame) for frame in frames]
    out_gif = tmp_path / "ordering_fu.gif"
    save_seq_gif(raster_images, str(out_gif))
    assert out_gif.exists()
