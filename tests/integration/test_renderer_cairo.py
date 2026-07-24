import pytest
from sketchkit.datasets import DifferSketching
from sketchkit.renderer import CairoRenderer, CairoRenderOptions


def test_cairo_renderer_renders_differ_sketch(tmp_path):
    try:
        dataset = DifferSketching()
    except Exception as exc:
        pytest.skip(f"DifferSketching dataset unavailable: {exc}")

    assert len(dataset) > 0
    sketch = dataset[0]
    options = CairoRenderOptions(
        canvas_size=(128, 128),
        background_color=(255, 255, 255),
        stroke_width=2,
    )
    renderer = CairoRenderer(options)
    raster_image = renderer.render(sketch)

    assert raster_image is not None
    output_path = tmp_path / "cairo_rendering.png"
    raster_image.save(str(output_path))
    assert output_path.exists()
