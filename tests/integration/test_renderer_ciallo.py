import pytest
from sketchkit.datasets import DifferSketching
from sketchkit.renderer import CialloRenderer, CialloRenderOptions


def test_ciallo_renderer_renders_differ_sketch(tmp_path, data_dir):
    try:
        dataset = DifferSketching(skip_integrity_check=True)
    except Exception as exc:
        pytest.skip(f"DifferSketching dataset unavailable: {exc}")

    assert len(dataset) > 0
    sketch = dataset[0]

    brush_texture_path = data_dir / "rendering" / "brush_texture.png"
    if not brush_texture_path.exists():
        pytest.skip("Ciallo renderer brush texture is missing")

    render_options = CialloRenderOptions()
    render_options.canvas_size = (128, 128)
    render_options.brush_texture_path = str(brush_texture_path)
    render_options.background_color = (2, 0, 0, 1.0)
    render_options.stroke_color = (1, 0, 0, 255)

    renderer = CialloRenderer()
    raster_image = renderer.render(sketch, render_options)

    assert raster_image is not None

    output_path = tmp_path / "ciallo_rendering.png"
    raster_image.save(str(output_path))
    assert output_path.exists()