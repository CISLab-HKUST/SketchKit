import pytest
from sketchkit.core.sketch import Sketch
from sketchkit.datasets import LineDrawer
from sketchkit.renderer.cairo_renderer import CairoRenderer


def test_serialization_roundtrip(tmp_path):
    try:
        dataset = LineDrawer()
    except Exception as exc:
        pytest.skip(f"LineDrawer dataset unavailable: {exc}")

    assert len(dataset) > 0
    sketch = dataset[0]
    renderer = CairoRenderer()

    json_path = tmp_path / "LineDrawer_test.json"
    svg_path = tmp_path / "LineDrawer_test.svg"

    sketch.to_json(str(json_path))
    sketch.to_svg(filename=str(svg_path))

    loaded_json = Sketch.from_json(str(json_path))
    assert loaded_json.path_num == sketch.path_num

    loaded_svg = Sketch.from_svg(
        str(svg_path),
        default_thickness=2.0,
        default_pressure=None,
        cast_canvas_to_int=True,
    )
    assert loaded_svg is not None

    raster_image = renderer.render(loaded_json)
    assert raster_image is not None
