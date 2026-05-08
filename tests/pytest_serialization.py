import pytest
from sketchkit.core.sketch import Sketch, Path, Curve, Point
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


def test_attributes_roundtrip(tmp_path):
    sketch = Sketch(
        height=128,
        width=128,
        attributes={"author": "test", "version": 1},
        paths=[
            Path(
                label="demo",
                attributes={"group": "foreground"},
                curves=[
                    Curve(
                        p_start=Point(
                            0,
                            0,
                            pressure=0.5,
                            attributes={"role": "start"},
                        ),
                        p_end=Point(
                            10,
                            10,
                            pressure=0.9,
                            attributes={"role": "end"},
                        ),
                        p_crtl1=Point(3, 1, attributes={"handle": 1}),
                        p_crtl2=Point(7, 9, attributes={"handle": 2}),
                        attributes={"curve_id": "c0"},
                    )
                ],
            )
        ],
    )

    json_path = tmp_path / "attributes_roundtrip.json"
    sketch.to_json(str(json_path))

    loaded = Sketch.from_json(str(json_path))
    curve = loaded.paths[0].curves[0]

    assert loaded.attributes == {"author": "test", "version": 1}
    assert loaded.paths[0].attributes == {"group": "foreground"}
    assert curve.attributes == {"curve_id": "c0"}
    assert curve.p_start.attributes == {"role": "start"}
    assert curve.p_end.attributes == {"role": "end"}
    assert curve.p_crtl1.attributes == {"handle": 1}
    assert curve.p_crtl2.attributes == {"handle": 2}
