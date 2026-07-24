from sketchkit.core.sketch import Curve, Path, Point, Sketch


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
