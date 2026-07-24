import math
import pytest
import numpy as np
from sketchkit.core.camera import Camera
from sketchkit.core.sketch3d import (
    Sketch3D,
    Path3D,
    Curve3D,
    Point3D,
)
from sketchkit.utils3d.transforms import wxyz_from_look_at


def make_cis_sketch3d(variable_width=False):
    sketch = Sketch3D()
    depth_C, depth_I, depth_S = -0.1, 0.0, 0.1

    def make_curve(ctrls, color, thickness, variable=False):
        if variable:
            widths = list(thickness)
            v0 = Point3D(*ctrls[0], thickness=widths[0], color=color, opacity=1.0)
            v3 = Point3D(*ctrls[3], thickness=widths[3], color=color, opacity=1.0)
            c1 = Point3D(*ctrls[1], thickness=widths[1])
            c2 = Point3D(*ctrls[2], thickness=widths[2])
        else:
            v0 = Point3D(*ctrls[0], thickness=thickness, color=color, opacity=1.0)
            v3 = Point3D(*ctrls[3], thickness=thickness, color=color, opacity=1.0)
            c1 = Point3D(*ctrls[1], thickness=thickness)
            c2 = Point3D(*ctrls[2], thickness=thickness)
        return Curve3D(v0, v3, c1, c2)

    C_ctrls = [
        (-0.4, 0.6, depth_C),
        (-1.3, 0.6, depth_C),
        (-1.3, -0.6, depth_C),
        (-0.4, -0.6, depth_C),
    ]
    I_ctrls = [
        (0.0, -0.6, depth_I),
        (0.0, -0.2, depth_I),
        (0.0, 0.2, depth_I),
        (0.0, 0.6, depth_I),
    ]
    S_ctrls = [
        (0.9, 0.6, depth_S),
        (0.3, 0.2, depth_S),
        (1.3, -0.2, depth_S),
        (0.9, -0.6, depth_S),
    ]

    if variable_width:
        pathC = Path3D([
            make_curve(C_ctrls, (1, 0, 0), [8.0, 2.0, 2.0, 8.0], variable=True)
        ])
        pathI = Path3D([
            make_curve(I_ctrls, (0, 0.6, 0), [1.0, 5.0, 5.0, 1.0], variable=True)
        ])
        pathS = Path3D([
            make_curve(S_ctrls, (0, 0, 1), [15.0, 10.0, 5.0, 2.0], variable=True)
        ])
    else:
        pathC = Path3D([make_curve(C_ctrls, (1, 0, 0), 5.0)])
        pathI = Path3D([make_curve(I_ctrls, (0, 0.6, 0), 5.0)])
        pathS = Path3D([make_curve(S_ctrls, (0, 0, 1), 5.0)])

    sketch.add(pathC)
    sketch.add(pathI)
    sketch.add(pathS)
    return sketch


def generate_orbit_views(num_frames, radius=1.0, fov_deg=60):
    azimuths = np.linspace(0, 2.0 * np.pi, num_frames).tolist()
    elevations = [0.25 * np.pi for _ in azimuths]
    cameras = []
    for azimuth, elevation in zip(azimuths, elevations):
        r_xz = radius * math.cos(elevation)
        y = radius * math.sin(elevation)
        x = r_xz * math.cos(azimuth)
        z = r_xz * math.sin(azimuth)
        xyz = np.array([x, y, z])

        camera = Camera()
        wxyz = wxyz_from_look_at(
            xyz,
            np.array([0, 0, 0]),
            up=np.array([0, 1, 0]),
        )
        camera.set_xyz(xyz)
        camera.set_wxyz(wxyz)
        camera.set_fov(fov=(fov_deg / 180) * np.pi)
        cameras.append(camera)
    return cameras


def test_make_cis_sketch3d_default():
    sketch = make_cis_sketch3d()
    assert sketch.path_num == 3
    assert sketch.curve_num >= 3


def test_generate_orbit_views():
    cameras = generate_orbit_views(num_frames=4, radius=1.0, fov_deg=60)
    assert len(cameras) == 4
    for camera in cameras:
        assert isinstance(camera, Camera)


def test_diffvg_renderer_init_and_json_roundtrip(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pydiffvg")

    from sketchkit.renderer3d import DiffVGRenderer
    from sketchkit.renderer3d.diffvg_renderer import DiffVGRenderOptions

    options = DiffVGRenderOptions(canvas_size=(64, 64), stroke_width=2.0)
    renderer = DiffVGRenderer(render_options=options)
    assert renderer is not None

    out_json = tmp_path / "cis3d.json"
    sketch = make_cis_sketch3d()
    sketch.to_json(str(out_json))
    assert out_json.exists()

    loaded = Sketch3D.from_json(str(out_json))
    assert loaded.path_num == sketch.path_num
    assert loaded.curve_num == sketch.curve_num


def test_3d_attributes_roundtrip(tmp_path):
    sketch = Sketch3D(
        attributes={"scene": "demo"},
        paths=[
            Path3D(
                attributes={"layer": "front"},
                curves=[
                    Curve3D(
                        p_start=Point3D(0, 0, 0, thickness=1.0, color=(1, 0, 0), opacity=1.0, attributes={"role": "start"}),
                        p_end=Point3D(1, 1, 1, thickness=2.0, color=(0, 1, 0), opacity=0.8, attributes={"role": "end"}),
                        p_ctrl1=Point3D(0.2, 0.3, 0.4, thickness=1.5, attributes={"handle": 1}),
                        p_ctrl2=Point3D(0.6, 0.7, 0.8, thickness=1.6, attributes={"handle": 2}),
                        attributes={"curve_id": "3d-c0"},
                    )
                ],
            )
        ],
    )

    json_path = tmp_path / "attributes_3d.json"
    sketch.to_json(str(json_path))
    loaded = Sketch3D.from_json(str(json_path))
    curve = loaded.paths[0].curves[0]

    assert loaded.attributes == {"scene": "demo"}
    assert loaded.paths[0].attributes == {"layer": "front"}
    assert curve.attributes == {"curve_id": "3d-c0"}
    assert curve.p_start.attributes == {"role": "start"}
    assert curve.p_end.attributes == {"role": "end"}
    assert curve.p_ctrl1.attributes == {"handle": 1}
    assert curve.p_ctrl2.attributes == {"handle": 2}
