import os
import glob
from pathlib import Path
import random
from PIL import Image
import numpy as np
import torch
import argparse
from copy import deepcopy

from sketchkit.renderer3d import DiffVGRenderer
from sketchkit.renderer3d.diffvg_renderer import DiffVGRenderOptions
from sketchkit.utils.file import save_seq_gif, save_seq_video
from sketchkit.core.camera import Camera
from sketchkit.utils3d.transforms import wxyz_from_look_at
from sketchkit.renderer3d.renderer3d import Render3DOptions

from sketchkit.core.sketch3d import Sketch3D, Path3D, Curve3D, Point3D

import hsluv


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def generate_orbit_views(num_frames, fov_deg=60, radius=2.0):
    azimuths = torch.linspace(0, 2.0 * np.pi, num_frames)
    elevations = torch.ones_like(azimuths) * np.pi * 0.08
    azimuths = azimuths.tolist()
    elevations = elevations.tolist()

    cameras = []

    for azimuth, elevation in zip(azimuths, elevations):
        r_xz = radius * np.cos(elevation)
        y = radius * np.sin(elevation)
        x = r_xz * np.cos(azimuth)
        z = r_xz * np.sin(azimuth)

        xyz = np.array([x, y, z])

        camera = Camera()
        # camera.set_look_at([0, 0, 0])
        # Manually calculate wxyz with up=[0,1,0] for Y-up scene
        wxyz = wxyz_from_look_at(xyz, np.array([0, 0, 0]), up=np.array([0, 1, 0]))

        camera.set_xyz(xyz)
        camera.set_wxyz(wxyz)
        camera.set_fov(fov=(fov_deg / 180) * np.pi)
        cameras.append(camera)

    return cameras


def main(
    out_dir="./sketch3d_test_out",
    views=60,
    width=512,
    height=512,
    make_gif=True,
    stroke_color: tuple[int, int, int] | None = None,
    sketch_json: str | None = None,
):
    os.makedirs(out_dir, exist_ok=True)
    cameras = generate_orbit_views(views, fov_deg=60, radius=60)

    # Load or create sketch
    if sketch_json and os.path.exists(sketch_json):
        print(f"Loading sketch from {sketch_json}")
        sketch3d = Sketch3D.from_json(sketch_json)
        sketch3d1 = deepcopy(sketch3d)
        sketch3d2 = deepcopy(sketch3d)
    else:
        return

    # for path in sketch3d.paths:
    #     for curve in path.curves:
    #         curve.p_start.color = np.array((0, 0, 0))
    #         curve.p_start.thickness = 2.0
    #         curve.p_end.thickness = 2.0
    #         curve.p_ctrl1.thickness = 2.0
    #         curve.p_ctrl2.thickness = 2.0

    # for path in sketch3d1.paths:
    #     hue = np.random.rand() * 360
    #     r, g, b = hsluv.hsluv_to_rgb((hue, 90, 60))
    #     for curve in path.curves:
    #         curve.p_start.color = np.array((r, g, b))
    #         curve.p_start.z += 0.3
    #         curve.p_end.z += 0.3
    #         curve.p_ctrl1.z += 0.3
    #         curve.p_ctrl2.z += 0.3

    #         curve.p_start.thickness = 2.0 + random.uniform(2, 5)
    #         curve.p_end.thickness = 2.0 
    #         curve.p_ctrl1.thickness = 2.0 
    #         curve.p_ctrl2.thickness = 2.0 + random.uniform(2, 5)

    # for path in sketch3d2.paths:
    #     hue = np.random.rand() * 360
    #     r, g, b = hsluv.hsluv_to_rgb((hue, 90, 60))
    #     for curve in path.curves:
    #         curve.p_start.color = np.array((r, g, b))
    #         curve.p_start.z += 0.6
    #         curve.p_end.z += 0.6
    #         curve.p_ctrl1.z += 0.6
    #         curve.p_ctrl2.z += 0.6

    #         curve.p_start.thickness = 2.0 + random.uniform(2, 5)
    #         curve.p_end.thickness = 2.0
    #         curve.p_ctrl1.thickness = 2.0 
    #         curve.p_ctrl2.thickness = 2.0 + random.uniform(2, 5)

    print(sketch3d.path_num, sketch3d.curve_num)

    sketch3d.paths.extend(sketch3d1.paths)
    sketch3d.paths.extend(sketch3d2.paths)
    # sketch3d.paths = sketch3d.paths[::-1]

    print(sketch3d.path_num, sketch3d.curve_num)

    options = DiffVGRenderOptions(canvas_size=(width, height), stroke_width=0.1)

    options.device = "cuda" if torch.cuda.is_available() else "cpu"

    renderer = DiffVGRenderer(render_options=options)
    if renderer is None:
        raise RuntimeError(
            "Neither CairoRenderer nor pydiffvg backends are available for Sketch3DRenderer."
        )

    images = renderer.render(
        sketch3d,
        cameras=cameras[:10],
        render_options=options,
    )
    
    for img in images:
        img.save(os.path.join(out_dir, f"{images.index(img):03d}.png"))

    if make_gif:
        save_seq_video(images, os.path.join(out_dir, "orbit.mp4"), fps=30)
        save_seq_gif(images, os.path.join(out_dir, "orbit.gif"), fps=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "3dsketch rendering test with a LOGO curves (360 views)"
    )
    parser.add_argument("--out", type=str, default="./sketch3d_test_out")
    parser.add_argument("--views", type=int, default=60)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--stroke_color",
        type=int,
        nargs=3,
        help="Override stroke color (R G B) 0-255",
        default=None,
    )
    parser.add_argument(
        "--sketch_json",
        type=str,
        help="Path to JSON file to load sketch from",
        default=str(DATA_DIR / "sketch3d" / "sample.json"),
    )
    parser.add_argument(
        "--save_json",
        type=str,
        help="Path to save the generated sketch as JSON",
        default="./sketch3d_test_out/cis3d_sketch.json",
    )
    args = parser.parse_args()

    stroke_color = tuple(args.stroke_color) if args.stroke_color else None

    main(
        args.out,
        args.views,
        args.width,
        args.height,
        # make_gif=not args.no_gif,
        stroke_color=stroke_color,
        sketch_json=args.sketch_json,
    )
