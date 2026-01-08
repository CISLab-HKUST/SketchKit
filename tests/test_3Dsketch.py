import os
import glob
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import argparse

from sketchkit.renderer3d import DiffVGRenderer
from sketchkit.renderer3d.diffvg_renderer import DiffVGRenderOptions
from sketchkit.utils.file import save_seq_gif, save_seq_video
from sketchkit.core.camera import Camera
from sketchkit.utils3d.transforms import wxyz_from_look_at
from sketchkit.renderer3d.renderer3d import Render3DOptions

from sketchkit.core.sketch3d import Sketch3D, Path3D, Curve3D, Vertex3D, Point3D


# for 360 views rendering (wir3d way)
def generate_orbit_views(num_frames, fov_deg=60, radius=2.0):
    azimuths = torch.linspace(0, 2.0 * np.pi, num_frames)
    elevations = torch.zeros_like(azimuths) * np.pi * 0.25
    azimuths = azimuths.tolist()
    elevations = elevations.tolist()

    cameras = []

    for azimuth, elevation in zip(azimuths, elevations):
        # Orbit around Y axis (Y is up)
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


# create the CIS Logo
def make_cis_sketch3d(thicknesses=None, variable_width=False) -> Sketch3D:
    """Higher-level 3D sketch for letters CIS with depth offsets."""
    if thicknesses is None:
        thicknesses = [5.0, 5.0, 5.0]

    sketch = Sketch3D()
    # Depth layering: C slightly behind, I center, S forward
    depth_C, depth_I, depth_S = -0.1, 0.0, 0.1
    # Reuse 2D layout and add z
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

    # Helper to create a curve from control points and color
    def make_curve(ctrls, color, thickness, use_variable=False):
        if use_variable and isinstance(thickness, (list, tuple, np.ndarray)):
            # Variable width mode: thickness should be array of 4 values
            widths = list(thickness)
            v0 = Vertex3D(*ctrls[0], thickness=widths[0], color=color, opacity=1.0)
            v3 = Vertex3D(*ctrls[3], thickness=widths[3], color=color, opacity=1.0)
            c1 = Point3D(*ctrls[1], thickness=widths[1])
            c2 = Point3D(*ctrls[2], thickness=widths[2])
        else:
            # Constant width mode
            v0 = Vertex3D(*ctrls[0], thickness=thickness, color=color, opacity=1.0)
            v3 = Vertex3D(*ctrls[3], thickness=thickness, color=color, opacity=1.0)
            c1 = Point3D(*ctrls[1], thickness=thickness)
            c2 = Point3D(*ctrls[2], thickness=thickness)

        return Curve3D(v0, v3, c1, c2)

    # colors: C-red, I-green, S-blue
    if variable_width:
        # Demonstrate variable width: thick at ends, thin in middle
        pathC = Path3D(
            [make_curve(C_ctrls, (1, 0, 0), [8.0, 2.0, 2.0, 8.0], use_variable=True)]
        )
        pathI = Path3D(
            [make_curve(I_ctrls, (0, 0.6, 0), [1.0, 5.0, 5.0, 1.0], use_variable=True)]
        )
        pathS = Path3D(
            [make_curve(S_ctrls, (0, 0, 1), [15.0, 10.0, 5.0, 2.0], use_variable=True)]
        )
    else:
        pathC = Path3D([make_curve(C_ctrls, (1, 0, 0), thicknesses[0])])
        pathI = Path3D([make_curve(I_ctrls, (0, 0.6, 0), thicknesses[1])])
        pathS = Path3D([make_curve(S_ctrls, (0, 0, 1), thicknesses[2])])

    sketch.add(pathC)
    sketch.add(pathI)
    sketch.add(pathS)

    print(
        "3D sketch created with paths:",
        sketch.path_num,
        "and curves:",
        sketch.curve_num,
    )
    return sketch


# Utility to save a sequence of PNG frames as a GIF
def save_gif_from_dir(frames_dir, out_gif, fps=30):
    files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not files:
        return
    imgs = [Image.open(f).convert("RGB") for f in files]
    imgs[0].save(
        out_gif, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0
    )


# def _select_renderer(
#     options: Render3DOptions
# ):
#     """Return renderer matching backend preference ('cairo', 'diffvg')."""
#     if Sketch3DRenderer is None:
#         return None, None

#     renderer = Sketch3DRenderer(
#         render_options=options,
#     )
#     return options.back_end, renderer


# Main function to render CIS logo sketch from multiple views
def main(
    renderlist_path="./renderlist.json",
    out_dir="./sketch3d_test_out",
    views=60,
    width=512,
    height=512,
    make_gif=True,
    backend: str | None = "auto",
    stroke_color: tuple[int, int, int] | None = None,
    single_view: bool = False,
    varied_thickness: bool = False,
    variable_width: bool = False,
    sketch_json: str | None = None,
    save_json: str | None = None,
):
    os.makedirs(out_dir, exist_ok=True)
    # positions, lookats, fov = load_renderlist(renderlist_path)
    cameras = generate_orbit_views(views, fov_deg=60, radius=2.2)

    # Load or create sketch
    if sketch_json and os.path.exists(sketch_json):
        print(f"Loading sketch from {sketch_json}")
        cis3d = load_sketch3d_from_json(sketch_json)
    else:
        # Curves in world coordinates
        if varied_thickness:
            cis3d = make_cis_sketch3d(thicknesses=[5.0, 1.0, 13.0])
        else:
            cis3d = make_cis_sketch3d(variable_width=variable_width)
        
        # Save to JSON if requested
        if save_json:
            cis3d.to_json(save_json)

    if varied_thickness:
        options = DiffVGRenderOptions(canvas_size=(width, height), stroke_width=None)
    elif variable_width:
        # For variable width mode, don't use global stroke_width
        options = DiffVGRenderOptions(canvas_size=(width, height), stroke_width=None)
    else:
        options = DiffVGRenderOptions(canvas_size=(width, height), stroke_width=10.0)

    # options.back_end = backend
    options.device = "cuda" if torch.cuda.is_available() else "cpu"

    if stroke_color is not None:
        options.stroke_color = stroke_color

    # backend_name, renderer = _select_renderer(options)
    renderer = DiffVGRenderer(render_options=options)
    if renderer is None:
        raise RuntimeError(
            "Neither CairoRenderer nor pydiffvg backends are available for Sketch3DRenderer."
        )

    if single_view:
        print("Testing single camera render...")
        img = renderer.render(
            cis3d,
            cameras=cameras,
            render_options=options,
        )[0]
        img.save(os.path.join(out_dir, "single_view.png"))
        print(f"Saved single view to {os.path.join(out_dir, 'single_view.png')}")
        return

    images = renderer.render(
        cis3d,
        cameras=cameras,
        render_options=options,
    )

    # for i, pil in enumerate(images):
    #     pil.save(os.path.join(out_dir, f"{i:03d}.png"))
    if make_gif:
        save_seq_video(images, os.path.join(out_dir, "orbit.mp4"), fps=30)
        save_seq_gif(images, os.path.join(out_dir, "orbit.gif"), fps=10)
        # save_gif_from_dir(out_dir, os.path.join(out_dir, "orbit.gif"), fps=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "3dsketch rendering test with CIS LOGO curves (360 views)"
    )
    parser.add_argument("--renderlist", type=str, default="./renderlist.json")
    parser.add_argument("--out", type=str, default="./sketch3d_test_out")
    parser.add_argument("--views", type=int, default=60)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument(
        "--backend",
        type=str,
        default="cairo",
        choices=["cairo", "diffvg"],
        help="Rendering backend to use",
    )
    parser.add_argument(
        "--stroke_color",
        type=int,
        nargs=3,
        help="Override stroke color (R G B) 0-255",
        default=None,
    )
    # add single view test
    parser.add_argument(
        "--single_view", action="store_true", help="Render a single view for testing"
    )
    parser.add_argument(
        "--varied_thickness",
        action="store_true",
        help="Use varied thickness for C, I, S and no global stroke_width",
    )
    parser.add_argument(
        "--variable_width",
        action="store_true",
        help="Use variable width along each curve (4-point widths)",
    )
    parser.add_argument(
        "--sketch_json",
        type=str,
        help="Path to JSON file to load sketch from",
        default=None,
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
        args.renderlist,
        args.out,
        args.views,
        args.width,
        args.height,
        make_gif=not args.no_gif,
        backend=args.backend,
        stroke_color=stroke_color,
        single_view=args.single_view,
        varied_thickness=args.varied_thickness,
        variable_width=args.variable_width,
        sketch_json=args.sketch_json,
        save_json=args.save_json,
    )
