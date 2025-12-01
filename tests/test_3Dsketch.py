
import os, sys, json, glob
from PIL import Image
import numpy as np
import torch
from PIL import Image, ImageDraw
import argparse  

#sanity check for Cairo/SketchKit
try:
    from sketchkit.core.sketch import Vertex, Point, Curve, Path, Sketch
    from sketchkit.renderer.cairo_renderer import CairoRenderer
    _HAS_CAIRO = True
except ImportError:
    _HAS_CAIRO = False

from sketchkit.utils3d.sketch3d import Sketch3D, Path3D, Curve3D, Vertex3D, Point3D
from PIL import Image, ImageDraw

#transform
from sketchkit.utils3d.transforms import project_curves_ndc_opencv

#for 360 views rendering (wir3d way)
def load_renderlist(renderlist_path):
    with open(renderlist_path, "r") as f:
        rl = json.load(f)
    positions = np.asarray(rl["positions"], dtype=np.float32)
    lookats   = np.asarray(rl["lookats"],   dtype=np.float32)
    fov       = float(rl["fov"])  # radians
    return positions, lookats, fov

#create the CIS Logo
def make_cis_sketch3d() -> Sketch3D:
    """Higher-level 3D sketch for letters CIS with depth offsets."""
    sketch = Sketch3D()
    # Depth layering: C slightly behind, I center, S forward
    depth_C, depth_I, depth_S = -0.1, 0.0, 0.1
    # Reuse 2D layout and add z
    C_ctrls = [(-0.4, 0.6, depth_C), (-1.3,0.6,depth_C), (-1.3,-0.6,depth_C), (-0.4,-0.6,depth_C)]
    I_ctrls = [(0.0,-0.6,depth_I),(0.0,-0.2,depth_I),(0.0,0.2,depth_I),(0.0,0.6,depth_I)]
    S_ctrls = [(0.9,0.6,depth_S),(0.3,0.2,depth_S),(1.3,-0.2,depth_S),(0.9,-0.6,depth_S)]
    def make_curve(ctrls, color):
        v0 = Vertex3D(*ctrls[0], thickness=2.0, color=color, opacity=1.0)
        v3 = Vertex3D(*ctrls[3], thickness=2.0, color=color, opacity=1.0)
        c1 = Point3D(*ctrls[1]); c2 = Point3D(*ctrls[2])
        return Curve3D(v0,v3,c1,c2)
    pathC = Path3D([make_curve(C_ctrls,(1,0,0))])
    pathI = Path3D([make_curve(I_ctrls,(0,0.6,0))])
    pathS = Path3D([make_curve(S_ctrls,(0,0,1))])
    sketch.add(pathC); sketch.add(pathI); sketch.add(pathS)

    print("3D sketch created with paths:", sketch.path_num, "and curves:", sketch.curve_num)
    return sketch

# Utility to save a sequence of PNG frames as a GIF
def save_gif_from_dir(frames_dir, out_gif, fps=30):
    files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not files:
        return
    imgs = [Image.open(f).convert("RGB") for f in files]
    imgs[0].save(out_gif, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)

def render_with_cairo(vertices_ndc, width, height, colors=None, stroke_width=2.0, bg_rgb=(1,1,1)):
    """Render curves using SketchKit CairoRenderer.

    vertices_ndc: [N,4,2] in [0,1]
    colors: optional RGBA or RGB per curve (values 0-1)
    stroke_width: thickness to assign to start/end vertices
    Returns PIL.Image.
    """
    assert _HAS_CAIRO, "Cairo/SketchKit not available in environment"
    paths = []
    #print(f"Rendering with Cairo/SketchKit... {len(vertices_ndc)} curves")
    for ci, ctrl in enumerate(vertices_ndc):
        # Map ndc -> pixel coordinates (float)
        px = ctrl[:,0] * width
        #py = ctrl[:,1] * height
        py = (1.0 - ctrl[:,1]) * height
        # Create vertices and control points
        rgb = None
        if colors is not None:
            rgb = colors[ci][:3]
            #print(f"Using color: {rgb} for curve {ci}")
        v_start = Vertex(float(px[0]), float(py[0]), thickness=stroke_width, color=rgb, opacity=1.0)
        v_end   = Vertex(float(px[3]), float(py[3]), thickness=stroke_width, color=rgb, opacity=1.0)
        p_ctrl1 = Point(float(px[1]), float(py[1]))
        p_ctrl2 = Point(float(px[2]), float(py[2]))
        curve = Curve(v_start, v_end, p_ctrl1, p_ctrl2)
        path = Path([curve])
        paths.append(path)
    sketch = Sketch(paths)
    renderer = CairoRenderer(width, bg_rgb)
    arr = renderer.render(sketch)  # expects (H,W,3) np array RGB 0-255 or float
    if arr.dtype != np.uint8:
        arr = np.clip(arr*255,0,255).astype(np.uint8)
    return Image.fromarray(arr, 'RGB')


def main(renderlist_path="./renderlist.json",
    out_dir="./sketch3d_test_out",
    views=60, width=512, height=512, make_gif=True):
    os.makedirs(out_dir, exist_ok=True)
    positions, lookats, fov = load_renderlist(renderlist_path)
    if views is not None:
        positions = positions[:views]
        lookats   = lookats[:views]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Curves in world coordinates
    cis3d = make_cis_sketch3d()
    curves = cis3d.stack_tensor().to(device)  # [3,4,3]

    colors = np.array([
        [1, 0, 0, 1],   # C - red
        [0, 0.6, 0, 1], # I - green
        [0, 0, 1, 1],   # S - blue
    ], dtype=np.float32)


    verts_ndc = project_curves_ndc_opencv(curves, positions, lookats, fov, width, height)  # [V,3,4,2]
    
    #debug rendering info
    if  _HAS_CAIRO:
         print(f"Rendering with Cairo/SketchKit...")

    # Render each view
    for i in range(verts_ndc.shape[0]):
        view_ctrl = verts_ndc[i].detach().cpu().numpy()  # [N_curves,4,2]
        if _HAS_CAIRO:
            pil = render_with_cairo(view_ctrl, width, height, colors=colors, stroke_width=2.0)
        pil.save(os.path.join(out_dir, f"{i:03d}.png"))
    if make_gif:
        save_gif_from_dir(out_dir, os.path.join(out_dir, "orbit.gif"), fps=30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("3dsketch rendering test with CIS LOGO curves (360 views)")
    parser.add_argument("--renderlist", type=str,
                        default="./renderlist.json")
    parser.add_argument("--out", type=str, default="./sketch3d_test_out")
    parser.add_argument("--views", type=int, default=60)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    main(args.renderlist, args.out, args.views, args.width, args.height, make_gif=not args.no_gif)