# 3D Renderer

The 3D renderer module converts `Sketch3D` Bezier curves into 2D raster images from one or more camera views.

This page documents the implementation in `sketchkit/renderer3d` and the practical usage pattern in `tests/test_3Dsketch.py`.

## Available Methods

```{eval-rst}
.. autosummary::
   :toctree: generated

   sketchkit.renderer3d.DiffVGRenderer
   sketchkit.renderer3d.diffvg_renderer.DiffVGRenderOptions
   sketchkit.renderer3d.renderer3d.Render3DOptions
```

## Module Layout

- `sketchkit/renderer3d/renderer3d.py`
  - Base abstractions: `Render3DOptions`, `Renderer3D`
  - Shared logic: camera projection, stroke color/width resolution, option merging
- `sketchkit/renderer3d/diffvg_renderer.py`
  - Concrete backend: `DiffVGRenderer` using `pydiffvg`
  - Option extension: `DiffVGRenderOptions`
- `sketchkit/renderer3d/cairo_renderer.py`
  - Alternate backend prototype for Cairo/DiffVG mixed logic
  - Not exported in `sketchkit.renderer3d.__init__` at present

## Sketch3D at a Glance

`Sketch3D` is the geometric container consumed by the 3D renderer.

- Hierarchy
  - `Sketch3D` contains multiple `Path3D` objects.
  - Each `Path3D` contains one or more cubic `Curve3D` objects.
  - Each `Curve3D` is defined by 4 control points: `p_start`, `p_ctrl1`, `p_ctrl2`, `p_end`.

- Point/vertex semantics
  - `Point3D` stores `(x, y, z)` and optional `thickness`.
  - `Vertex3D` extends `Point3D` with optional `color` and `opacity`.
  - This design enables geometry-only control points while keeping stroke style on endpoints.

- Tensor interface
  - `Curve3D.as_tensor()` exports control points as shape `[4, 3]`.
  - `Path3D.stack_tensor()` returns `[C, 4, 3]`.
  - `Sketch3D.stack_tensor()` concatenates all paths into `[N, 4, 3]` for batched projection/rendering.

- Width behavior
  - `Curve3D.widths` derives per-control-point widths from endpoint/control-point thicknesses.
  - Mixed missing values are filled using the first available width.
  - If no thickness is provided, width information is `None` and renderer-level defaults are used.

- Serialization
  - `Sketch3D.to_json(...)` and `Sketch3D.from_json(...)` support compact persistence of 3D sketches.
  - This is used in tests/workflows to save editable sketch geometry and re-render it from different views.

## Core Data Flow

### 1) Input geometry

`Sketch3D.stack_tensor()` packs all curves into a tensor of shape `[N, 4, 3]`:

- `N`: number of cubic curves
- `4`: start point, control point 1, control point 2, end point
- `3`: xyz coordinates

### 2) Camera projection (`Renderer3D.project_curves_ndc`)

For each camera view:

1. Convert control points to homogeneous coordinates.
2. Transform world points to camera space with `Camera.extrinsics`.
3. Perspective divide by depth `z`.
4. Apply `Camera.intrinsics`.

The camera intrinsics are generated from FoV by `intrinsics_from_fov_xy`, using normalized principal point `(0.5, 0.5)`. The projected output is organized as `[V, N, 4, 2]`, where `V` is number of views.

### 3) Style resolution

The base class resolves styling in a deterministic order:

- Color (`_resolve_stroke_colors`)
  - If `render_options.stroke_color` is set, it is used for all curves.
  - Else colors are read from each curve start vertex (`curve.p_start.color`).
  - Missing colors fall back to black.

- Width (`_resolve_stroke_widths`)
  - If `render_options.stroke_width` is set, it is treated as a global width.
  - Else per-curve widths are collected from `curve.widths` (4-point widths).
  - Else fallback width is `2.0`.

This supports both constant-width rendering and variable-width rendering along each Bezier curve.

### 4) Backend rasterization

`DiffVGRenderer._render_batch` iterates over views and calls `_render_view_diffvg`.

For each curve it creates:

- `pydiffvg.Path` with `num_control_points=[2]` (cubic Bezier)
- `points`: projected control points in pixel coordinates
- `stroke_width`: scalar or 4-value tensor
- `pydiffvg.ShapeGroup` with stroke RGBA

Then it serializes and renders the scene with `pydiffvg.RenderFunction.apply(...)`.

If DiffVG returns RGBA, alpha is composited onto `background_color`:

- `result = rgb + (1 - alpha) * background`

Final output is converted to `PIL.Image` in `RGB` mode.

## Public API Behavior

### `Render3DOptions`

Base options in `renderer3d.py`:

- `canvas_size: tuple[int, int] = (512, 512)`
- `background_color: tuple[int, int, int] = (255, 255, 255)`
- `stroke_color: tuple[int, int, int] | None = None`
- `stroke_width: float | None = None`

### `DiffVGRenderOptions`

Extends `Render3DOptions` with:

- `device: str = "cpu"`

### `DiffVGRenderer.render(...)`

Inherited from `Renderer3D.render`:

- Accepts `sketch3d`, `cameras`, and optional `render_options`
- Supports options as dataclass instance or `dict`
- Returns `list[PIL.Image.Image]` (one image per camera)

## Practical Workflow (from `test_3Dsketch.py`)

The test demonstrates a full 360-view pipeline:

1. Build orbit cameras (`generate_orbit_views`) around the sketch.
2. Build or load a `Sketch3D` (`Sketch3D.from_json` optional).
3. Configure `DiffVGRenderOptions`:
   - fixed width mode: `stroke_width=2.0`
   - varied or variable-width mode: `stroke_width=None`
4. Select runtime device (`"cuda"` if available, else `"cpu"`).
5. Render all views with `DiffVGRenderer.render(...)`.
6. Save animation with `save_seq_video` and `save_seq_gif`.

## Example

```python
import numpy as np
import torch

from sketchkit.core.camera import Camera
from sketchkit.core.sketch3d import Sketch3D
from sketchkit.renderer3d import DiffVGRenderer
from sketchkit.renderer3d.diffvg_renderer import DiffVGRenderOptions
from sketchkit.utils3d.transforms import wxyz_from_look_at


def build_orbit_cameras(num_views: int = 60, radius: float = 1.0):
    cameras = []
    for az in np.linspace(0.0, 2.0 * np.pi, num_views, endpoint=False):
        xyz = np.array([radius * np.cos(az), 0.0, radius * np.sin(az)])
        cam = Camera()
        cam.set_xyz(xyz)
        cam.set_wxyz(wxyz_from_look_at(xyz, np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 1.0, 0.0])))
        cam.set_fov(fov=np.deg2rad(60.0))
        cameras.append(cam)
    return cameras


sketch = Sketch3D.from_json("./sketch3d_test_out/cis3d_sketch.json")
cameras = build_orbit_cameras()

opts = DiffVGRenderOptions(canvas_size=(512, 512), stroke_width=None)
opts.device = "cuda" if torch.cuda.is_available() else "cpu"

renderer = DiffVGRenderer(render_options=opts)
images = renderer.render(sketch, cameras=cameras, render_options=opts)

images[0].save("first_view.png")
```

## Notes

- `pydiffvg` is required for `DiffVGRenderer`; initialization raises `RuntimeError` if unavailable.
- Current package export (`sketchkit.renderer3d`) exposes `DiffVGRenderer` directly.