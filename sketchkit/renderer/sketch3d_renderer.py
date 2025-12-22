import numpy as np
import torch
from PIL import Image
from copy import deepcopy

from sketchkit.core.sketch import Curve, Path, Point, Sketch, Vertex
from sketchkit.core.sketch3d import Sketch3D
from sketchkit.renderer.cairo_renderer import CairoRenderer
from sketchkit.utils3d.camera import intrinsics_from_fov_xy
from sketchkit.core.camera import Camera
from sketchkit.renderer.renderer import Renderer3D, Render3DOptions

try:
    import pydiffvg  # type: ignore

    _HAS_DIFFVG = True
except Exception:  # pragma: no cover - optional dependency
    pydiffvg = None
    _HAS_DIFFVG = False


class Sketch3DRenderer(Renderer3D):
    """Project and render 3D sketch curves with selectable renderer backend."""

    def __init__(
        self,
        render_options: Render3DOptions | None = None,
    ) -> None:
        if render_options is not None:
            self.render_options = render_options
        else:
            # Initialize default Render3DOptions
            bg_int = (255, 255, 255)
            self.render_options = Render3DOptions(
                canvas_size=(512, 512),   #default size
                background_color=bg_int, # type: ignore
                stroke_width=2.0,
                back_end="cairo",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

        # Setup device
        if isinstance(self.render_options.device, str):
            self.device = torch.device(self.render_options.device if torch.cuda.is_available() and "cuda" in self.render_options.device else "cpu")
        else:
            self.device = self.render_options.device

        # Setup backend
        self.backend = self.render_options.back_end.lower()
        if self.backend not in {"cairo", "diffvg"}:
            raise ValueError("Unsupported backend '{backend}'. Only 'cairo' and 'diffvg' are available.".format(backend=self.backend))
        
        if self.backend == "cairo":
            # Canvas is square in CairoRenderer; use max dimension to avoid cropping.
            w, h = self.render_options.canvas_size
            canvas_size = max(w, h)
            bg_float = tuple(c / 255.0 for c in self.render_options.background_color)
            self._cairo_renderer = CairoRenderer(canvas_size=canvas_size, canvas_color=bg_float)
        else:
            if not _HAS_DIFFVG:
                raise RuntimeError("pydiffvg is not installed; unable to use 'diffvg' backend.")
            self._cairo_renderer = None

    #exposed as public method
    def project_curves_ndc(
        self,
        curves: torch.Tensor,
        cameras: list[Camera] | Camera | None = None,
    ) -> torch.Tensor:
        if cameras is not None:
            if isinstance(cameras, Camera):
                cameras = [cameras]
            
            """Project Bézier control points to normalized device coordinates using camera matrices."""
            device = curves.device
            
            V = len(cameras)
            N, P = curves.shape[0], curves.shape[1]

            verts = curves.detach().cpu().numpy().reshape(-1, 3)
            # Homogeneous coordinates
            verts_homo = np.concatenate([verts, np.ones((verts.shape[0], 1))], axis=1)

            out = np.zeros((V, N, P, 2), dtype=np.float32)

            for vi in range(V):
                if cameras is not None:
                    ext = cameras[vi].extrinsics
                    K = cameras[vi].intrinsics

                # Camera coordinates: Xc = E @ Xw
                Xc_homo = (ext @ verts_homo.T).T
                z = np.clip(Xc_homo[:, 2:3], 1e-6, None)
                x_norm = Xc_homo[:, 0:1] / z
                y_norm = Xc_homo[:, 1:2] / z

                u = K[0, 0] * (x_norm) + K[0, 2]
                v = K[1, 1] * y_norm + K[1, 2]
                out[vi] = np.concatenate([u, v], axis=1).reshape(N, P, 2)

            return torch.from_numpy(out).to(device)
        else:
            raise ValueError("Must provide 'cameras' for projection.");

    @staticmethod
    def _normalize_colors(colors: np.ndarray | torch.Tensor | list | None) -> np.ndarray | None:
        if colors is None:
            return None
        arr = np.asarray(colors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] not in (3, 4):
            raise ValueError("Colors must be shaped [N,3] or [N,4].")
        return arr

    def _get_sketch_colors(self, sketch: Sketch3D) -> np.ndarray | None:
        colors = []
        has_color = False
        for path in sketch.paths:
            for curve in path.curves:
                c = curve.p_start.color
                if c is not None:
                    has_color = True
                    # Ensure we have a consistent shape, e.g. (3,) or (4,)
                    # If some are None and some are set, we need a default for None.
                    # We'll handle normalization later, just collect raw values here.
                    colors.append(c)
                else:
                    colors.append(None)
        
        if not has_color:
            return None
            
        # Fill Nones with black or some default if we have mixed content
        # Assuming if has_color is True, we want to use per-curve colors.
        final_colors = []
        for c in colors:
            if c is None:
                final_colors.append([0.0, 0.0, 0.0])
            else:
                final_colors.append(c)
        
        return np.array(final_colors, dtype=np.float32)

    def _resolve_stroke_colors(self, sketch: Sketch3D, options: Render3DOptions, num_curves: int) -> np.ndarray:
        """Resolve stroke colors from options or sketch, returning normalized [0,1] float array."""
        if options.stroke_color is not None:
            # options.stroke_color is 0-255 int/float tuple
            stroke_color = np.array([c / 255.0 for c in options.stroke_color], dtype=np.float32)
            colors_arr = np.tile(stroke_color, (num_curves, 1))
        else:
            colors_arr = self._get_sketch_colors(sketch)
            if colors_arr is None:
                colors_arr = np.zeros((num_curves, 3), dtype=np.float32)
        return colors_arr

    def _get_sketch_widths(self, sketch: Sketch3D) -> np.ndarray | None:
        widths = []
        has_width = False
        for path in sketch.paths:
            for curve in path.curves:
                w = curve.p_start.thickness
                if w is not None:
                    has_width = True
                    widths.append(w)
                else:
                    widths.append(None)
        
        if not has_width:
            return None
            
        final_widths = []
        for w in widths:
            if w is None:
                final_widths.append(2.0)
            else:
                final_widths.append(w)
        return np.array(final_widths, dtype=np.float32)

    def _resolve_stroke_widths(self, sketch: Sketch3D, options: Render3DOptions, num_curves: int) -> np.ndarray | float:
        """Resolve stroke widths from options or sketch."""
        sw = None
        if options is not None:
            # Check if options is a dict or object
            if isinstance(options, dict):
                 sw = options.get('stroke_width')
            else:
                 sw = getattr(options, 'stroke_width', None)

        if sw is not None:
            return float(sw)
        
        widths_arr = self._get_sketch_widths(sketch)
        if widths_arr is None:
            return 2.0
        return widths_arr

    # Project control points into a 2D Sketch representation for Cairo
    def _sketch_from_projected(
        self,
        projected: np.ndarray,
        colors: np.ndarray | None,
        stroke_width: float | np.ndarray,
    ) -> Sketch:
        paths: list[Path] = []
        for curve_idx, ctrl in enumerate(projected):
            px = ctrl[:, 0] * float(self.render_options.canvas_size[0])
            py = ctrl[:, 1] * float(self.render_options.canvas_size[1])   # no need to perform flip #(1.0 - ctrl[:, 1]) * float(self.height)
            rgb = None
            if colors is not None:
                rgb = tuple(float(c) for c in colors[curve_idx][:3])
            
            if stroke_width is None:
                sw = 1.0
            elif isinstance(stroke_width, (float, int)):
                sw = float(stroke_width)
            else:
                sw = float(stroke_width[curve_idx])

            v_start = Vertex(px[0], py[0], thickness=sw, color=rgb, opacity=1.0)
            v_end = Vertex(px[3], py[3], thickness=sw, color=rgb, opacity=1.0)
            p_ctrl1 = Point(px[1], py[1])
            p_ctrl2 = Point(px[2], py[2])
            paths.append(Path([Curve(v_start, v_end, p_ctrl1, p_ctrl2)]))
        return Sketch(paths)

    def _render_view_cairo(
        self,
        projected: np.ndarray,
        colors: np.ndarray | None,
        stroke_width: float | np.ndarray,
    ) -> Image.Image:
        if self._cairo_renderer is None:
            raise RuntimeError("Cairo renderer not initialized.")
        sketch2d = self._sketch_from_projected(projected, colors, stroke_width)
        
        w_arg = 2
        if isinstance(stroke_width, (float, int)):   #directly update with render_option
             w_arg = int(stroke_width)

        raster = self._cairo_renderer.render(sketch2d, width=w_arg)
        if raster.dtype != np.uint8:
            raster = np.clip(raster * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(raster, "RGB")

    @staticmethod
    def _diffvg_color(colors: np.ndarray | None, idx: int) -> torch.Tensor:
        if colors is None:
            return torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        row = colors[idx]
        if row.shape[0] == 3:
            rgba = np.empty(4, dtype=np.float32)
            rgba[:3] = row
            rgba[3] = 1.0
        else:
            rgba = row
        return torch.tensor(rgba, dtype=torch.float32)

    def _render_view_diffvg(
        self,
        projected: np.ndarray,
        colors: np.ndarray | None,
        stroke_width: float | np.ndarray,
        background_color: tuple[int, int, int] = (255, 255, 255),
        seed: int = 0,
    ) -> Image.Image:
        if not _HAS_DIFFVG:
            raise RuntimeError("pydiffvg is required for the 'diffvg' backend.")

        shapes = []
        shape_groups = []
        num_control_points = torch.zeros(1, dtype=torch.int32) + 2

        # synchronize the use of stroke_width from render_options and the vertex definitions
        if isinstance(stroke_width, (float, int)) and (stroke_width is None or stroke_width <= 0):
            stroke_width = 1.0   #should use some default value
        elif stroke_width is None:
            stroke_width = 1.0

        for curve_idx, ctrl in enumerate(projected):
            px = torch.tensor(ctrl[:, 0] * float(self.render_options.canvas_size[0]), dtype=torch.float32)
            py = torch.tensor(ctrl[:, 1] * float(self.render_options.canvas_size[1]), dtype=torch.float32)
            canvas_points = torch.stack([px, py], dim=-1)
            
            if isinstance(stroke_width, (float, int)):
                sw = float(stroke_width)
            else:
                sw = float(stroke_width[curve_idx])

            path = pydiffvg.Path(
                num_control_points=num_control_points,
                points=canvas_points,
                stroke_width=torch.tensor(sw, dtype=torch.float32),
                is_closed=False,
            )
            shapes.append(path)
            rgba = self._diffvg_color(colors, curve_idx)
            shape_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=rgba,
            )
            shape_groups.append(shape_group)

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.render_options.canvas_size[0], self.render_options.canvas_size[1], shapes, shape_groups
        )
        img = pydiffvg.RenderFunction.apply(
            self.render_options.canvas_size[0], self.render_options.canvas_size[1], 2, 2, seed, None, *scene_args
        )
        
        # Compositing with background
        bg_color = torch.tensor([c / 255.0 for c in background_color], device=img.device, dtype=torch.float32)
        
        if img.shape[-1] == 4:
            alpha = img[..., 3:4]
            rgb = img[..., :3]
            # DiffVG returns premultiplied alpha. Blend with background.
            # result = src_color + (1 - src_alpha) * dst_color
            img = rgb + (1 - alpha) * bg_color

        if img.dim() == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        #img = torch.flip(img, dims=(0,))
        img_np = (img.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(img_np, "RGB")

    # might need to be tested more thoroughly
    def _render(
        self,
        sketch: Sketch3D,
        camera: Camera,
        options: Render3DOptions
    ) -> np.ndarray:
        """
        Implementation of the abstract _render method from Renderer3D.
        Renders a single view to a numpy array.
        """
        
        curves = sketch.stack_tensor().to(self.device)
        colors_arr = self._resolve_stroke_colors(sketch, options, curves.shape[0])
        widths_arr = self._resolve_stroke_widths(sketch, options, curves.shape[0])
        
        projected = self.project_curves_ndc(curves, cameras=[camera])
        view_ctrl = projected[0].detach().cpu().numpy()
        
        if self.backend == "cairo":
            if self._cairo_renderer:
                 bg_float = tuple(c / 255.0 for c in options.background_color)
                 self._cairo_renderer.canvas_color = bg_float
            img = self._render_view_cairo(view_ctrl, colors_arr, widths_arr)
        else:
            img = self._render_view_diffvg(view_ctrl, colors_arr, widths_arr, background_color=options.background_color)
            
        return np.array(img)

    def render_batch(
        self,
        sketch3d: Sketch3D,
        cameras: list[Camera],
        options: Render3DOptions | dict | None = None
    ) -> list[Image.Image]:
        """Render each camera view to a PIL image using the configured backend."""

        curves = sketch3d.stack_tensor().to(self.device)
        colors_arr = self._resolve_stroke_colors(sketch3d, options, curves.shape[0])
        widths_arr = self._resolve_stroke_widths(sketch3d, options, curves.shape[0])
        projected = self.project_curves_ndc(curves, cameras=cameras)
        
        images: list[Image.Image] = []

        for view_idx in range(projected.shape[0]):
            view_ctrl = projected[view_idx].detach().cpu().numpy()
            if self.backend == "cairo":
                if self._cairo_renderer:
                     bg_float = tuple(c / 255.0 for c in options.background_color)
                     self._cairo_renderer.canvas_color = bg_float
                images.append(
                    self._render_view_cairo(view_ctrl, colors_arr, widths_arr)
                )
            else:
                images.append(
                    self._render_view_diffvg(view_ctrl, colors_arr, widths_arr, background_color=options.background_color)
                )
        return images

    #all from render_options
    def render(
        self,
        sketch3d: Sketch3D,
        cameras: list[Camera] | Camera | None = None,
        render_options: Render3DOptions | dict | None = None
    ) -> list[Image.Image] | np.ndarray:
        """
        Overridden render method to support both legacy batch rendering and new Renderer3D interface.
        """
        if render_options is None:
            opts = self.render_options
        elif isinstance(render_options, dict):
            opts = deepcopy(self.render_options)
            for k, v in render_options.items():
                if hasattr(opts, k):
                    setattr(opts, k, v)
        else:
            opts = render_options

        # Update dimensions if options differ
        w, h = opts.canvas_size
        if (w != self.render_options.canvas_size[0]) or (h != self.render_options.canvas_size[1]):
            if self.backend == "cairo" and self._cairo_renderer is not None:
                canvas_size = max(w, h)
                self._cairo_renderer.canvas_size = canvas_size

        if isinstance(cameras, list):
            # Legacy/Batch mode
            return self.render_batch(sketch3d, cameras, opts)
        elif isinstance(cameras, Camera):
            # New interface mode: render(sketch, camera, options)
            return self._render(sketch3d, cameras, opts)
        else:
             raise ValueError("Invalid arguments for render.")
