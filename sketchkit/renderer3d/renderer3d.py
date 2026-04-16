from PIL import Image

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from copy import deepcopy


from sketchkit.core.sketch3d import Sketch3D
from sketchkit.core.camera import Camera
from sketchkit.core.sketch import Curve, Path, Point, Sketch, Vertex


@dataclass
class Render3DOptions:
    canvas_size: tuple[int, int] = (512, 512)
    background_color: tuple[int, int, int] = (255, 255, 255)
    stroke_color: tuple[int, int, int] | None = None
    stroke_width: float | None = None
    # back_end: str = "cairo"  # "cairo" or "pydiffvg"
    # device: str = "cuda"  # "cpu" or "cuda"


class Renderer3D(ABC):
    render_options = Render3DOptions()

    def _project_curves_numpy(
        self,
        curves: torch.Tensor,
        cameras: list[Camera],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project curves and return (uv, mean_depth, visibility) in numpy."""
        N, P = curves.shape[0], curves.shape[1]
        verts = curves.detach().cpu().numpy().reshape(-1, 3)
        verts_homo = np.concatenate([verts, np.ones((verts.shape[0], 1))], axis=1)

        projected = np.zeros((len(cameras), N, P, 2), dtype=np.float32)
        mean_depth = np.zeros((len(cameras), N), dtype=np.float32)
        visible = np.zeros((len(cameras), N), dtype=bool)

        for vi, camera in enumerate(cameras):
            ext = camera.extrinsics
            K = camera.intrinsics

            # Camera coordinates: Xc = E @ Xw
            Xc_homo = (ext @ verts_homo.T).T
            z = Xc_homo[:, 2:3]

            # Avoid unstable projection around the camera plane.
            near = float(max(getattr(camera, "z_near", 0.1), 1e-6))
            safe_z = np.where(z > near, z, near)

            x_norm = Xc_homo[:, 0:1] / safe_z
            y_norm = Xc_homo[:, 1:2] / safe_z

            u = K[0, 0] * x_norm + K[0, 2]
            v = K[1, 1] * y_norm + K[1, 2]
            projected[vi] = np.concatenate([u, v], axis=1).reshape(N, P, 2)

            z_curve = z.reshape(N, P)
            mean_depth[vi] = z_curve.mean(axis=1).astype(np.float32)
            visible[vi] = np.any(z_curve > near, axis=1)

        return projected, mean_depth, visible

    # exposed as public method
    def project_curves_ndc(
        self,
        curves: torch.Tensor,
        cameras: list[Camera] | Camera | None = None,
    ) -> torch.Tensor:
        if cameras is not None:
            if isinstance(cameras, Camera):
                cameras = [cameras]

            projected, _, _ = self._project_curves_numpy(curves, cameras)
            return torch.from_numpy(projected).to(curves.device)
        else:
            raise ValueError("Must provide 'cameras' for projection.")

    def project_curves_ndc_with_depth(
        self,
        curves: torch.Tensor,
        cameras: list[Camera] | Camera | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project curves and return (uv, mean_depth, visibility_mask)."""
        if cameras is None:
            raise ValueError("Must provide 'cameras' for projection.")

        if isinstance(cameras, Camera):
            cameras = [cameras]

        projected, mean_depth, visible = self._project_curves_numpy(curves, cameras)

        device = curves.device
        return (
            torch.from_numpy(projected).to(device),
            torch.from_numpy(mean_depth).to(device),
            torch.from_numpy(visible).to(device),
        )

    @staticmethod
    def _normalize_colors(
        colors: np.ndarray | torch.Tensor | list | None,
    ) -> np.ndarray | None:
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

    def _resolve_stroke_colors(
        self, sketch: Sketch3D, options: Render3DOptions, num_curves: int
    ) -> np.ndarray:
        """Resolve stroke colors from options or sketch, returning normalized [0,1] float array."""
        if options.stroke_color is not None:
            # options.stroke_color is 0-255 int/float tuple
            stroke_color = np.array(
                [c / 255.0 for c in options.stroke_color], dtype=np.float32
            )
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
                # Check if curve has 4-point width array
                if curve.widths is not None:
                    has_width = True
                    widths.append(curve.widths)  # shape (4,)
                # Fallback to vertex thickness
                elif curve.p_start.thickness is not None:
                    has_width = True
                    # Use same width for all 4 control points
                    w = curve.p_start.thickness
                    widths.append(np.array([w, w, w, w], dtype=np.float32))
                else:
                    widths.append(None)

        if not has_width:
            return None

        final_widths = []
        for w in widths:
            if w is None:
                final_widths.append(np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32))
            else:
                final_widths.append(w)
        return np.array(final_widths, dtype=np.float32)  # shape (N, 4)

    def _resolve_stroke_widths(
        self, sketch: Sketch3D, options: Render3DOptions, num_curves: int
    ) -> np.ndarray | float:
        """Resolve stroke widths from options or sketch."""
        sw = None
        if options is not None:
            # Check if options is a dict or object
            if isinstance(options, dict):
                sw = options.get("stroke_width")
            else:
                sw = getattr(options, "stroke_width", None)

        if sw is not None:
            # Global stroke width: return scalar or expand to (N, 4)
            return float(sw)

        widths_arr = self._get_sketch_widths(sketch)
        if widths_arr is None:
            return 2.0
        # widths_arr is now shape (N, 4) with per-control-point widths
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
            py = ctrl[:, 1] * float(
                self.render_options.canvas_size[1]
            )  # no need to perform flip #(1.0 - ctrl[:, 1]) * float(self.height)
            rgb = None
            if colors is not None:
                rgb = tuple(float(c) for c in colors[curve_idx][:3])

            if stroke_width is None:
                sw = 1.0
            elif isinstance(stroke_width, (float, int)):
                sw = float(stroke_width)
            elif isinstance(stroke_width, np.ndarray):
                if stroke_width.ndim == 1:
                    # Per-curve scalar: shape (N,)
                    sw = float(stroke_width[curve_idx])
                elif stroke_width.ndim == 2:
                    # Per-curve 4-point widths: shape (N, 4)
                    # For Cairo, use average or starting width
                    sw = float(stroke_width[curve_idx, 0])
                else:
                    sw = 1.0
            else:
                sw = 1.0

            v_start = Vertex(px[0], py[0], thickness=sw, color=rgb, opacity=1.0)
            v_end = Vertex(px[3], py[3], thickness=sw, color=rgb, opacity=1.0)
            p_ctrl1 = Point(px[1], py[1])
            p_ctrl2 = Point(px[2], py[2])
            paths.append(Path([Curve(v_start, v_end, p_ctrl1, p_ctrl2)]))
        return Sketch(paths)

    def render(
        self,
        sketch3d: Sketch3D,
        cameras: list[Camera],
        render_options: Render3DOptions | dict | None = None,
    ) -> list[Image.Image]:
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

        return self._render_batch(sketch3d, cameras, opts)

    @abstractmethod
    def _render_batch(self, *args, **kwargs) -> list[Image.Image]:
        pass
