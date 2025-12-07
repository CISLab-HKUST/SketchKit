import numpy as np
import torch
from PIL import Image

from sketchkit.core.sketch import Curve, Path, Point, Sketch, Vertex
from sketchkit.core.sketch3d import Sketch3D
from sketchkit.renderer.cairo_renderer import CairoRenderer
from sketchkit.utils3d.camera import intrinsics_from_fov_xy

try:
    import pydiffvg  # type: ignore

    _HAS_DIFFVG = True
except Exception:  # pragma: no cover - optional dependency
    pydiffvg = None
    _HAS_DIFFVG = False


class Sketch3DRenderer:
    """Project and render 3D sketch curves with selectable renderer backend."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        backend: str = "cairo",
        device: torch.device | None = None,
        up: np.ndarray | tuple[float, float, float] = (0.0, 1.0, 0.0),
        background_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.device = device if device is not None else torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.up = np.asarray(up, dtype=np.float32)
        backend = backend.lower()
        if backend not in {"cairo", "diffvg"}:
            raise ValueError("Unsupported backend '{backend}'. Only 'cairo' and 'diffvg' are available.".format(backend=backend))
        self.backend = backend

        if self.backend == "cairo":
            # Canvas is square in CairoRenderer; use max dimension to avoid cropping.
            canvas_size = max(self.width, self.height)
            self._cairo_renderer = CairoRenderer(canvas_size=canvas_size, canvas_color=background_rgb)
        else:
            if not _HAS_DIFFVG:
                raise RuntimeError("pydiffvg is not installed; unable to use 'diffvg' backend.")
            self._cairo_renderer = None

    @staticmethod
    def _rt_from_lookat_opencv(
        eye: np.ndarray, lookat: np.ndarray, up: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute OpenCV-style rotation and translation from look-at parameters."""
        z = lookat - eye
        z = z / (np.linalg.norm(z) + 1e-9)
        x = np.cross(z, up)
        x = x / (np.linalg.norm(x) + 1e-9)
        y = np.cross(x, z)
        R = np.stack([x, y, z], axis=0).astype(np.float32)
        t = (-R @ eye.astype(np.float32))
        return R, t

    @staticmethod
    def _project_curves_ndc_opencv(
        curves: torch.Tensor,
        cam_position: np.ndarray,
        cam_lookat: np.ndarray,
        fov_x: float,
        width: int,
        height: int,
        up: np.ndarray | tuple[float, float, float],
    ) -> torch.Tensor:
        """Project Bézier control points to normalized device coordinates."""
        device = curves.device
        V = cam_position.shape[0]
        N, P = curves.shape[0], curves.shape[1]

        aspect = float(height) / float(width)
        fov_y = 2.0 * np.arctan(aspect * np.tan(fov_x / 2.0))
        K = intrinsics_from_fov_xy(fov_x, fov_y)

        verts = curves.detach().cpu().numpy().reshape(-1, 3)
        out = np.zeros((V, N, P, 2), dtype=np.float32)
        up_vec = np.asarray(up, dtype=np.float32)

        for vi in range(V):
            eye = cam_position[vi].astype(np.float32)
            tgt = cam_lookat[vi].astype(np.float32)
            R, t = Sketch3DRenderer._rt_from_lookat_opencv(eye, tgt, up_vec)

            Xc = (R @ verts.T + t[:, None]).T
            z = np.clip(Xc[:, 2:3], 1e-6, None)
            x_norm = Xc[:, 0:1] / z
            y_norm = Xc[:, 1:2] / z
            u = K[0, 0] * x_norm + K[0, 2]
            v = K[1, 1] * y_norm + K[1, 2]
            out[vi] = np.concatenate([u, v], axis=1).reshape(N, P, 2)

        return torch.from_numpy(out).to(device)

    def project_curves_ndc_opencv(
        self,
        curves: torch.Tensor,
        cam_position: np.ndarray,
        cam_lookat: np.ndarray,
        fov_x: float,
    ) -> torch.Tensor:
        return self._project_curves_ndc_opencv(
            curves,
            cam_position,
            cam_lookat,
            fov_x,
            self.width,
            self.height,
            self.up,
        )

    @staticmethod
    def _normalize_colors(colors: np.ndarray | torch.Tensor | list | None) -> np.ndarray | None:
        if colors is None:
            return None
        arr = np.asarray(colors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] not in (3, 4):
            raise ValueError("Colors must be shaped [N,3] or [N,4].")
        return arr

    # Project control points into a 2D Sketch representation for Cairo
    def _sketch_from_projected(
        self,
        projected: np.ndarray,
        colors: np.ndarray | None,
        stroke_width: float,
    ) -> Sketch:
        paths: list[Path] = []
        for curve_idx, ctrl in enumerate(projected):
            px = ctrl[:, 0] * float(self.width)
            py = (1.0 - ctrl[:, 1]) * float(self.height)
            rgb = None
            if colors is not None:
                rgb = tuple(float(c) for c in colors[curve_idx][:3])
            v_start = Vertex(px[0], py[0], thickness=stroke_width, color=rgb, opacity=1.0)
            v_end = Vertex(px[3], py[3], thickness=stroke_width, color=rgb, opacity=1.0)
            p_ctrl1 = Point(px[1], py[1])
            p_ctrl2 = Point(px[2], py[2])
            paths.append(Path([Curve(v_start, v_end, p_ctrl1, p_ctrl2)]))
        return Sketch(paths)

    def _render_view_cairo(
        self,
        projected: np.ndarray,
        colors: np.ndarray | None,
        stroke_width: float,
    ) -> Image.Image:
        if self._cairo_renderer is None:
            raise RuntimeError("Cairo renderer not initialized.")
        sketch2d = self._sketch_from_projected(projected, colors, stroke_width)
        raster = self._cairo_renderer.render(sketch2d, width=int(stroke_width))
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
        stroke_width: float,
        *,
        seed: int = 0,
        white_bg: bool = True,
    ) -> Image.Image:
        if not _HAS_DIFFVG:
            raise RuntimeError("pydiffvg is required for the 'diffvg' backend.")

        shapes = []
        shape_groups = []
        num_control_points = torch.zeros(1, dtype=torch.int32) + 2

        for curve_idx, ctrl in enumerate(projected):
            px = torch.tensor(ctrl[:, 0] * float(self.width), dtype=torch.float32)
            py = torch.tensor(ctrl[:, 1] * float(self.height), dtype=torch.float32)
            canvas_points = torch.stack([px, py], dim=-1)
            path = pydiffvg.Path(
                num_control_points=num_control_points,
                points=canvas_points,
                stroke_width=torch.tensor(float(stroke_width), dtype=torch.float32),
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
            self.width, self.height, shapes, shape_groups
        )
        img = pydiffvg.RenderFunction.apply(
            self.width, self.height, 2, 2, seed, None, *scene_args
        )
        if white_bg and img.shape[-1] == 4:
            opacity = img[..., 3:4]
            img = opacity * img[..., :3] + (1 - opacity)
        if img.dim() == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        img = torch.flip(img, dims=(0,))
        img_np = (img.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(img_np, "RGB")

    def render(
        self,
        sketch3d: Sketch3D,
        cam_position: np.ndarray,
        cam_lookat: np.ndarray,
        fov_x: float,
        *,
        colors: np.ndarray | torch.Tensor | list | None = None,
        stroke_width: float = 2.0,
        projected: torch.Tensor | None = None,
    ) -> list[Image.Image]:
        """Render each camera view to a PIL image using the configured backend."""
        curves = sketch3d.stack_tensor().to(self.device)
        colors_arr = self._normalize_colors(colors)
        if projected is None:
            projected = self.project_curves_ndc_opencv(curves, cam_position, cam_lookat, fov_x)
        images: list[Image.Image] = []

        for view_idx in range(projected.shape[0]):
            view_ctrl = projected[view_idx].detach().cpu().numpy()
            if self.backend == "cairo":
                images.append(self._render_view_cairo(view_ctrl, colors_arr, stroke_width))
            else:
                images.append(self._render_view_diffvg(view_ctrl, colors_arr, stroke_width))
        return images
