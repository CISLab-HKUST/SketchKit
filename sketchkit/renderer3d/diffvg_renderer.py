from dataclasses import dataclass
import numpy as np
import torch
from PIL import Image
import warnings

from sketchkit.core.sketch3d import Sketch3D
from sketchkit.core.camera import Camera
from sketchkit.renderer3d.renderer3d import Renderer3D, Render3DOptions

try:
    import pydiffvg  # type: ignore

    _HAS_DIFFVG = True
except Exception:  # pragma: no cover - optional dependency
    pydiffvg = None
    _HAS_DIFFVG = False


@dataclass
class DiffVGRenderOptions(Render3DOptions):
    device: str = "cpu"


class DiffVGRenderer(Renderer3D):
    """Project and render 3D sketch curves with selectable renderer backend."""

    def __init__(
        self,
        render_options: DiffVGRenderOptions | None = None,
    ) -> None:
        if render_options is not None:
            self.render_options = render_options
        else:
            # Initialize default Render3DOptions

            self.render_options = DiffVGRenderOptions(
                canvas_size=(512, 512),
                background_color=(255, 255, 255),
                stroke_width=None,
            )

        # # Setup device
        # if isinstance(self.render_options.device, str):
        #     self.device = torch.device(
        #         self.render_options.device
        #         if torch.cuda.is_available() and "cuda" in self.render_options.device
        #         else "cpu"
        #     )
        # else:
        #     self.device = self.render_options.device

        # Setup backend
        # self.backend = self.render_options.back_end.lower()
        # if self.backend not in {"cairo", "diffvg"}:
        #     raise ValueError(
        #         "Unsupported backend '{backend}'. Only 'cairo' and 'diffvg' are available.".format(
        #             backend=self.backend
        #         )
        #     )

        if not _HAS_DIFFVG:
            raise RuntimeError(
                "pydiffvg is not installed; unable to use 'diffvg' backend."
            )

        # Sticky guard: once DiffVG CUDA illegal access appears, keep this
        # renderer on CPU to avoid crashing subsequent frames.
        self._force_cpu_diffvg = False
        self._warned_cpu_fallback = False

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

    @staticmethod
    def _is_near_linear_cubic(points: torch.Tensor, eps: float = 1e-3) -> bool:
        """Detect cubics whose control points lie almost on the endpoint line."""
        if points.shape != (4, 2):
            return False

        p0, p1, p2, p3 = points
        base = p3 - p0
        base_len = torch.linalg.norm(base)
        if float(base_len) <= eps:
            return True

        # 2D point-to-line distance via normalized cross product magnitude.
        rel1 = p1 - p0
        rel2 = p2 - p0
        d1 = torch.abs(base[0] * rel1[1] - base[1] * rel1[0]) / base_len
        d2 = torch.abs(base[0] * rel2[1] - base[1] * rel2[0]) / base_len
        return bool(torch.max(d1, d2) <= eps)

    @staticmethod
    def _is_cuda_illegal_access_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return (
            "illegal memory access" in msg
            or ("cuda runtime error" in msg and "cuda" in msg)
            or ("cuda error" in msg and "illegal" in msg)
        )

    @staticmethod
    def _device_is_cuda(device: str | torch.device) -> bool:
        if isinstance(device, torch.device):
            return device.type == "cuda"
        if isinstance(device, str):
            return "cuda" in device.lower()
        return False

    def _set_pydiffvg_cpu_runtime(self) -> None:
        if not _HAS_DIFFVG:
            return
        if hasattr(pydiffvg, "set_use_gpu"):
            pydiffvg.set_use_gpu(False)
        if hasattr(pydiffvg, "set_device"):
            pydiffvg.set_device(torch.device("cpu"))

    @staticmethod
    def _max_stroke_width(stroke_width: float | np.ndarray) -> float:
        if isinstance(stroke_width, (float, int)):
            return float(stroke_width)
        if isinstance(stroke_width, np.ndarray):
            if stroke_width.size == 0:
                return 0.0
            return float(np.max(stroke_width))
        return 0.0

    def _should_force_cpu_view(
        self,
        projected: np.ndarray,
        stroke_width: float | np.ndarray,
    ) -> bool:
        """Detect unstable GPU render cases and force CPU before kernel launch."""
        if self._force_cpu_diffvg:
            return True
        if not _HAS_DIFFVG:
            return False
        if not hasattr(pydiffvg, "get_use_gpu") or not pydiffvg.get_use_gpu():
            return False
        if projected.ndim != 3 or projected.shape[0] < 3000:
            return False

        max_width = self._max_stroke_width(stroke_width)
        if max_width < 2.0:
            return False

        canvas_wh = np.array(self.render_options.canvas_size, dtype=np.float32)
        projected_px = projected * canvas_wh[None, None, :]
        span = np.max(projected_px, axis=1) - np.min(projected_px, axis=1)
        extent = np.max(span, axis=1)
        median_extent = float(np.median(extent))
        small_ratio = float(np.mean(extent < 0.4))

        return median_extent < 1.2 or small_ratio > 0.45

    def _apply_diffvg(self, scene_args: tuple, seed: int) -> torch.Tensor:
        return pydiffvg.RenderFunction.apply(
            self.render_options.canvas_size[0],
            self.render_options.canvas_size[1],
            2,
            2,
            seed,
            None,
            *scene_args,
        )

    def _apply_gpu_width_safety(
        self,
        projected: np.ndarray,
        stroke_width: float | np.ndarray,
    ) -> tuple[float | np.ndarray, bool]:
        """Clamp large widths for highly compressed GPU scenes to avoid DiffVG CUDA faults."""
        if not _HAS_DIFFVG:
            return stroke_width, False
        if self._force_cpu_diffvg:
            return stroke_width, False
        if not hasattr(pydiffvg, "get_use_gpu") or not pydiffvg.get_use_gpu():
            return stroke_width, False
        if projected.ndim != 3 or projected.shape[0] < 1000:
            return stroke_width, False

        canvas_wh = np.array(self.render_options.canvas_size, dtype=np.float32)
        projected_px = projected * canvas_wh[None, None, :]
        span = np.max(projected_px, axis=1) - np.min(projected_px, axis=1)
        extent = np.max(span, axis=1)
        median_extent = float(np.median(extent))

        # Empirically, CUDA instability appears when many curves collapse to near-subpixel size.
        if median_extent >= 0.5:
            return stroke_width, False

        safe_max_width = 2.0
        if isinstance(stroke_width, (float, int)):
            sw = float(stroke_width)
            if sw > safe_max_width:
                return safe_max_width, True
            return stroke_width, False

        if isinstance(stroke_width, np.ndarray):
            if stroke_width.size == 0:
                return stroke_width, False
            max_w = float(np.max(stroke_width))
            if max_w > safe_max_width:
                return np.minimum(stroke_width.astype(np.float32), safe_max_width), True

        return stroke_width, False

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
        if isinstance(stroke_width, (float, int)) and (
            stroke_width is None or stroke_width <= 0
        ):
            stroke_width = 1.0  # should use some default value
        elif stroke_width is None:
            stroke_width = 1.0

        if self._should_force_cpu_view(projected, stroke_width):
            self._force_cpu_diffvg = True
            self._set_pydiffvg_cpu_runtime()
            if not self._warned_cpu_fallback:
                warnings.warn(
                    "Forced DiffVG CPU rendering for a high-risk dense view to "
                    "avoid CUDA illegal memory access.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_cpu_fallback = True

        stroke_width, width_was_clamped = self._apply_gpu_width_safety(
            projected,
            stroke_width,
        )
        if width_was_clamped:
            warnings.warn(
                "Clamped stroke width for a highly compressed GPU view to avoid "
                "DiffVG CUDA illegal memory access.",
                RuntimeWarning,
                stacklevel=2,
            )

        for curve_idx, ctrl in enumerate(projected):
            px = torch.tensor(
                ctrl[:, 0] * float(self.render_options.canvas_size[0]),
                dtype=torch.float32,
            )
            py = torch.tensor(
                ctrl[:, 1] * float(self.render_options.canvas_size[1]),
                dtype=torch.float32,
            )
            canvas_points = torch.stack([px, py], dim=-1)

            # Handle variable width: can be scalar or array of 4 values
            if isinstance(stroke_width, (float, int)):
                # Global constant width for all curves
                sw = torch.tensor(float(stroke_width), dtype=torch.float32)
            elif isinstance(stroke_width, np.ndarray):
                if stroke_width.ndim == 1:
                    # Per-curve scalar: shape (N,)
                    sw = torch.tensor(
                        float(stroke_width[curve_idx]), dtype=torch.float32
                    )
                elif stroke_width.ndim == 2:
                    # Per-curve 4-point widths: shape (N, 4)
                    sw = torch.tensor(stroke_width[curve_idx], dtype=torch.float32)
                else:
                    raise ValueError(
                        f"Unexpected stroke_width shape: {stroke_width.shape}"
                    )
            else:
                sw = torch.tensor(1.0, dtype=torch.float32)

            # DiffVG can produce fragmented strokes for near-linear cubic Beziers.
            # Emit a true line segment in this specific case for robust rendering.
            if self._is_near_linear_cubic(canvas_points):
                path_num_control_points = torch.zeros(1, dtype=torch.int32)
                path_points = torch.stack([canvas_points[0], canvas_points[3]], dim=0)
            else:
                path_num_control_points = num_control_points
                path_points = canvas_points

            path = pydiffvg.Path(
                num_control_points=path_num_control_points,
                points=path_points,
                stroke_width=sw,
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
            self.render_options.canvas_size[0],
            self.render_options.canvas_size[1],
            shapes,
            shape_groups,
        )

        if self._force_cpu_diffvg:
            self._set_pydiffvg_cpu_runtime()

        try:
            img = self._apply_diffvg(scene_args, seed)
        except RuntimeError as exc:
            if not self._is_cuda_illegal_access_error(exc):
                raise

            # Switch permanently to CPU for this renderer instance.
            self._force_cpu_diffvg = True
            self._set_pydiffvg_cpu_runtime()
            if not self._warned_cpu_fallback:
                warnings.warn(
                    "Detected DiffVG CUDA illegal memory access; switched this "
                    "renderer to CPU for remaining views.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_cpu_fallback = True

            img = self._apply_diffvg(scene_args, seed)

        # Compositing with background
        bg_color = torch.tensor(
            [c / 255.0 for c in background_color],
            device=img.device,
            dtype=torch.float32,
        )

        if img.shape[-1] == 4:
            alpha = img[..., 3:4]
            rgb = img[..., :3]
            # DiffVG returns premultiplied alpha. Blend with background.
            # result = src_color + (1 - src_alpha) * dst_color
            img = rgb + (1 - alpha) * bg_color

        if img.dim() == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        # img = torch.flip(img, dims=(0,))
        img_np = (img.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(img_np, "RGB")

    def _render_batch(
        self,
        sketch3d: Sketch3D,
        cameras: list[Camera],
        options: DiffVGRenderOptions,
    ) -> list[Image.Image]:
        """Render each camera view to a PIL image using the configured backend."""

        curve_device = options.device
        if self._force_cpu_diffvg and self._device_is_cuda(curve_device):
            curve_device = torch.device("cpu")

        curves = sketch3d.stack_tensor().to(curve_device)
        colors_arr = self._resolve_stroke_colors(sketch3d, options, curves.shape[0])
        widths_arr = self._resolve_stroke_widths(sketch3d, options, curves.shape[0])
        projected, curve_depths, visibility = self.project_curves_ndc_with_depth(
            curves, cameras=cameras
        )

        images: list[Image.Image] = []

        for view_idx in range(projected.shape[0]):
            depth_np = curve_depths[view_idx].detach().cpu().numpy()
            visible_np = visibility[view_idx].detach().cpu().numpy().astype(bool)

            # Painter's algorithm: draw from far to near in camera space.
            order_np = np.argsort(depth_np)[::-1]
            order_np = order_np[visible_np[order_np]]

            if order_np.size == 0:
                images.append(
                    Image.new(
                        "RGB",
                        options.canvas_size,
                        options.background_color,
                    )
                )
                continue

            order_t = torch.as_tensor(
                order_np,
                dtype=torch.long,
                device=projected.device,
            )
            view_ctrl = projected[view_idx].index_select(0, order_t).detach().cpu().numpy()

            view_colors = colors_arr[order_np] if colors_arr is not None else None
            if isinstance(widths_arr, np.ndarray):
                view_widths = widths_arr[order_np]
            else:
                view_widths = widths_arr

            images.append(
                self._render_view_diffvg(
                    view_ctrl,
                    view_colors,
                    view_widths,
                    background_color=options.background_color,
                )
            )
        return images
