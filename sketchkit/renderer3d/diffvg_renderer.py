from dataclasses import dataclass
import numpy as np
import torch
from PIL import Image

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

            self.render_options = Render3DOptions(
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
        if isinstance(stroke_width, (float, int)) and (
            stroke_width is None or stroke_width <= 0
        ):
            stroke_width = 1.0  # should use some default value
        elif stroke_width is None:
            stroke_width = 1.0

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

            path = pydiffvg.Path(
                num_control_points=num_control_points,
                points=canvas_points,
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
        img = pydiffvg.RenderFunction.apply(
            self.render_options.canvas_size[0],
            self.render_options.canvas_size[1],
            2,
            2,
            seed,
            None,
            *scene_args,
        )

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

        curves = sketch3d.stack_tensor().to(options.device)
        colors_arr = self._resolve_stroke_colors(sketch3d, options, curves.shape[0])
        widths_arr = self._resolve_stroke_widths(sketch3d, options, curves.shape[0])
        projected = self.project_curves_ndc(curves, cameras=cameras)

        images: list[Image.Image] = []

        for view_idx in range(projected.shape[0]):
            view_ctrl = projected[view_idx].detach().cpu().numpy()
            images.append(
                self._render_view_diffvg(
                    view_ctrl,
                    colors_arr,
                    widths_arr,
                    background_color=options.background_color,
                )
            )
        return images
