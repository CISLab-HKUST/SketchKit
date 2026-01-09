import numpy as np
import cairocffi as cairo
import math
import hsluv
from PIL import Image
from dataclasses import dataclass

from sketchkit.core.sketch import Sketch, Curve
from sketchkit.renderer.renderer import Renderer, RenderOptions


@dataclass
class CairoRenderOptions(RenderOptions):
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    stroke_color: tuple[float, float, float] | None = None
    stroke_width: float | None = None
    stroke_opacity: float | None = None
    random_color: bool = False


# ---- Bézier helpers ----
def bezier_point(p0, p1, p2, p3, t):
    u = 1.0 - t
    return (
        u * u * u * p0[0]
        + 3 * u * u * t * p1[0]
        + 3 * u * t * t * p2[0]
        + t * t * t * p3[0],
        u * u * u * p0[1]
        + 3 * u * u * t * p1[1]
        + 3 * u * t * t * p2[1]
        + t * t * t * p3[1],
    )


def bezier_tangent(p0, p1, p2, p3, t):
    u = 1.0 - t
    dx = (
        3 * u * u * (p1[0] - p0[0])
        + 6 * u * t * (p2[0] - p1[0])
        + 3 * t * t * (p3[0] - p2[0])
    )
    dy = (
        3 * u * u * (p1[1] - p0[1])
        + 6 * u * t * (p2[1] - p1[1])
        + 3 * t * t * (p3[1] - p2[1])
    )
    return dx, dy


def polyline_lengths(pts):
    L = [0.0]
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        L.append(L[-1] + math.hypot(x1 - x0, y1 - y0))
    return L


def lerp(a, b, t):
    return a + (b - a) * t


class CairoRenderer(Renderer):
    def __init__(self, render_options: CairoRenderOptions | None = None):
        if render_options is None:
            render_options = CairoRenderOptions()
        self.render_options = render_options
        self.new_ctx(render_options)

    def new_ctx(self, render_options: CairoRenderOptions):
        if hasattr(self, "ctx"):
            del self.ctx
        if hasattr(self, "surface"):
            del self.surface

        canvas_size = (
            (512, 512)
            if render_options.canvas_size is None
            else render_options.canvas_size
        )
        surface = cairo.ImageSurface(
            cairo.FORMAT_ARGB32,
            canvas_size[0],
            canvas_size[1],
        )
        ctx = cairo.Context(surface)
        ctx.set_antialias(cairo.ANTIALIAS_BEST)

        ctx.set_antialias(cairo.ANTIALIAS_BEST)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)

        self.surface = surface
        self.ctx = ctx

    def clear_ctx(self):
        self.ctx.save()
        self.ctx.set_operator(cairo.OPERATOR_CLEAR)
        self.ctx.paint()
        self.ctx.restore()

    def _render(
        self, sketch: Sketch, render_options: CairoRenderOptions
    ) -> Image.Image:
        """
        Renders the given sketch using Cairo's graphics system with the provided render options.

        Args:
            sketch (Sketch): The sketch to be rendered.
            render_options (CairoRenderOptions): The options controlling the rendering behavior.

        Returns:
            np.ndarray: The rendered sketch as a 3-channel NumPy array.
        """
        if (
            self.surface.get_width() != render_options.canvas_size[0]
            or self.surface.get_height() != render_options.canvas_size[1]
        ):
            self.new_ctx(render_options)
        else:
            self.clear_ctx()

        # print(render_options.background_color)
        self.ctx.set_source_rgb(
            *render_options.background_color
        )  # Convert to range [0, 1]
        self.ctx.paint()

        # Rendering the sketch on the canvas
        for each_path in sketch.paths:
            if render_options.random_color:
                hue = np.random.rand() * 360
                r, g, b = hsluv.hsluv_to_rgb((hue, 90, 60))
            for curve in each_path.curves:
                if isinstance(curve, Curve):
                    p0, p1, p2, p3 = (
                        curve.p_start,
                        curve.p_crtl1,
                        curve.p_crtl2,
                        curve.p_end,
                    )

                    if not render_options.random_color:
                        # Determine color
                        if render_options.stroke_color is not None:
                            r, g, b = render_options.stroke_color
                        elif curve.p_start.color is not None:
                            r, g, b = curve.p_start.color
                        else:
                            r, g, b = (0, 0, 0)

                    # Determine opacity
                    if render_options.stroke_opacity is not None:
                        opacity = render_options.stroke_opacity
                    else:
                        opacity = curve.p_start.opacity
                    if opacity is None:
                        opacity = 1.0

                    self.ctx.set_source_rgba(r, g, b, opacity)

                    # Determine width
                    if render_options.stroke_width is not None:
                        self.ctx.set_line_width(render_options.stroke_width)
                    elif curve.p_start.thickness is not None:
                        self.ctx.set_line_width(curve.p_start.thickness)
                    else:
                        self.ctx.set_line_width(1.0)

                    self.ctx.move_to(p0.x, p0.y)
                    self.ctx.curve_to(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y)
                    self.ctx.stroke()

        surface_data = self.surface.get_data()
        raster_image = np.copy(np.asarray(surface_data)).reshape(
            render_options.canvas_size[1], render_options.canvas_size[0], 4
        )[:, :, :3]

        return Image.fromarray(raster_image, "RGB")
