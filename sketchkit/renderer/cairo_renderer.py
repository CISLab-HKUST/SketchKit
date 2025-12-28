from sketchkit.core.sketch import Sketch, Curve
import numpy as np
import cairocffi as cairo
import math
from PIL import Image
import random
from sketchkit.renderer import Renderer
from dataclasses import dataclass

@dataclass
class CairoRenderOptions:
    canvas_size: tuple[int, int] = (512, 512)
    background_color: tuple[int, int, int] = (255, 255, 255)
    stroke_color: tuple[int, int, int] = (0, 0, 0)
    stroke_width: float = 1.0
    output_path: str = "cairo_1228.png"

# ---- Bézier helpers ----
def bezier_point(p0, p1, p2, p3, t):
    u = 1.0 - t
    return (
        u*u*u*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t*t*t*p3[0],
        u*u*u*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t*t*t*p3[1],
    )

def bezier_tangent(p0, p1, p2, p3, t):
    u = 1.0 - t
    dx = 3*u*u*(p1[0]-p0[0]) + 6*u*t*(p2[0]-p1[0]) + 3*t*t*(p3[0]-p2[0])
    dy = 3*u*u*(p1[1]-p0[1]) + 6*u*t*(p2[1]-p1[1]) + 3*t*t*(p3[1]-p2[1])
    return dx, dy

def polyline_lengths(pts):
    L = [0.0]
    for i in range(1, len(pts)):
        x0,y0 = pts[i-1]; x1,y1 = pts[i]
        L.append(L[-1] + math.hypot(x1-x0, y1-y0))
    return L

def lerp(a, b, t): return a + (b - a) * t

def rgb_to_cairo(rgb):
    return tuple(c / 255.0 for c in rgb)


class CairoRenderer(Renderer):
    def __init__(self, render_options: CairoRenderOptions = None):
        if render_options is None:
            render_options = CairoRenderOptions()
        self.render_options = render_options


    def _render(self, sketch: Sketch, render_options: CairoRenderOptions) -> np.ndarray:
        """
        Renders the given sketch using Cairo's graphics system with the provided render options.

        Args:
            sketch (Sketch): The sketch to be rendered.
            render_options (CairoRenderOptions): The options controlling the rendering behavior.

        Returns:
            np.ndarray: The rendered sketch as a 3-channel NumPy array.
        """
        
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, render_options.canvas_size[0], render_options.canvas_size[1])
        ctx = cairo.Context(surface)
        ctx.set_antialias(cairo.ANTIALIAS_BEST)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_source_rgb(*rgb_to_cairo(render_options.background_color))  # Convert to range [0, 1]
        ctx.set_line_width(render_options.stroke_width)
        ctx.paint()

        # Rendering the sketch on the canvas
        for each_path in sketch.paths:
            for curve in each_path.curves:
                if isinstance(curve, Curve):
                    p0, p1, p2, p3 = curve.p_start, curve.p_crtl1, curve.p_crtl2, curve.p_end
                    ctx.set_source_rgb(*rgb_to_cairo(render_options.stroke_color))  # Convert to range [0, 1]
                    ctx.move_to(p0.x, p0.y)
                    ctx.curve_to(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y)
                    ctx.stroke()

        surface_data = surface.get_data()
        raster_image = np.copy(np.asarray(surface_data)).reshape(render_options.canvas_size[1], render_options.canvas_size[0], 4)[:, :, :3]

        temp = Image.fromarray(raster_image, 'RGB')
        temp.save(self.render_options.output_path, 'PNG')
        # return raster_image




