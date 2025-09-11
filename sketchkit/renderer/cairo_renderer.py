from sketchkit.core.sketch import Sketch, Curve
import numpy as np
import cairocffi as cairo

class CairoRenderer:
    """
    A class for rendering sketches using Cairo graphics.

    This class provides functionality to render a sketch onto a canvas, where the sketch
    is composed of paths with curves that are drawn using Cairo's vector graphics system.
    
    Attributes:
        canvas_size (int): The size (width and height) of the canvas in pixels. Default is 512.
        canvas_color (tuple): The background color of the canvas in RGB format (0.0 to 1.0). Default is white (1, 1, 1).
    """
    def __init__(self, canvas_size=512, canvas_color=(1, 1, 1)):
        """
        Initializes the CairoRenderer with a given canvas size and color.

        Args:
            canvas_size (int, optional): The size of the canvas (both width and height) in pixels. Default is 512.
            canvas_color (tuple, optional): The background color of the canvas as an (R, G, B) tuple. Default is (1, 1, 1) (white).
        """
        self.canvas_size = canvas_size
        self.canvas_color = canvas_color

    def render(self, sketch: Sketch, width: int = 3, fit_canvas: bool = False):
        """
        Renders the given sketch onto a canvas using Cairo's drawing functions.

        This method initializes a Cairo surface, sets drawing properties like antialiasing, line width,
        and color, and then iterates over the provided `sketch` object to draw its paths and curves.

        Args:
            sketch (Sketch): A Sketch object containing paths and strokes to be rendered.
            width (int, optional): The width of the lines used to draw the sketch. Default is 3.
            fit_canvas (bool, optional): Whether to scale and center the sketch to fit the canvas. Default is False.

        Returns:
            np.ndarray: A 3-channel image (height x width x 3) representing the rendered sketch, as a NumPy array.
            
        Raises:
            ValueError: If any stroke in the sketch is not of type `Stroke`.
        """
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.canvas_size, self.canvas_size)
        ctx = cairo.Context(surface)
        ctx.set_antialias(cairo.ANTIALIAS_BEST)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_source_rgb(*self.canvas_color)
        ctx.set_line_width(width)
        ctx.paint()

        # 适配画布
        if fit_canvas:
            points = []
            for each_path in sketch.paths:
                for curve in each_path.curves:
                    if isinstance(curve, Curve):
                        points.extend([
                            (curve.p_start.x, curve.p_start.y),
                            (curve.p_crtl1.x, curve.p_crtl1.y),
                            (curve.p_crtl2.x, curve.p_crtl2.y),
                            (curve.p_end.x, curve.p_end.y),
                        ])
            if points:
                xs, ys = zip(*points)
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                padding = 0.05
                width, height = max_x - min_x, max_y - min_y
                if width == 0: width = 1
                if height == 0: height = 1
                scale = (1 - 2 * padding) * self.canvas_size / max(width, height)
                offset_x = (self.canvas_size - scale * width) / 2 - scale * min_x
                offset_y = (self.canvas_size - scale * height) / 2 - scale * min_y
            else:
                scale = 1
                offset_x = 0
                offset_y = 0
            def tx(x): return x * scale + offset_x
            def ty(y): return y * scale + offset_y
        else:
            tx = lambda x: x
            ty = lambda y: y

        seq_i = 0
        for j, each_path in enumerate(sketch.paths):
            for curve in each_path.curves:
                if isinstance(curve, Curve):
                    p0, p1, p2, p3 = curve.p_start, curve.p_crtl1, curve.p_crtl2, curve.p_end
                    ctx.set_source_rgb(0, 0, 0)
                    ctx.move_to(tx(p0.x), ty(p0.y))
                    ctx.curve_to(tx(p1.x), ty(p1.y), tx(p2.x), ty(p2.y), tx(p3.x), ty(p3.y))
                    ctx.stroke()
                else:
                    print("curve is something else")
        surface_data = surface.get_data()
        raster_image = np.copy(np.asarray(surface_data)).reshape(self.canvas_size, self.canvas_size, 4)[:, :, :3]
        return raster_image


