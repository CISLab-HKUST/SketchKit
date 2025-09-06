from sketchkit.core.sketch import Sketch, Curve
import numpy as np
import cairocffi as cairo

def generate_random_colors(n):
    """
    Generate n distinct random colors.

    Args:
        n (int): The number of distinct colors to generate.

    Returns:
        list: A list of n tuples, each representing an RGB color with values in the range [0, 1].
    """
    colors = []
    for _ in range(n):
        color = (np.random.rand(), np.random.rand(), np.random.rand())
        colors.append(color)
    return colors
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

    def render(self, sketch: Sketch, random_color=True):
        """
        Renders the given sketch onto a canvas using Cairo's drawing functions.

        This method initializes a Cairo surface, sets drawing properties like antialiasing, line width,
        and color, and then iterates over the provided `sketch` object to draw its paths and curves.

        Args:
            sketch (Sketch): A Sketch object containing paths and strokes to be rendered.
            random_color (bool, optional): If True, each path will be drawn in a random color. Default is True.

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
        ctx.set_line_width(3)
        ctx.paint()
        
        if random_color:
            colors = generate_random_colors(len(sketch.paths))
        seq_i = 0
        for j, each_path in enumerate(sketch.paths):
            for curve in each_path.curves:
                if isinstance(curve, Curve):
                    p0, p1, p2, p3 = curve.p_start, curve.p_crtl1, curve.p_crtl2, curve.p_end
                    x0, y0 = p0.x, p0.y
                    x1, y1 = p1.x, p1.y
                    x2, y2 = p2.x, p2.y
                    x3, y3 = p3.x, p3.y
                    # ctx.set_source_rgb(colors[seq_i][0], colors[seq_i][1], colors[seq_i][2])
                    if random_color:
                        ctx.set_source_rgb(colors[j][0], colors[j][1], colors[j][2])
                    else:
                        ctx.set_source_rgb(0,0,0)
                    # seq_i += 1
                    ctx.move_to(x0, y0)
                    ctx.curve_to(x1, y1, x2, y2, x3, y3)
                    ctx.stroke()
                else:
                    print("curve is something else")
        surface_data = surface.get_data()
        raster_image = np.copy(np.asarray(surface_data)).reshape(self.canvas_size, self.canvas_size, 4)[:, :, :3]
        return raster_image


