from sketchkit.core.sketch import Sketch, Curve
import numpy as np
import cairocffi as cairo
import math
from PIL import Image
import random

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

    def render(self, sketch: Sketch, width: int = 3, fit_canvas: bool = False, texture: str = None):
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
    
    def render_texture_along_bezier(
        self,
        texture_img,  # PIL.Image
        control_points,  # [(x0,y0),(x1,y1),(x2,y2),(x3,y3)]
        canvas,
        stroke_width=3,
        spacing=0.001,   # <1 overlaps; 1 butts; >1 gaps
        samples=1000,  # more samples => smoother arc-length stepping
        background=(0,0,0,0),  # RGBA or RGB
        antialias=True,
    ):
        """
        Render a texture along a cubic Bézier curve by stamping a resampled tile 
        along the curve path.

        Parameters
        ----------
        texture_img : PIL.Image
            Input texture image. It will be converted to RGBA if needed.
        control_points : list of tuple
            Four control points [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] 
            defining a cubic Bézier curve.
        canvas : PIL.Image
            Target canvas image onto which the textured curve is rendered.
        stroke_width : int, optional
            Width of the rendered stroke in pixels. Default is 3.
        spacing : float, optional
            Factor controlling spacing between texture tiles along the curve.
            - <1 → overlap  
            - =1 → touching (butting)  
            - >1 → gaps  
            Default is 0.001.
        samples : int, optional
            Number of samples along the curve used for arc-length approximation. 
            Higher values yield smoother placement. Default is 1000.
        background : tuple, optional
            Background color for the canvas in RGB or RGBA. Default is transparent (0,0,0,0).
        antialias : bool, optional
            Whether to use bicubic resampling (True) or nearest-neighbor (False). 
            Default is True.

        Returns
        -------
        PIL.Image
            The updated canvas with the textured Bézier stroke rendered on top.

        """
        
        assert len(control_points) == 4, "Provide 4 control points for a cubic Bézier."
        p0,p1,p2,p3 = control_points
        scale = 1.0
        p0 = (float(p0[0]) * scale, float(p0[1]) * scale)
        p1 = (float(p1[0]) * scale, float(p1[1]) * scale)
        p2 = (float(p2[0]) * scale, float(p2[1]) * scale)
        p3 = (float(p3[0]) * scale, float(p3[1]) * scale)

        # Sample the curve densely
        ts = [i/(samples-1) for i in range(samples)]
        pts = [bezier_point(p0,p1,p2,p3,t) for t in ts]
        lens = polyline_lengths(pts)

        total_len = lens[-1] if lens else 0.0

        # Prepare texture: ensure RGBA and orient width=along-path, height=thickness
        tex = texture_img.convert("RGBA") #.resize((64, 64), resample=Image.BICUBIC)
        tw, th = tex.size
        if th > tw:
            tex = tex.transpose(Image.ROTATE_90)
            tw, th = tex.size
        resample = Image.BICUBIC if antialias else Image.NEAREST
        base_tile = tex.resize((stroke_width, stroke_width), resample=resample)
        tile_len = base_tile.width
        step = max(1, int(tile_len * spacing))

        mode = "RGBA" if len(background) == 4 else "RGB"

        def paste_center_rotated(img, cx, cy, angle_deg):
            r = img.rotate(angle_deg, resample=resample, expand=True)
            rx, ry = int(cx - r.width/2), int(cy - r.height/2)
            if mode == "RGBA":
                canvas.alpha_composite(r, (rx, ry))
            else:
                canvas.paste(r, (rx, ry), r)

        # Map a desired arc-length s to point & tangent by binary search on lens
        import bisect
        def pose_at_s(s):
            s = min(max(0.0, s), total_len)
            i = bisect.bisect_left(lens, s)
            if i <= 0: t = 0.0
            elif i >= len(lens): t = 1.0
            else:
                s0, s1 = lens[i-1], lens[i]
                # local fraction between ts[i-1] and ts[i]
                frac = 0.0 if s1==s0 else (s - s0) / (s1 - s0)
                t = lerp(ts[i-1], ts[i], frac)
            x,y = bezier_point(p0,p1,p2,p3,t)
            dx,dy = bezier_tangent(p0,p1,p2,p3,t)
            # angle = math.degrees(math.atan2(dy, dx))
            angle = random.uniform(-180, 180)
            return x, y, angle

        # Stamp along the curve
        s = 0.0

        while s + step <= total_len - 1:
            x,y,ang = pose_at_s(s)
            paste_center_rotated(base_tile, x, y, ang)
            s += step

        # compute where the next tile would start (its leading edge at s + tile_len/2)
        next_center = s
        if next_center < total_len - 1:
            part_len = int(min(tile_len, total_len - next_center + step))
            if part_len > 1:
                partial = base_tile.crop((0, 0, part_len, base_tile.height))
                x,y,ang = pose_at_s(min(next_center, total_len - 1e-6))
                paste_center_rotated(partial, x, y, ang)

        return canvas

    def render_with_texture(self, sketch: Sketch, width: int = 4, texture: str = None, samples = 1000, spacing=0.2, antialias=True):
        
        """
        Render a full sketch composed of Bézier curves using a given texture.

        Parameters
        ----------
        sketch : Sketch
            A Sketch object containing one or more paths, each with curves. 
            Curves must be instances of `Curve` with four Bézier control points 
            (start, control1, control2, end).
        width : int, optional
            Stroke width in pixels. Default is 4.
        texture : str, optional
            Path to the texture image file used for rendering strokes.
        samples : int, optional
            Number of curve samples for arc-length approximation. Default is 1000.
        spacing : float, optional
            Tile spacing factor passed to `render_texture_along_bezier`. 
            Default is 0.2.
        antialias : bool, optional
            Whether to enable bicubic resampling for smoother rendering. 
            Default is True.

        Returns
        -------
        PIL.Image
            A canvas image of size `(self.canvas_size, self.canvas_size)` with 
            all sketch paths rendered using the texture.

        """
        # Prepare texture: ensure RGBA and orient width=along-path, height=thickness
        tex = Image.open(texture)
        mode = "RGBA"
        canvas = Image.new(mode, (self.canvas_size, self.canvas_size), 'white')
            
        for each_path in sketch.paths:
            for curve in each_path.curves:
                if isinstance(curve, Curve):
                    p0, p1, p2, p3 = curve.p_start, curve.p_crtl1, curve.p_crtl2, curve.p_end
                    control_points = [(p0.x, p0.y), (p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]

                    canvas = self.render_texture_along_bezier(texture_img=tex,control_points=control_points,canvas=canvas,stroke_width=width,spacing=spacing,samples=samples,antialias=antialias)

                else:
                    print("curve is something else")
                
        return canvas


