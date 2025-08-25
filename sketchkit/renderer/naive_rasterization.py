from sketchkit.core.sketch import SketchVG, CubicBezierCurve
import numpy as np
import cairocffi as cairo

class Renderer:
    def __init__(self, canvas_size=512, canvas_color=(1, 1, 1)):
        self.canvas_size = canvas_size
        self.canvas_color = canvas_color

    def render(self, sketch: SketchVG):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.canvas_size, self.canvas_size)
        ctx = cairo.Context(surface)
        ctx.set_antialias(cairo.ANTIALIAS_BEST)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_width(10)
        ctx.set_source_rgb(*self.canvas_color)
        ctx.paint()
        

        seq_i = 0
        for j, each_path in enumerate(sketch.path_list):
            for curve in each_path.stroke_list:  # (N, 2) for polyline, (N, 4, 2) for bezier
                if isinstance(curve, CubicBezierCurve):
                    print("curve is a CubicBezierCurve")
                    p0, p1, p2, p3 = curve.p_start, curve.p_crtl1, curve.p_crtl2, curve.p_end
                    print(p0)
                    x0, y0 = p0.x, p0.y
                    x1, y1 = p1.x, p1.y
                    x2, y2 = p2.x, p2.y
                    x3, y3 = p3.x, p3.y
                    # ctx.set_source_rgb(colors[seq_i][0], colors[seq_i][1], colors[seq_i][2])
                    ctx.set_source_rgb(0, 0, 0)
                    # seq_i += 1
                    ctx.move_to(x0, y0)
                    ctx.curve_to(x1, y1, x2, y2, x3, y3)
                    ctx.stroke()
                else:
                    print("curve is something else")
        surface_data = surface.get_data()
        raster_image = np.copy(np.asarray(surface_data)).reshape(self.canvas_size, self.canvas_size, 4)[:, :, :3]
        return raster_image


