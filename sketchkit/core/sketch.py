import numpy as np

class Vertex:
    """
    Represents a 2D vertex/point with x and y coordinates.
    A Vertex is a fundamental geometric primitive that defines a point in 2D space.
    It serves as a building block for more complex geometric shapes and sketching operations.

    Args:
        x : The x-coordinate position.
        y : The y-coordinate position.

    Attributes:
        x (np.float32): The x-coordinate of the vertex.
        y (np.float32): The y-coordinate of the vertex.

    Example:
        >>> vertex = Vertex(10.0, 20.0)
        >>> print(vertex.x, vertex.y)
        10.0 20.0
    """

    def __init__(self, x: float|np.float32, y: float|np.float32):
        self.x = np.float32(x)
        self.y = np.float32(y)


class BaseStroke:
    """ A base class representing a stroke in a sketch.
    A stroke is a fundamental drawing element that connects two points with specific
    visual properties like thickness, pressure, color, opacity, and texture.

    Args:
        p_start (Vertex): The starting point of the stroke.
        p_end (Vertex): The ending point of the stroke.
        thickness: The thickness/width of the stroke.
        pressure: The pressure applied when drawing the stroke.
        color: The color of the stroke.
        opacity: The opacity/transparency level of the stroke.
        texture: The texture pattern applied to the stroke.

    Attributes:
        p_start (Vertex): The starting vertex of the stroke.
        p_end (Vertex): The ending vertex of the stroke.
        thickness: The stroke thickness.
        pressure: The applied pressure value.
        color: The stroke color.
        opacity: The stroke opacity level.
        texture: The stroke texture.
    """

    def __init__(self, p_start: Vertex, p_end: Vertex, thickness, pressure, color, opacity, texture):
        self.p_start = p_start
        self.p_end = p_end
        self.thickness = thickness
        self.pressure = pressure
        self.color = color
        self.opacity = opacity
        self.texture = texture


# class Line(BaseStroke):
#     """ A class representing a stroke with a polyline.
#     A polyline connects two points.
#
#     Args:
#         p_start (Vertex): The starting point of the stroke.
#         p_end (Vertex): The ending point of the stroke.
#         thickness: The thickness/width of the stroke.
#         pressure: The pressure applied when drawing the stroke.
#         color: The color of the stroke.
#         opacity: The opacity/transparency level of the stroke.
#         texture: The texture pattern applied to the stroke.
#     """
#
#     def __init__(self, p_start: Vertex, p_end: Vertex,
#                  thickness=None, pressure=None, color=None, opacity=None, texture=None):
#         super(Line, self).__init__(p_start, p_end, thickness, pressure, color, opacity, texture)


# class QuadraticBezierCurve(BaseStroke):
#     """ A class representing a stroke with a quadratic Bézier curve.
#     A quadratic Bézier curve is made up of a starting point, a control point, and an end point.
#
#     Args:
#         p_start (Vertex): The starting point of the stroke.
#         p_crtl (Vertex): The control point of the stroke.
#         p_end (Vertex): The ending point of the stroke.
#         thickness: The thickness/width of the stroke.
#         pressure: The pressure applied when drawing the stroke.
#         color: The color of the stroke.
#         opacity: The opacity/transparency level of the stroke.
#         texture: The texture pattern applied to the stroke.
#
#     Attributes:
#         p_crtl (Vertex): The control point of the stroke.
#     """
#
#     def __init__(self, p_start: Vertex, p_crtl: Vertex, p_end: Vertex,
#                  thickness=None, pressure=None, color=None, opacity=None, texture=None):
#         super(QuadraticBezierCurve, self).__init__(p_start, p_end, thickness, pressure, color, opacity, texture)
#         self.p_crtl = p_crtl


class CubicBezierCurve(BaseStroke):
    """ A class representing a stroke with a cubic Bézier curve.
    A cubic Bézier curve is made up of a starting point, two control points, and an end point.

    Args:
        p_start (Vertex): The starting point of the stroke.
        p_crtl1 (Vertex): The first control point of the stroke.
        p_crtl2 (Vertex): The second control point of the stroke.
        p_end (Vertex): The ending point of the stroke.
        thickness: The thickness/width of the stroke.
        pressure: The pressure applied when drawing the stroke.
        color: The color of the stroke.
        opacity: The opacity/transparency level of the stroke.
        texture: The texture pattern applied to the stroke.

    Attributes:
        p_crtl1 (Vertex): The first control point of the stroke.
        p_crtl2 (Vertex): The second control point of the stroke.
    """

    def __init__(self, p_start: Vertex, p_crtl1: Vertex, p_crtl2: Vertex, p_end: Vertex,
                 thickness=None, pressure=None, color=None, opacity=None, texture=None):
        super(CubicBezierCurve, self).__init__(p_start, p_end, thickness, pressure, color, opacity, texture)
        self.p_crtl1 = p_crtl1
        self.p_crtl2 = p_crtl2


class Path:
    """ The class for a path.
    A path contains a list of strokes.

    Args:
        stroke_list (list): A list of strokes, each stroke in shape (4, 2) indicating a cubic Bézier curve with four control points.
        thickness_list: In size (N_strokes). Set to None by default.
        pressure_list: In size (N_strokes). Set to None by default.
        color_list: In shape (N_strokes, 3). Set to None by default.
        opacity_list: In size (N_strokes). Set to None by default.
        texture_list: In size (N_strokes). Set to None by default.

    Attributes:
        stroke_list (list): A list of strokes, each stroke is a CubicBezierCurve.
    """

    def __init__(self, stroke_list: list,
                 thickness_list=None, pressure_list=None,
                 color_list=None, opacity_list=None, texture_list=None):
        self.stroke_list = []

        stroke_num = len(stroke_list)
        for stroke_i in range(stroke_num):
            stroke = stroke_list[stroke_i]
            stroke_thickness = thickness_list[stroke_i] if thickness_list is not None else None
            stroke_pressure = pressure_list[stroke_i] if pressure_list is not None else None
            stroke_color = color_list[stroke_i] if color_list is not None else None
            stroke_opacity = opacity_list[stroke_i] if opacity_list is not None else None
            stroke_texture = texture_list[stroke_i] if texture_list is not None else None

            p_start = Vertex(stroke[0][0], stroke[0][1])
            p_crtl1 = Vertex(stroke[1][0], stroke[1][1])
            p_crtl2 = Vertex(stroke[2][0], stroke[2][1])
            p_end = Vertex(stroke[3][0], stroke[3][1])
            self.stroke_list.append(CubicBezierCurve(p_start, p_crtl1, p_crtl2, p_end,
                                                     stroke_thickness, stroke_pressure, stroke_color, stroke_opacity, stroke_texture))

    def get_length(self):
        return len(self.stroke_list)


class SketchVG:
    """ The universal sketch class.
    A sketch contains a list of paths. Each path contains a list of strokes. Each stroke is a cubic Bézier curve with four control points.

    Args:
        path_list (list): A list of paths, each path in shape (N_strokes, 4, 2).
        stroke_thickness_list: A list of stroke thickness, each in size (N_strokes). Set to None by default.
        stroke_pressure_list: A list of stroke pressure, each in size (N_strokes). Set to None by default.
        stroke_color_list: A list of stroke colors, each in shape (N_strokes, 3). Set to None by default.
        stroke_opacity_list: A list of stroke opacity, each in size (N_strokes). Set to None by default.
        stroke_texture_list: A list of stroke texture, each in size (N_strokes). Set to None by default.

    Attributes:
        path_list (list): A list of paths, each path is a list of strokes.
    """

    def __init__(self, path_list: list,
                 stroke_thickness_list=None, stroke_pressure_list=None,
                 stroke_color_list=None, stroke_opacity_list=None, stroke_texture_list=None):
        self.path_list = []

        for path_i in range(len(path_list)):
            path = path_list[path_i]
            path_thickness = stroke_thickness_list[path_i] if stroke_thickness_list is not None else None
            path_pressure = stroke_pressure_list[path_i] if stroke_pressure_list is not None else None
            path_color = stroke_color_list[path_i] if stroke_color_list is not None else None
            path_opacity = stroke_opacity_list[path_i] if stroke_opacity_list is not None else None
            path_texture = stroke_texture_list[path_i] if stroke_texture_list is not None else None

            self.path_list.append(Path(path, path_thickness, path_pressure,
                                       path_color, path_opacity, path_texture))

    def get_path_num(self):
        return len(self.path_list)

    def get_stroke_num(self):
        path_num = self.get_path_num()
        stroke_num = 0
        for path_i in range(path_num):
            stroke_num += self.path_list[path_i].get_length()
        return stroke_num

    def rasterize(self):
        # TODO:
        print()

    def vectorize(self):
        # TODO:
        print()

    def save(self, save_path):
        # TODO:
        print()
