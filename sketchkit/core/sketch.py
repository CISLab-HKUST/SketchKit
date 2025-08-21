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


class Line(BaseStroke):
    """ A class representing a stroke with a polyline.
    A polyline connects two points.

    Args:
        p_start (Vertex): The starting point of the stroke.
        p_end (Vertex): The ending point of the stroke.
        thickness: The thickness/width of the stroke.
        pressure: The pressure applied when drawing the stroke.
        color: The color of the stroke.
        opacity: The opacity/transparency level of the stroke.
        texture: The texture pattern applied to the stroke.
    """

    def __init__(self, p_start: Vertex, p_end: Vertex,
                 thickness=None, pressure=None, color=None, opacity=None, texture=None):
        super(Line, self).__init__(p_start, p_end, thickness, pressure, color, opacity, texture)

    def to_cubic(self):
        start_x, start_y = self.p_start.x, self.p_start.y
        end_x, end_y = self.p_end.x, self.p_end.y
        ctrl1_x = 2.0 / 3.0 * start_x + 1.0 / 3.0 * end_x
        ctrl1_y = 2.0 / 3.0 * start_y + 1.0 / 3.0 * end_y
        ctrl2_x = 1.0 / 3.0 * start_x + 2.0 / 3.0 * end_x
        ctrl2_y = 1.0 / 3.0 * start_y + 2.0 / 3.0 * end_y

        p_crtl1 = Vertex(ctrl1_x, ctrl1_y)
        p_crtl2 = Vertex(ctrl2_x, ctrl2_y)

        return CubicBezierCurve(self.p_start, p_crtl1, p_crtl2, self.p_end,
                                self.thickness, self.pressure, self.color, self.opacity, self.texture)


class QuadraticBezierCurve(BaseStroke):
    """ A class representing a stroke with a quadratic Bézier curve.
    A quadratic Bézier curve is made up of a starting point, a control point, and an end point.

    Args:
        p_start (Vertex): The starting point of the stroke.
        p_crtl (Vertex): The control point of the stroke.
        p_end (Vertex): The ending point of the stroke.
        thickness: The thickness/width of the stroke.
        pressure: The pressure applied when drawing the stroke.
        color: The color of the stroke.
        opacity: The opacity/transparency level of the stroke.
        texture: The texture pattern applied to the stroke.

    Attributes:
        p_crtl (Vertex): The control point of the stroke.
    """

    def __init__(self, p_start: Vertex, p_crtl: Vertex, p_end: Vertex,
                 thickness=None, pressure=None, color=None, opacity=None, texture=None):
        super(QuadraticBezierCurve, self).__init__(p_start, p_end, thickness, pressure, color, opacity, texture)
        self.p_crtl = p_crtl


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
        stroke_type (str): "line", "qbc", or "cbc".
        stroke_list (list): A list of strokes, each stroke in shape (N_points, 2), with N_points = 2/3/4 for "line", "qbc", and "cbc", respectively.
        thickness_list: In size (N_strokes). Set to None by default.
        pressure_list: In size (N_strokes). Set to None by default.
        color_list: In shape (N_strokes, 3). Set to None by default.
        opacity_list: In size (N_strokes). Set to None by default.
        texture_list: In size (N_strokes). Set to None by default.

    Attributes:
        stroke_type: "line", "qbc", or "cbc".
        stroke_list (list): A list of strokes, each stroke in class Line, QuadraticBezierCurve, or CubicBezierCurve.
    """

    def __init__(self, stroke_type: str, stroke_list: list,
                 thickness_list=None, pressure_list=None,
                 color_list=None, opacity_list=None, texture_list=None):
        self.stroke_type = stroke_type
        self.stroke_list = []

        stroke_num = len(stroke_list)
        for stroke_i in range(stroke_num):
            stroke = stroke_list[stroke_i]
            stroke_thickness = thickness_list[stroke_i] if thickness_list is not None else None
            stroke_pressure = pressure_list[stroke_i] if pressure_list is not None else None
            stroke_color = color_list[stroke_i] if color_list is not None else None
            stroke_opacity = opacity_list[stroke_i] if opacity_list is not None else None
            stroke_texture = texture_list[stroke_i] if texture_list is not None else None

            if stroke_type == "line":
                p_start = Vertex(stroke[0][0], stroke[0][1])
                p_end = Vertex(stroke[1][0], stroke[1][1])
                self.stroke_list.append(Line(p_start, p_end,
                                             stroke_thickness, stroke_pressure, stroke_color, stroke_opacity, stroke_texture))
            elif stroke_type == "qbc":
                p_start = Vertex(stroke[0][0], stroke[0][1])
                p_crtl = Vertex(stroke[1][0], stroke[1][1])
                p_end = Vertex(stroke[2][0], stroke[2][1])
                self.stroke_list.append(QuadraticBezierCurve(p_start, p_crtl, p_end,
                                                             stroke_thickness, stroke_pressure, stroke_color, stroke_opacity, stroke_texture))
            elif stroke_type == "cbc":
                p_start = Vertex(stroke[0][0], stroke[0][1])
                p_crtl1 = Vertex(stroke[1][0], stroke[1][1])
                p_crtl2 = Vertex(stroke[2][0], stroke[2][1])
                p_end = Vertex(stroke[3][0], stroke[3][1])
                self.stroke_list.append(CubicBezierCurve(p_start, p_crtl1, p_crtl2, p_end,
                                                         stroke_thickness, stroke_pressure, stroke_color, stroke_opacity, stroke_texture))
            else:
                raise Exception('Unknown stroke_type:', stroke_type)

    def get_length(self):
        return len(self.stroke_list)

    def line_to_cubic(self):
        if self.stroke_type == "line":
            stroke_list_convert = [line_item.to_cubic() for line_item in self.stroke_list]
            self.stroke_list = stroke_list_convert
            self.stroke_type = "cbc"
        elif self.stroke_type == "qbc":
            print("Quadratic Bezier curve is not support for conversion.")
        elif self.stroke_type == "cbc":
            print("The strokes are in cubic Bezier curves already. No need for conversion.")
        else:
            raise Exception('Unknown stroke_type:', self.stroke_type)


class SketchVG:
    """ The universal sketch class.
    A sketch contains a list of paths. A path contains a list of strokes.

    Args:
        stroke_type (str): "line", "qbc", or "cbc". For Line, QuadraticBezierCurve, and CubicBezierCurve, respectively.
        path_list (list): A list of paths, each path in shape (N_strokes, N_points, 2), with N_points = 2/3/4 for "line", "qbc", and "cbc", respectively.
        stroke_thickness_list: A list of stroke thickness, each in size (N_strokes). Set to None by default.
        stroke_pressure_list: A list of stroke pressure, each in size (N_strokes). Set to None by default.
        stroke_color_list: A list of stroke colors, each in shape (N_strokes, 3). Set to None by default.
        stroke_opacity_list: A list of stroke opacity, each in size (N_strokes). Set to None by default.
        stroke_texture_list: A list of stroke texture, each in size (N_strokes). Set to None by default.

    Attributes:
        stroke_type: "line", "qbc", or "cbc".
        path_list (list): A list of strokes, each stroke in class Line, QuadraticBezierCurve, or CubicBezierCurve.
    """

    def __init__(self, stroke_type: str, path_list: list,
                 stroke_thickness_list=None, stroke_pressure_list=None,
                 stroke_color_list=None, stroke_opacity_list=None, stroke_texture_list=None):
        self.stroke_type = stroke_type
        self.path_list = []

        for path_i in range(len(path_list)):
            path = path_list[path_i]
            path_thickness = stroke_thickness_list[path_i] if stroke_thickness_list is not None else None
            path_pressure = stroke_pressure_list[path_i] if stroke_pressure_list is not None else None
            path_color = stroke_color_list[path_i] if stroke_color_list is not None else None
            path_opacity = stroke_opacity_list[path_i] if stroke_opacity_list is not None else None
            path_texture = stroke_texture_list[path_i] if stroke_texture_list is not None else None

            self.path_list.append(Path(stroke_type, path,
                                       path_thickness, path_pressure,
                                       path_color, path_opacity, path_texture))

    def get_path_num(self):
        return len(self.path_list)

    def get_stroke_num(self):
        path_num = self.get_path_num()
        stroke_num = 0
        for path_i in range(path_num):
            stroke_num += self.path_list[path_i].get_length()
        return stroke_num

    def line_to_cubic(self):
        if self.stroke_type == "line":
            path_num = self.get_path_num()
            for path_i in range(path_num):
                self.path_list[path_i].line_to_cubic()
            self.stroke_type = "cbc"
        elif self.stroke_type == "qbc":
            print("Quadratic Bezier curve is not support for conversion.")
        elif self.stroke_type == "cbc":
            print("The strokes are in cubic Bezier curves already. No need for conversion.")
        else:
            raise Exception('Unknown stroke_type:', self.stroke_type)

    def rasterize(self):
        # TODO:
        print()

    def vectorize(self):
        # TODO:
        print()

    def save(self, save_path):
        # TODO:
        print()
