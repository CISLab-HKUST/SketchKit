import numpy as np


class Point:
    """A 2D point with x and y coordinates stored as 32-bit floats."""

    def __init__(
        self,
        x: float | np.float32,
        y: float | np.float32,
    ):
        self.x = np.float32(x)
        self.y = np.float32(y)


class Vertex(Point):
    """
    A vertex point with additional drawing attributes.
    Extends the base Point class with stroke-specific properties including pressure,
    thickness, color, and opacity for rendering purposes.
    Args:
        x (float | np.float32): The x-coordinate of the vertex.
        y (float | np.float32): The y-coordinate of the vertex.
        pressure (None | float | np.float32, optional): The pressure value applied
            at this vertex. Defaults to None.
        thickness (None | float | np.float32, optional): The stroke thickness at
            this vertex. Defaults to None.
        color (None | tuple[float, float, float] | np.ndarray, optional): RGB color
            values as a tuple or numpy array. Defaults to None.
        opacity (None | float | np.float32, optional): The opacity/transparency
            value (0.0 to 1.0). Defaults to None.
    Attributes:
        pressure (np.float32 | None): Normalized pressure value.
        thickness (np.float32 | None): Stroke thickness value.
        color (np.ndarray | None): RGB color array with float32 dtype.
        opacity (np.float32 | None): Opacity value.
    """

    def __init__(
        self,
        x: float | np.float32,
        y: float | np.float32,
        pressure: None | float | np.float32 = None,
        thickness: None | float | np.float32 = None,
        color: None | tuple[float, float, float] | np.ndarray = None,
        opacity: None | float | np.float32 = None,
    ):
        super().__init__(x, y)
        self.pressure = np.float32(pressure) if pressure is not None else None
        self.thickness = np.float32(thickness) if thickness is not None else None
        self.color = np.array(color, dtype=np.float32) if color is not None else None
        self.opacity = np.float32(opacity) if opacity is not None else None


class Stroke:
    """
    Represents a stroke in a sketch, defined by two vertices and two control points.
    A stroke is a fundamental drawing element that connects two vertices (start and end points)
    with optional control points for creating curved paths. It supports various visual properties
    such as color, pressure, thickness, and opacity that can be applied to both vertices.
    Attributes:
        p_start (Vertex): The starting vertex of the stroke.
        p_end (Vertex): The ending vertex of the stroke.
        p_crtl1 (Point): The first control point for curve definition.
        p_crtl2 (Point): The second control point for curve definition.
    Properties:
        color (list | None): RGB color values as a list of floats for both vertices.
            Can be set with a tuple, numpy array, or None.
        pressure (list | None): Pressure values for both vertices as floats.
            Can be set with a single float value or None.
        thickness (list | None): Thickness values for both vertices as floats.
            Can be set with a single float value or None.
        opacity (list | None): Opacity values for both vertices as floats.
            Can be set with a single float value or None.
    Note:
        When setting properties, the same value is applied to both start and end vertices.
        All numeric properties are automatically converted to numpy float32 format.
    """

    def __init__(
        self,
        p_start: Vertex,
        p_end: Vertex,
        p_crtl1: Point,
        p_crtl2: Point,
    ):
        self.p_start = p_start
        self.p_end = p_end
        self.p_crtl1 = p_crtl1
        self.p_crtl2 = p_crtl2

    def _get_vertices_attribute(self, attribute: str):
        return [
            getattr(self.p_start, attribute),
            getattr(self.p_end, attribute),
        ]

    def _set_vertices_attribute(self, attribute: str, value):
        setattr(self.p_start, attribute, value)
        setattr(self.p_end, attribute, value)

    @property
    def color(self):
        """
        Get the color attribute of all vertices in the sketch.
        
        Returns:
            numpy.ndarray or None: Array containing color values for each vertex,
                or None if no color attribute is set.
        """
        return self._get_vertices_attribute("color")

    @color.setter
    def color(self, value: None | tuple[float, float, float] | np.ndarray):
        color_value = np.array(value, dtype=np.float32) if value is not None else None
        self._set_vertices_attribute("color", color_value)

    @property
    def pressure(self):
        return self._get_vertices_attribute("pressure")

    @pressure.setter
    def pressure(self, value: None | float | np.float32):
        pressure_value = np.float32(value) if value is not None else None
        self._set_vertices_attribute("pressure", pressure_value)

    @property
    def thickness(self):
        return self._get_vertices_attribute("thickness")

    @thickness.setter
    def thickness(self, value: None | float | np.float32):
        thickness_value = np.float32(value) if value is not None else None
        self._set_vertices_attribute("thickness", thickness_value)

    @property
    def opacity(self):
        return self._get_vertices_attribute("opacity")

    @opacity.setter
    def opacity(self, value: None | float | np.float32):
        opacity_value = np.float32(value) if value is not None else None
        self._set_vertices_attribute("opacity", opacity_value)


class Path:
    """
    A Path represents a collection of strokes that can be manipulated as a group.
    This class provides a convenient way to manage multiple strokes and apply
    common operations to all strokes in the path simultaneously.
    Attributes:
        strokes (list[Stroke]): List of stroke objects that make up this path.
    Properties:
        stroke_num (int): Number of strokes in the path.
        color (list): List of colors for each stroke in the path.
        pressure (list): List of pressure values for each stroke in the path.
        thickness (list): List of thickness values for each stroke in the path.
        opacity (list): List of opacity values for each stroke in the path.
    Example:
        >>> path = Path([stroke1, stroke2, stroke3])
        >>> path.color = (1.0, 0.0, 0.0)  # Set all strokes to red
        >>> path.thickness = 2.5  # Set all strokes to thickness 2.5
        >>> print(path.stroke_num)  # Print number of strokes
    """

    def __init__(self, strokes: list[Stroke] | None = None):
        self.strokes = strokes if strokes else []

    @property
    def stroke_num(self):
        return len(self.strokes)

    def _get_strokes_attribute(self, attribute: str):
        return [getattr(stroke, attribute) for stroke in self.strokes]

    def _set_strokes_attribute(self, attribute: str, value):
        for stroke in self.strokes:
            setattr(stroke, attribute, value)

    @property
    def color(self):
        return self._get_strokes_attribute("color")

    @color.setter
    def color(self, value: None | tuple[float, float, float] | np.ndarray):
        color_value = np.array(value, dtype=np.float32) if value is not None else None
        self._set_strokes_attribute("color", color_value)

    @property
    def pressure(self):
        return self._get_strokes_attribute("pressure")

    @pressure.setter
    def pressure(self, value: None | float | np.float32):
        pressure_value = np.float32(value) if value is not None else None
        self._set_strokes_attribute("pressure", pressure_value)

    @property
    def thickness(self):
        return self._get_strokes_attribute("thickness")

    @thickness.setter
    def thickness(self, value: None | float | np.float32):
        thickness_value = np.float32(value) if value is not None else None
        self._set_strokes_attribute("thickness", thickness_value)

    @property
    def opacity(self):
        return self._get_strokes_attribute("opacity")

    @opacity.setter
    def opacity(self, value: None | float | np.float32):
        opacity_value = np.float32(value) if value is not None else None
        self._set_strokes_attribute("opacity", opacity_value)


class Sketch:
    """
    A class representing a sketch composed of multiple paths.
    A Sketch is a collection of Path objects that together form a complete drawing.
    It provides convenient methods to access and modify properties of all paths
    collectively, such as color, pressure, thickness, and opacity.
    Attributes:
        paths (list[Path]): List of Path objects that make up the sketch.
    Properties:
        path_num (int): Number of paths in the sketch.
        stroke_num (int): Total number of strokes across all paths in the sketch.
        color (list): List of colors for each path in the sketch.
        pressure (list): List of pressure values for each path in the sketch.
        thickness (list): List of thickness values for each path in the sketch.
        opacity (list): List of opacity values for each path in the sketch.
    Methods:
        __init__(paths=None): Initialize a sketch with optional list of paths.
    Example:
        >>> sketch = Sketch()
        >>> sketch = Sketch([path1, path2, path3])
        >>> sketch.color = (1.0, 0.0, 0.0)  # Set all paths to red
        >>> sketch.thickness = 2.5  # Set all paths to thickness 2.5
    """

    def __init__(self, paths: list[Path] | None = None):
        self.paths = paths if paths else []

    @property
    def path_num(self):
        return len(self.paths)

    @property
    def stroke_num(self):
        return sum(path.stroke_num for path in self.paths)
    
    def _get_paths_attribute(self, attribute: str):
        return [getattr(path, attribute) for path in self.paths]

    def _set_paths_attribute(self, attribute: str, value):
        for path in self.paths:
            setattr(path, attribute, value)

    @property
    def color(self):
        return self._get_paths_attribute("color")

    @color.setter
    def color(self, value: None | tuple[float, float, float] | np.ndarray):
        color_value = np.array(value, dtype=np.float32) if value is not None else None
        self._set_paths_attribute("color", color_value)

    @property
    def pressure(self):
        return self._get_paths_attribute("pressure")

    @pressure.setter
    def pressure(self, value: None | float | np.float32):
        pressure_value = np.float32(value) if value is not None else None
        self._set_paths_attribute("pressure", pressure_value)

    @property
    def thickness(self):
        return self._get_paths_attribute("thickness")

    @thickness.setter
    def thickness(self, value: None | float | np.float32):
        thickness_value = np.float32(value) if value is not None else None
        self._set_paths_attribute("thickness", thickness_value)

    @property
    def opacity(self):
        return self._get_paths_attribute("opacity")

    @opacity.setter
    def opacity(self, value: None | float | np.float32):
        opacity_value = np.float32(value) if value is not None else None
        self._set_paths_attribute("opacity", opacity_value)
