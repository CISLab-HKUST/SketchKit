import numpy as np


class Point:
    """2D point.

    Represents a 2D point with x and y coordinates stored as 32-bit floats.

    Attributes:
        x (np.float32): X coordinate.
        y (np.float32): Y coordinate.
    """

    def __init__(
        self,
        x: float | np.float32,
        y: float | np.float32,
    ):
        self.x = np.float32(x)
        self.y = np.float32(y)


class Vertex(Point):
    """Vertex point with drawing attributes.

    Extends Point with curve-specific properties: pressure, thickness, color,
    and opacity for rendering.

    Attributes:
        pressure (np.float32 | None): Normalized pressure value.
        thickness (np.float32 | None): Stroke thickness value.
        color (np.ndarray | None): RGB color array with float32 dtype.
        opacity (np.float32 | None): Opacity value (0.0-1.0).
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
        """Initialize a Vertex.

        Args:
            x (float | np.float32): X coordinate.
            y (float | np.float32): Y coordinate.
            pressure (float | np.float32 | None, optional): Pressure at this
                vertex. Defaults to None.
            thickness (float | np.float32 | None, optional): Stroke thickness at
                this vertex. Defaults to None.
            color (tuple[float, float, float] | np.ndarray | None, optional):
                RGB color as tuple or numpy array. Defaults to None.
            opacity (float | np.float32 | None, optional): Opacity (0.0-1.0).
                Defaults to None.
        """
        super().__init__(x, y)
        self.pressure = np.float32(pressure) if pressure is not None else None
        self.thickness = np.float32(thickness) if thickness is not None else None
        self.color = np.array(color, dtype=np.float32) if color is not None else None
        self.opacity = np.float32(opacity) if opacity is not None else None


class Curve:
    """Curve defined by two vertices and two control points.

    Connects two vertices (start and end) with optional control points for a
    curved path. Visual properties (color, pressure, thickness, opacity) are
    applied to both vertices together.

    Attributes:
        p_start (Vertex): Starting vertex.
        p_end (Vertex): Ending vertex.
        p_crtl1 (Point): First control point for curve definition.
        p_crtl2 (Point): Second control point for curve definition.

    Notes:
        When setting properties, the same value is applied to both start and end
        vertices. Numeric properties are converted to np.float32 automatically.
    """

    def __init__(
        self,
        p_start: Vertex,
        p_end: Vertex,
        p_crtl1: Point,
        p_crtl2: Point,
    ):
        """Initialize a Curve.

        Args:
            p_start (Vertex): Starting vertex.
            p_end (Vertex): Ending vertex.
            p_crtl1 (Point): First control point.
            p_crtl2 (Point): Second control point.
        """
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
        """Colors of the start and end vertices.

        Returns:
            list[np.ndarray | None]: A two-element list of RGB colors for the
            start and end vertices (each as float32 arrays) or None.
        """
        return self._get_vertices_attribute("color")

    @color.setter
    def color(self, value: None | tuple[float, float, float] | np.ndarray):
        """Set color for both vertices.

        Args:
            value (tuple[float, float, float] | np.ndarray | None): RGB color as
                tuple/ndarray, or None to clear.
        """
        color_value = np.array(value, dtype=np.float32) if value is not None else None
        self._set_vertices_attribute("color", color_value)

    @property
    def pressure(self):
        """Pressures of the start and end vertices.

        Returns:
            list[np.float32 | None]: Two-element list of pressures or None.
        """
        return self._get_vertices_attribute("pressure")

    @pressure.setter
    def pressure(self, value: None | float | np.float32):
        """Set pressure for both vertices.

        Args:
            value (float | np.float32 | None): Pressure value or None to clear.
        """
        pressure_value = np.float32(value) if value is not None else None
        self._set_vertices_attribute("pressure", pressure_value)

    @property
    def thickness(self):
        """Thicknesses of the start and end vertices.

        Returns:
            list[np.float32 | None]: Two-element list of thicknesses or None.
        """
        return self._get_vertices_attribute("thickness")

    @thickness.setter
    def thickness(self, value: None | float | np.float32):
        """Set thickness for both vertices.

        Args:
            value (float | np.float32 | None): Thickness value or None to clear.
        """
        thickness_value = np.float32(value) if value is not None else None
        self._set_vertices_attribute("thickness", thickness_value)

    @property
    def opacity(self):
        """Opacities of the start and end vertices.

        Returns:
            list[np.float32 | None]: Two-element list of opacities or None.
        """
        return self._get_vertices_attribute("opacity")

    @opacity.setter
    def opacity(self, value: None | float | np.float32):
        """Set opacity for both vertices.

        Args:
            value (float | np.float32 | None): Opacity (0.0–1.0) or None to clear.
        """
        opacity_value = np.float32(value) if value is not None else None
        self._set_vertices_attribute("opacity", opacity_value)


class Path:
    """Collection of curves.

    Manages multiple curves and applies common operations to all of them.

    Attributes:
        curves (list[Curve]): Curves in this path.

    Examples:
        >>> path = Path([curve1, curve2, curve3])
        >>> path.color = (1.0, 0.0, 0.0)  # Set all curves to red
        >>> path.thickness = 2.5          # Set all curves to thickness 2.5
        >>> path.curve_num
        3
    """

    def __init__(self, curves: list[Curve] | None = None):
        """Initialize a Path.

        Args:
            curves (list[Curve] | None, optional): Initial curves. Defaults to None.
        """
        self.curves = curves if curves else []

    @property
    def curve_num(self):
        """Number of curves in the path.

        Returns:
            int: Count of curves.
        """
        return len(self.curves)

    def _get_curves_attribute(self, attribute: str):
        return [getattr(curve, attribute) for curve in self.curves]

    def _set_curves_attribute(self, attribute: str, value):
        for curve in self.curves:
            setattr(curve, attribute, value)

    @property
    def color(self):
        """Colors for each curve.

        Returns:
            list[list[np.ndarray | None]]: For each curve, a two-element list of
            vertex colors (RGB arrays) or None.
        """
        return self._get_curves_attribute("color")

    @color.setter
    def color(self, value: None | tuple[float, float, float] | np.ndarray):
        """Set color for all curves.

        Args:
            value (tuple[float, float, float] | np.ndarray | None): RGB color as
                tuple/ndarray, or None to clear.
        """
        color_value = np.array(value, dtype=np.float32) if value is not None else None
        self._set_curves_attribute("color", color_value)

    @property
    def pressure(self):
        """Pressures for each curve.

        Returns:
            list[list[np.float32 | None]]: For each curve, a two-element list of
            vertex pressures or None.
        """
        return self._get_curves_attribute("pressure")

    @pressure.setter
    def pressure(self, value: None | float | np.float32):
        """Set pressure for all curves.

        Args:
            value (float | np.float32 | None): Pressure value or None to clear.
        """
        pressure_value = np.float32(value) if value is not None else None
        self._set_curves_attribute("pressure", pressure_value)

    @property
    def thickness(self):
        """Thicknesses for each curve.

        Returns:
            list[list[np.float32 | None]]: For each curve, a two-element list of
            vertex thicknesses or None.
        """
        return self._get_curves_attribute("thickness")

    @thickness.setter
    def thickness(self, value: None | float | np.float32):
        """Set thickness for all curves.

        Args:
            value (float | np.float32 | None): Thickness value or None to clear.
        """
        thickness_value = np.float32(value) if value is not None else None
        self._set_curves_attribute("thickness", thickness_value)

    @property
    def opacity(self):
        """Opacities for each curve.

        Returns:
            list[list[np.float32 | None]]: For each curve, a two-element list of
            vertex opacities or None.
        """
        return self._get_curves_attribute("opacity")

    @opacity.setter
    def opacity(self, value: None | float | np.float32):
        """Set opacity for all curves.

        Args:
            value (float | np.float32 | None): Opacity (0.0–1.0) or None to clear.
        """
        opacity_value = np.float32(value) if value is not None else None
        self._set_curves_attribute("opacity", opacity_value)


class Sketch:
    """Sketch composed of multiple paths.

    A Sketch collects Path objects to form a complete drawing and exposes
    convenient accessors for color, pressure, thickness, and opacity.

    Attributes:
        paths (list[Path]): Paths in the sketch.

    Examples:
        >>> sketch = Sketch([path1, path2, path3])
        >>> sketch.color = (1.0, 0.0, 0.0)  # Set all paths to red
        >>> sketch.thickness = 2.5          # Set all paths to thickness 2.5
    """

    def __init__(self, paths: list[Path] | None = None):
        """Initialize a Sketch.

        Args:
            paths (list[Path] | None, optional): Initial paths. Defaults to None.
        """
        self.paths = paths if paths else []

    @property
    def path_num(self):
        """Number of paths in the sketch.

        Returns:
            int: Count of paths.
        """
        return len(self.paths)

    @property
    def curve_num(self):
        """Total number of curves across all paths.

        Returns:
            int: Sum of curves in all paths.
        """
        return sum(path.curve_num for path in self.paths)

    def _get_paths_attribute(self, attribute: str):
        return [getattr(path, attribute) for path in self.paths]

    def _set_paths_attribute(self, attribute: str, value):
        for path in self.paths:
            setattr(path, attribute, value)

    @property
    def color(self):
        """Colors per path.

        Returns:
            list[list[list[np.ndarray | None]]]: For each path, a list of curve
            colors where each curve has a two-element list of vertex colors.
        """
        return self._get_paths_attribute("color")

    @color.setter
    def color(self, value: None | tuple[float, float, float] | np.ndarray):
        """Set color for all paths.

        Args:
            value (tuple[float, float, float] | np.ndarray | None): RGB color as
                tuple/ndarray, or None to clear.
        """
        color_value = np.array(value, dtype=np.float32) if value is not None else None
        self._set_paths_attribute("color", color_value)

    @property
    def pressure(self):
        """Pressures per path.

        Returns:
            list[list[list[np.float32 | None]]]: For each path, a list of curve
            pressures where each curve has a two-element list of vertex pressures.
        """
        return self._get_paths_attribute("pressure")

    @pressure.setter
    def pressure(self, value: None | float | np.float32):
        """Set pressure for all paths.

        Args:
            value (float | np.float32 | None): Pressure value or None to clear.
        """
        pressure_value = np.float32(value) if value is not None else None
        self._set_paths_attribute("pressure", pressure_value)

    @property
    def thickness(self):
        """Thicknesses per path.

        Returns:
            list[list[list[np.float32 | None]]]: For each path, a list of curve
            thicknesses where each curve has a two-element list of vertex values.
        """
        return self._get_paths_attribute("thickness")

    @thickness.setter
    def thickness(self, value: None | float | np.float32):
        """Set thickness for all paths.

        Args:
            value (float | np.float32 | None): Thickness value or None to clear.
        """
        thickness_value = np.float32(value) if value is not None else None
        self._set_paths_attribute("thickness", thickness_value)

    @property
    def opacity(self):
        """Opacities per path.

        Returns:
            list[list[list[np.float32 | None]]]: For each path, a list of curve
            opacities where each curve has a two-element list of vertex values.
        """
        return self._get_paths_attribute("opacity")

    @opacity.setter
    def opacity(self, value: None | float | np.float32):
        """Set opacity for all paths.

        Args:
            value (float | np.float32 | None): Opacity (0.0–1.0) or None to clear.
        """
        opacity_value = np.float32(value) if value is not None else None
        self._set_paths_attribute("opacity", opacity_value)
