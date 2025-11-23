import numpy as np
from sketchkit.utils.geometry import bezier_lengths, is_line
import svgpathtools


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

    def is_line(self) -> bool:
        """Check if the curve is a line.

        Returns:
            bool: True if the curve is a line, False otherwise.
        """
        return is_line(self.np_points.reshape(1, 4, 2))[0]

    @property
    def np_points(self) -> np.ndarray:
        """Get the points of the curve.

        Returns:
            np.ndarray: Array of points (shape: [4, 2]).
        """
        return np.array([
            [self.p_start.x, self.p_start.y],
            [self.p_crtl1.x, self.p_crtl1.y],
            [self.p_crtl2.x, self.p_crtl2.y],
            [self.p_end.x, self.p_end.y],
        ])

    def to_svg_path(self) -> str:
        """Convert the curve to an SVG path string.

        Returns:
            str: SVG path representation of the curve.
        """
        p0, p1, p2, p3 = self.p_start, self.p_crtl1, self.p_crtl2, self.p_end
        return f'M {p0.x} {p0.y} C {p1.x} {p1.y}, {p2.x} {p2.y}, {p3.x} {p3.y}'

    @property
    def length(self) -> float:
        """Calculate the length of the curve.

        Returns:
            float: Length of the curve.
        """
        
        # 将p1,p2,p3,p4整合成[1,4,2]
        points = self.np_points.reshape(1, 4, 2)
        return bezier_lengths(points)[0]
    
    def svg_length(self) -> float:
        """Calculate the length of the curve in SVG path units.

        Returns:
            float: Length of the curve in SVG path units.
        """
        path = svgpathtools.parse_path(self.to_svg_path())
        return path.length()
    
    
    def get_division_points(self, ts: np.ndarray) -> np.ndarray:
        """Get points on the curve at specified parameter values.

        Args:
            ts (np.ndarray): Array of parameter values between 0 and 1.

        Returns:
            np.ndarray: Array of points on the curve at the specified parameter values.
        """
        svg_path = svgpathtools.parse_path(self.to_svg_path())
        return np.array([[svg_path.point(t).real, svg_path.point(t).imag] for t in ts], dtype=np.float32)


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

    def curve_lengths(self):
        """Calculate the length of each curve in the path.

        Returns:
            float: Total length of the path.
        """
        points = np.array([curve.np_points for curve in self.curves])
        lengths = bezier_lengths(points)
        return lengths

    @property
    def length(self):
        """Calculate the total length of all curves in the path.

        Returns:
            float: Total length of the path.
        """
        return sum(self.curve_lengths())[0]
    
    def svg_curve_lengths(self) -> list[float]:
        """Calculate the lengths of each curve in SVG path units.

        Returns:
            np.ndarray: Array of lengths for each curve in SVG path units.
        """
        return np.array([curve.svg_length() for curve in self.curves])

    def svg_length(self) -> float:
        """Calculate the total length of all curves in the path in SVG path units.

        Returns:
            float: Total length of the path in SVG path units.
        """
        return sum(self.svg_curve_lengths())
    
    def get_division_points(self, ts: np.ndarray) -> np.ndarray:
        """Get points on the path at parameters ts.

        Args:
            ts (np.ndarray): Array of parameter values between 0 and 1.

        Returns:
            np.ndarray: Array of points on the path at parameters ts.
        """
        assert np.all((0.0 <= ts) & (ts <= 1.0)), "All t values must be in [0, 1]"
        lengths = self.svg_curve_lengths()
        pre_sum_lengths = np.cumsum(lengths)
        total_length = pre_sum_lengths[-1]
        target_lengths = ts * total_length
        curve_indices = np.searchsorted(pre_sum_lengths, target_lengths, side='right')
        points = np.zeros((len(ts), 2), dtype=np.float32)
        for i, curve_index in enumerate(curve_indices):
            if curve_index == 0:
                t_on_curve = target_lengths[i] / lengths[0]
                points[i] = self.curves[0].point(t_on_curve)
            elif curve_index < len(self.curves):
                t_on_curve = (target_lengths[i] - pre_sum_lengths[curve_index - 1]) / lengths[curve_index]
                points[i] = self.curves[curve_index].point(t_on_curve)
            else:
                points[i] = self.curves[-1].point(t_on_curve)
        return points



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

    def to_svg(self, size: int = 512, filename: str = None, fit_size: bool = False) -> str:
            """Convert the sketch to an SVG string.

            Args:
                size: Size of the SVG canvas (width and height in pixels).
                filename: Optional filename to save the SVG.
                fit_size: If True, scale and center the sketch while preserving aspect ratio.

            Returns:
                str: SVG representation of the sketch.
            """
            svg_parts = []
            svg_parts.append('<?xml version="1.0"?>')
            svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{size}" height="{size}">')
            svg_parts.append('<defs/>')
            svg_parts.append('<g>')

            # 计算缩放和偏移参数
            scale = 1.0
            offset_x = 0.0
            offset_y = 0.0
            
            if fit_size:
                # 收集所有点以确定边界
                all_points = []
                for path in self.paths:
                    for curve in path.curves:
                        if not isinstance(curve, Curve):
                            raise ValueError("All strokes must be of type `Curve`.")
                        
                        p0, p1, p2, p3 = curve.p_start, curve.p_crtl1, curve.p_crtl2, curve.p_end
                        all_points.append([p0.x, p0.y])
                        all_points.append([p1.x, p1.y])
                        all_points.append([p2.x, p2.y])
                        all_points.append([p3.x, p3.y])
                
                if all_points:
                    # 计算边界
                    points_array = np.array(all_points)
                    min_x, min_y = np.min(points_array, axis=0)
                    max_x, max_y = np.max(points_array, axis=0)
                    
                    # 计算大小和中心
                    width = max_x - min_x
                    height = max_y - min_y
                    center_x = (min_x + max_x) / 2
                    center_y = (min_y + max_y) / 2
                    
                    # 计算缩放因子以适应画布并留出边距
                    padding = 0.05  # 5%边距
                    scale = min((size * (1 - padding * 2)) / width, (size * (1 - padding * 2)) / height)
                    
                    # 计算平移以居中草图
                    offset_x = size / 2 - center_x * scale
                    offset_y = size / 2 - center_y * scale

            for path in self.paths:
                for curve in path.curves:
                    if not isinstance(curve, Curve):
                        raise ValueError("All strokes must be of type `Curve`.")

                    p0, p1, p2, p3 = curve.p_start, curve.p_crtl1, curve.p_crtl2, curve.p_end
                    color = curve.color[0] if curve.color[0] is not None else (0, 0, 0)
                    color_hex = f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}'
                    thickness = curve.thickness[0] if curve.thickness[0] is not None else 1.0
                    opacity = curve.opacity[0] if curve.opacity[0] is not None else 1.0

                    # 直接应用缩放和偏移到每个点的坐标
                    x0 = p0.x * scale + offset_x
                    y0 = p0.y * scale + offset_y
                    x1 = p1.x * scale + offset_x
                    y1 = p1.y * scale + offset_y
                    x2 = p2.x * scale + offset_x
                    y2 = p2.y * scale + offset_y
                    x3 = p3.x * scale + offset_x
                    y3 = p3.y * scale + offset_y

                    path_d = f'M {x0} {y0} C {x1} {y1}, {x2} {y2}, {x3} {y3}'
                    svg_parts.append(f'<path d="{path_d}" stroke="{color_hex}" stroke-width="{thickness}" fill="none" stroke-opacity="{opacity}"/>')

            svg_parts.append('</g>')
            svg_parts.append('</svg>')
            final_str = '\n'.join(svg_parts)
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(final_str)
            return final_str