import numpy as np
from sketchkit.utils.geometry import bezier_lengths, is_line
import svgpathtools
import json
from .schema import (
    SketchSchema,
    PathSchema,
    CurveArray,
    VertexArray,
    PointArray
)

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

    @classmethod
    def from_line(cls, line) -> "Curve":
        """
        Args:
            line: In shape (2, 2), including two points.

        Returns:
            cubic: Curve object.

        """

        ps, pe = line[0], line[1]
        start_x, start_y = ps[0], ps[1]
        end_x, end_y = pe[0], pe[1]

        p_start = Vertex(start_x, start_y)
        p_end = Vertex(end_x, end_y)

        ctrl1_x = 2.0 / 3.0 * start_x + 1.0 / 3.0 * end_x
        ctrl1_y = 2.0 / 3.0 * start_y + 1.0 / 3.0 * end_y
        ctrl2_x = 1.0 / 3.0 * start_x + 2.0 / 3.0 * end_x
        ctrl2_y = 1.0 / 3.0 * start_y + 2.0 / 3.0 * end_y

        ctrl1 = Point(ctrl1_x, ctrl1_y)
        ctrl2 = Point(ctrl2_x, ctrl2_y)
        cubic = Curve(p_start, p_end, ctrl1, ctrl2)
        # ctrl1 = np.stack([ctrl1_x, ctrl1_y])
        # ctrl2 = np.stack([ctrl2_x, ctrl2_y])
        # cubic = np.stack([p_start, ctrl1, ctrl2, p_end], axis=0)  # (4, 2)
        return cubic

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
        return np.array(
            [
                [self.p_start.x, self.p_start.y],
                [self.p_crtl1.x, self.p_crtl1.y],
                [self.p_crtl2.x, self.p_crtl2.y],
                [self.p_end.x, self.p_end.y],
            ]
        )

    def to_svg_path(self) -> str:
        """Convert the curve to an SVG path string.

        Returns:
            str: SVG path representation of the curve.
        """
        p0, p1, p2, p3 = self.p_start, self.p_crtl1, self.p_crtl2, self.p_end
        return f"M {p0.x} {p0.y} C {p1.x} {p1.y}, {p2.x} {p2.y}, {p3.x} {p3.y}"

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
        return np.array(
            [[svg_path.point(t).real, svg_path.point(t).imag] for t in ts],
            dtype=np.float32,
        )

    def is_zero_length(self) -> bool:
        """Check if the curve is zero length.

        Returns:
            bool: True if the curve is zero length, False otherwise.
        """
        points = self.np_points
        if np.all(points[0] == points[3]) and self.is_line():
            return True
        return False
    
    def split(self, t: float) -> tuple["Curve", "Curve"]:
        """Split the curve at the given parameter value.

        Args:
            t (float): Parameter value between 0 and 1.

        Returns:
            tuple[Curve, Curve]: Two new curves resulting from the split.
        """
        def lerp(A, B, t):
            return A + t * (B - A)

        # (N, 4, 2) -> 取出 4 个控制点
        P0, P1, P2, P3 = self.np_points.reshape(4, 2).unbind(dim=0)          # 每个 (2)

        P01 = lerp(P0, P1, t)                            # (2)
        P12 = lerp(P1, P2, t)                            # (2)
        P23 = lerp(P2, P3, t)                            # (2)

        P012 = lerp(P01, P12, t)                        # (2)
        P123 = lerp(P12, P23, t)                        # (2)

        P0123 = lerp(P012, P123, t)                     # (2)

        left_end = Vertex(P0123[0], P0123[1], self.p_start.pressure, self.p_start.thickness, self.p_start.color, self.p_start.opacity)
        left_ctrl1 = Point(P01[0], P01[1])
        left_ctrl2 = Point(P012[0], P012[1])

        right_start = Vertex(P0123[0], P0123[1], self.p_end.pressure, self.p_end.thickness, self.p_end.color, self.p_end.opacity)
        right_ctrl1 = Point(P123[0], P123[1])
        right_ctrl2 = Point(P23[0], P23[1])

        # 组装 left/right
        left  = Curve(self.p_start, left_ctrl1, left_ctrl2, left_end)
        right = Curve(right_start, right_ctrl1, right_ctrl2, self.p_end)

        return left, right


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

    def __init__(self, curves: list[Curve] | None = None, label: str | None = None):
        """Initialize a Path.

        Args:
            curves (list[Curve] | None, optional): Initial curves. Defaults to None.
            label (str | None, optional): Label for the path. Defaults to None.
        """
        self.curves = curves if curves else []
        self.label = label

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

        def _get_point(curve: Curve, t: float) -> np.ndarray:
            """Get a point on the curve at a specified parameter value.

            Args:
                t (float): Parameter value between 0 and 1.

            Returns:
                np.ndarray: Point on the curve at the specified parameter value.
            """
            m1 = curve.np_points[0]
            m2 = curve.np_points[1]
            m3 = curve.np_points[2]
            m4 = curve.np_points[3]
            p = (
                (1 - t) ** 3 * m1
                + 3 * (1 - t) ** 2 * t * m2
                + 3 * (1 - t) * t**2 * m3
                + t**3 * m4
            )
            return p

        assert np.all((0.0 <= ts) & (ts <= 1.0)), "All t values must be in [0, 1]"
        lengths = self.svg_curve_lengths()
        pre_sum_lengths = np.cumsum(lengths)
        total_length = pre_sum_lengths[-1]
        target_lengths = ts * total_length
        curve_indices = np.searchsorted(pre_sum_lengths, target_lengths, side="right")
        points = np.zeros((len(ts), 2), dtype=np.float32)
        for i, curve_index in enumerate(curve_indices):
            if curve_index == 0:
                t_on_curve = target_lengths[i] / lengths[0]
                points[i] = _get_point(self.curves[0], t_on_curve)
            elif curve_index < len(self.curves):
                t_on_curve = (
                    target_lengths[i] - pre_sum_lengths[curve_index - 1]
                ) / lengths[curve_index]
                points[i] = _get_point(self.curves[curve_index], t_on_curve)
            else:
                points[i] = _get_point(self.curves[-1], t_on_curve)
        return points

    # 剔除掉所有长度为0的curve
    def remove_empty_curves(self):
        """Remove all curves with zero length from the path."""
        self.curves = [curve for curve in self.curves if not curve.is_zero_length()]


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

    def __init__(self, height: float, width: float, paths: list[Path] | None = None):
        """Initialize a Sketch.

        Args:
            height (float, optional): Height of the sketch canvas. Defaults to 512.
            width (float, optional): Width of the sketch canvas. Defaults to 512.
            paths (list[Path] | None, optional): Initial paths. Defaults to None.
        """
        self.height = height
        self.width = width
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

    def to_svg(
        self,
        height: int | None = None,
        width: int | None = None,
        fit_size: bool = False,
        is_polyline: bool = False,
        filename: str | None = None,
    ) -> str:
        """Convert the sketch to an SVG string.

        Args:
            size: Size of the SVG canvas (width and height in pixels).
            filename: Optional filename to save the SVG.
            fit_size: If True, scale and center the sketch while preserving aspect ratio.

        Returns:
            str: SVG representation of the sketch.
        """
        width = self.width if width is None else width
        height = self.height if height is None else height

        svg_parts = []
        svg_parts.append('<?xml version="1.0"?>')
        svg_parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{width}" height="{height}">'
        )
        svg_parts.append("<defs/>")
        svg_parts.append("<g>")

        scale = 1.0
        offset_x = 0.0
        offset_y = 0.0

        if fit_size:
            all_points = []
            for path in self.paths:
                for curve in path.curves:
                    if not isinstance(curve, Curve):
                        raise ValueError("All strokes must be of type `Curve`.")

                    p0, p1, p2, p3 = (
                        curve.p_start,
                        curve.p_crtl1,
                        curve.p_crtl2,
                        curve.p_end,
                    )
                    all_points.append([p0.x, p0.y])
                    all_points.append([p1.x, p1.y])
                    all_points.append([p2.x, p2.y])
                    all_points.append([p3.x, p3.y])

            if all_points:
                points_array = np.array(all_points)
                min_x, min_y = np.min(points_array, axis=0)
                max_x, max_y = np.max(points_array, axis=0)

                t_width = max_x - min_x
                t_height = max_y - min_y
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2

                padding = 0.05  # 5%边距
                scale = min(
                    (width * (1 - padding * 2)) / t_width,
                    (height * (1 - padding * 2)) / t_height,
                )

                offset_x = width / 2 - center_x * scale
                offset_y = height / 2 - center_y * scale
        if is_polyline:
            for path in self.paths:
                points = []
                for curve in path.curves:
                    if not isinstance(curve, Curve):
                        raise ValueError("All strokes must be of type `Curve`.")
                    p0 = curve.p_start
                    points.append(
                        f"{p0.x * scale + offset_x},{p0.y * scale + offset_y}"
                    )
                end_point = path.curves[-1].p_end
                points.append(
                    f"{end_point.x * scale + offset_x},{end_point.y * scale + offset_y}"
                )
                color = path.color[0][0] if path.color[0][0] is not None else (0, 0, 0)

                color_hex = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"

                # width = np.average(
                #     path.pressure if path.pressure is not None else [1.0, 1.0]
                # ) * (path.thickness[0][0] if path.thickness[0][0] is not None else 1.0)
                width = 0.2
                svg_parts.append(
                    f'<polyline points="{" ".join(points)}" stroke="{color_hex}" stroke-width="{width}" fill="none"/>'
                )
        else:
            for path in self.paths:
                if not path.curves:
                    continue

                # Use attributes from the first curve of the path for the entire path
                first_curve = path.curves[0]
                color = first_curve.color[0] if first_curve.color[0] is not None else (0, 0, 0)
                color_hex = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
                thickness = (
                    first_curve.thickness[0] if first_curve.thickness[0] is not None else 1.0
                )
                opacity = first_curve.opacity[0] if first_curve.opacity[0] is not None else 1.0

                d_parts = []
                
                # Initial MoveTo
                p0 = first_curve.p_start
                last_x = p0.x * scale + offset_x
                last_y = p0.y * scale + offset_y
                d_parts.append(f"M {last_x} {last_y}")

                for curve in path.curves:
                    if not isinstance(curve, Curve):
                        raise ValueError("All strokes must be of type `Curve`.")

                    p0, p1, p2, p3 = (
                        curve.p_start,
                        curve.p_crtl1,
                        curve.p_crtl2,
                        curve.p_end,
                    )

                    x0 = p0.x * scale + offset_x
                    y0 = p0.y * scale + offset_y
                    x1 = p1.x * scale + offset_x
                    y1 = p1.y * scale + offset_y
                    x2 = p2.x * scale + offset_x
                    y2 = p2.y * scale + offset_y
                    x3 = p3.x * scale + offset_x
                    y3 = p3.y * scale + offset_y

                    # If the start of this curve doesn't match the end of the last one, issue a MoveTo
                    if abs(x0 - last_x) > 1e-4 or abs(y0 - last_y) > 1e-4:
                        d_parts.append(f"M {x0} {y0}")

                    d_parts.append(f"C {x1} {y1}, {x2} {y2}, {x3} {y3}")
                    last_x, last_y = x3, y3
                
                path_d = " ".join(d_parts)
                svg_parts.append(
                    f'<path d="{path_d}" stroke="{color_hex}" stroke-width="{thickness}" fill="none" stroke-opacity="{opacity}"/>'
                )

        svg_parts.append("</g>")
        svg_parts.append("</svg>")
        final_str = "\n".join(svg_parts)

        if filename:
            with open(filename, "w") as f:
                f.write(final_str)
        return final_str

    def get_progressive_frames(self, length_per_frame: int = 10) -> list["Sketch"]:
        import queue
        sketch_frames = []

        for i in range(self.path_num):
            base_pathes = [] if i == 0 else self.paths[:i]
            current_path = self.paths[i]
            # curve_lengths = current_path.svg_curve_lengths()
            # path_length = np.sum(curve_lengths)
            segment_length = 0
            curves = current_path.curves.copy()
            curves = curves.reverse()
            new_curves = []
            while len(curves) > 0:
                current_curve = curves.pop()
                curve_length = current_curve.svg_length()
                if segment_length + curve_length <= length_per_frame:
                    segment_length += curve_length
                    new_curves.append(current_curve)
                else:
                    split_t = (length_per_frame - segment_length) / curve_length
                    left_curve, right_curve = current_curve.split(split_t)
                    curves.append(right_curve)
                    new_curves.append(left_curve)
                    sketch_frames.append(Sketch(base_pathes + [Path(new_curves)]))
                    segment_length = 0

        return sketch_frames

    def resize(self, height: float, width: float):
        """Resize the sketch canvas and scale all strokes accordingly.

        This rescales all curve points (including Bezier control points) from the
        current canvas size to the new canvas size.

        Notes:
            - X coordinates are scaled by ``width / old_width``.
            - Y coordinates are scaled by ``height / old_height``.
            - Stroke thickness (if present) is scaled by the average of the two
              scale factors.

        Args:
            height (int): New height.
            width (int): New width.
        """

        old_height = self.height
        old_width = self.width

        sx = float(width) / float(old_width)
        sy = float(height) / float(old_height)
        s_thickness = np.float32((sx + sy) / 2.0)

        def _scale_xy(p: Point):
            p.x = np.float32(float(p.x) * sx)
            p.y = np.float32(float(p.y) * sy)

        for path in self.paths:
            for curve in path.curves:
                if not isinstance(curve, Curve):
                    raise ValueError("All strokes must be of type `Curve`.")

                _scale_xy(curve.p_start)
                _scale_xy(curve.p_crtl1)
                _scale_xy(curve.p_crtl2)
                _scale_xy(curve.p_end)

                if curve.p_start.thickness is not None:
                    curve.p_start.thickness = (
                        np.float32(curve.p_start.thickness) * s_thickness
                    )
                if curve.p_end.thickness is not None:
                    curve.p_end.thickness = (
                        np.float32(curve.p_end.thickness) * s_thickness
                    )
        self.resize_canvas(height, width)

    def resize_canvas(self, height: float, width: float):
        """Resize the sketch canvas without scaling strokes.

        This changes the canvas size but leaves all curve points and stroke
        properties unchanged.

        Args:
            height (float): New height.
            width (float): New width.
        """
        self.height = height
        self.width = width

    def to_json(self, file_path: str):
        """Save the sketch to a JSON file (Compact Format).

        Args:
            file_path (str): Path to the output JSON file.
        """
        paths_data = []
        for path in self.paths:
            curves_data = []
            for curve in path.curves:
                p_start: VertexArray = (
                    float(curve.p_start.x),
                    float(curve.p_start.y),
                    float(curve.p_start.pressure)
                    if curve.p_start.pressure is not None
                    else None,
                    float(curve.p_start.thickness)
                    if curve.p_start.thickness is not None
                    else None,
                    tuple(curve.p_start.color.tolist())
                    if curve.p_start.color is not None
                    else None,
                    float(curve.p_start.opacity)
                    if curve.p_start.opacity is not None
                    else None,
                )
                p_end: VertexArray = (
                    float(curve.p_end.x),
                    float(curve.p_end.y),
                    float(curve.p_end.pressure)
                    if curve.p_end.pressure is not None
                    else None,
                    float(curve.p_end.thickness)
                    if curve.p_end.thickness is not None
                    else None,
                    tuple(curve.p_end.color.tolist())
                    if curve.p_end.color is not None
                    else None,
                    float(curve.p_end.opacity)
                    if curve.p_end.opacity is not None
                    else None,
                )
                p_crtl1: PointArray = (
                    float(curve.p_crtl1.x),
                    float(curve.p_crtl1.y),
                )
                p_crtl2: PointArray = (
                    float(curve.p_crtl2.x),
                    float(curve.p_crtl2.y),
                )

                # [start, c1, c2, end]
                curves_data.append((p_start, p_crtl1, p_crtl2, p_end))

            paths_data.append(PathSchema(curves=curves_data, label=path.label))

        sketch_data = SketchSchema(
            height=float(self.height), width=float(self.width), paths=paths_data
        )

        with open(file_path, "w") as f:
            f.write(sketch_data.model_dump_json(indent=None))  # Minified JSON

    @classmethod
    def from_json(cls, json_path: str) -> "Sketch":
        """Load a sketch from a JSON file (Compact Format).

        Args:
            json_path (str): Path to the JSON file.

        Returns:
            Sketch: Loaded sketch object.
        """
        with open(json_path, "r") as f:
            json_content = f.read()

        sketch_data = SketchSchema.model_validate_json(json_content)

        paths = []
        for path_data in sketch_data.paths:
            curves = []
            for curve_tuple in path_data.curves:
                # curve_tuple: [start, c1, c2, end]
                # start/end: [x, y, p, t, c, o]
                # c1/c2: [x, y]

                start_data = curve_tuple[0]
                c1_data = curve_tuple[1]
                c2_data = curve_tuple[2]
                end_data = curve_tuple[3]

                p_start = Vertex(
                    x=start_data[0],
                    y=start_data[1],
                    pressure=start_data[2],
                    thickness=start_data[3],
                    color=start_data[4],
                    opacity=start_data[5],
                )
                p_end = Vertex(
                    x=end_data[0],
                    y=end_data[1],
                    pressure=end_data[2],
                    thickness=end_data[3],
                    color=end_data[4],
                    opacity=end_data[5],
                )
                p_crtl1 = Point(x=c1_data[0], y=c1_data[1])
                p_crtl2 = Point(x=c2_data[0], y=c2_data[1])

                curves.append(Curve(p_start, p_end, p_crtl1, p_crtl2))

            paths.append(Path(curves=curves, label=path_data.label))

        return cls(height=sketch_data.height, width=sketch_data.width, paths=paths)
