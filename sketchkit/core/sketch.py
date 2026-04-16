import numpy as np
from sketchkit.utils.geometry import bezier_lengths, is_line
from sketchkit.utils.transform import (
    parse_single_path,
    parse_transform,
    apply_transform_to_path,
)
import svgpathtools
import json
from .schema import CurveSchema, SketchSchema, PathSchema, PointSchema
import xml.etree.ElementTree as ET


def _copy_attributes(attributes: dict | None) -> dict:
    return dict(attributes) if attributes is not None else {}


def _serialize_point(point: "Point") -> PointSchema:
    return PointSchema(
        x=float(point.x),
        y=float(point.y),
        pressure=float(point.pressure) if point.pressure is not None else None,
        thickness=float(point.thickness) if point.thickness is not None else None,
        color=tuple(point.color.tolist()) if point.color is not None else None,
        opacity=float(point.opacity) if point.opacity is not None else None,
        attributes=_copy_attributes(point.attributes),
    )


def _deserialize_point(point_data: PointSchema) -> "Point":
    return Point(
        x=point_data.x,
        y=point_data.y,
        pressure=point_data.pressure,
        thickness=point_data.thickness,
        color=point_data.color,
        opacity=point_data.opacity,
        attributes=point_data.attributes,
    )


class Point:
    """2D point with optional drawing attributes.

    `Point` is now the single 2D point type used across the library. It can
    represent both geometric control points and styled stroke endpoints.
    Custom user metadata can be stored in `attributes`.
    """

    def __init__(
        self,
        x: float | np.float32,
        y: float | np.float32,
        pressure: None | float | np.float32 = None,
        thickness: None | float | np.float32 = None,
        color: None | tuple[float, float, float] | np.ndarray = None,
        opacity: None | float | np.float32 = None,
        attributes: dict | None = None,
    ):
        self.x = np.float32(x)
        self.y = np.float32(y)
        self.pressure = np.float32(pressure) if pressure is not None else None
        self.thickness = np.float32(thickness) if thickness is not None else None
        self.color = np.array(color, dtype=np.float32) if color is not None else None
        self.opacity = np.float32(opacity) if opacity is not None else None
        self.attributes = _copy_attributes(attributes)

class Curve:
    """Curve defined by two endpoint points and two control points.

    Connects two endpoint points (start and end) with optional control points
    for a curved path. Visual properties (color, pressure, thickness, opacity)
    are applied to both endpoints together.

    Attributes:
        p_start (Point): Starting endpoint.
        p_end (Point): Ending endpoint.
        p_crtl1 (Point): First control point for curve definition.
        p_crtl2 (Point): Second control point for curve definition.
        attributes (dict): Custom user metadata for the curve.

    Notes:
        When setting properties, the same value is applied to both endpoints.
        Numeric properties are converted to np.float32 automatically.
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

        p_start = Point(start_x, start_y)
        p_end = Point(end_x, end_y)

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
        p_start: Point,
        p_end: Point,
        p_crtl1: Point,
        p_crtl2: Point,
        attributes: dict | None = None,
    ):
        """Initialize a Curve.

        Args:
            p_start (Point): Starting endpoint.
            p_end (Point): Ending endpoint.
            p_crtl1 (Point): First control point.
            p_crtl2 (Point): Second control point.
        """
        self.p_start = p_start
        self.p_end = p_end
        self.p_crtl1 = p_crtl1
        self.p_crtl2 = p_crtl2
        self.attributes = _copy_attributes(attributes)

    def _get_endpoint_attribute(self, attribute: str):
        return [
            getattr(self.p_start, attribute),
            getattr(self.p_end, attribute),
        ]

    def _set_endpoint_attribute(self, attribute: str, value):
        setattr(self.p_start, attribute, value)
        setattr(self.p_end, attribute, value)

    @property
    def color(self):
        """Colors of the start and end endpoints.

        Returns:
            list[np.ndarray | None]: A two-element list of RGB colors for the
            start and end endpoints (each as float32 arrays) or None.
        """
        return self._get_endpoint_attribute("color")

    @color.setter
    def color(self, value: None | tuple[float, float, float] | np.ndarray):
        """Set color for both endpoints.

        Args:
            value (tuple[float, float, float] | np.ndarray | None): RGB color as
                tuple/ndarray, or None to clear.
        """
        color_value = np.array(value, dtype=np.float32) if value is not None else None
        self._set_endpoint_attribute("color", color_value)

    @property
    def pressure(self):
        """Pressures of the start and end endpoints.

        Returns:
            list[np.float32 | None]: Two-element list of pressures or None.
        """
        return self._get_endpoint_attribute("pressure")

    @pressure.setter
    def pressure(self, value: None | float | np.float32):
        """Set pressure for both endpoints.

        Args:
            value (float | np.float32 | None): Pressure value or None to clear.
        """
        pressure_value = np.float32(value) if value is not None else None
        self._set_endpoint_attribute("pressure", pressure_value)

    @property
    def thickness(self):
        """Thicknesses of the start and end endpoints.

        Returns:
            list[np.float32 | None]: Two-element list of thicknesses or None.
        """
        return self._get_endpoint_attribute("thickness")

    @thickness.setter
    def thickness(self, value: None | float | np.float32):
        """Set thickness for both endpoints.

        Args:
            value (float | np.float32 | None): Thickness value or None to clear.
        """
        thickness_value = np.float32(value) if value is not None else None
        self._set_endpoint_attribute("thickness", thickness_value)

    @property
    def opacity(self):
        """Opacities of the start and end endpoints.

        Returns:
            list[np.float32 | None]: Two-element list of opacities or None.
        """
        return self._get_endpoint_attribute("opacity")

    @opacity.setter
    def opacity(self, value: None | float | np.float32):
        """Set opacity for both endpoints.

        Args:
            value (float | np.float32 | None): Opacity (0.0–1.0) or None to clear.
        """
        opacity_value = np.float32(value) if value is not None else None
        self._set_endpoint_attribute("opacity", opacity_value)

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
        P0, P1, P2, P3 = self.np_points[0], self.np_points[1], self.np_points[2], self.np_points[3]          # 每个 (2)

        P01 = lerp(P0, P1, t)                            # (2)
        P12 = lerp(P1, P2, t)                            # (2)
        P23 = lerp(P2, P3, t)                            # (2)

        P012 = lerp(P01, P12, t)                        # (2)
        P123 = lerp(P12, P23, t)                        # (2)

        P0123 = lerp(P012, P123, t)                     # (2)

        left_end = Point(
            P0123[0],
            P0123[1],
            self.p_start.pressure if self.p_start.pressure is not None else None,
            self.p_start.thickness,
            self.p_start.color,
            self.p_start.opacity,
            attributes=self.p_start.attributes,
        )
        left_ctrl1 = Point(P01[0], P01[1], attributes=self.p_crtl1.attributes)
        left_ctrl2 = Point(P012[0], P012[1], attributes=self.p_crtl2.attributes)

        right_start = Point(
            P0123[0],
            P0123[1],
            self.p_end.pressure if self.p_end.pressure is not None else None,
            self.p_end.thickness,
            self.p_end.color,
            self.p_end.opacity,
            attributes=self.p_end.attributes,
        )
        right_ctrl1 = Point(P123[0], P123[1], attributes=self.p_crtl1.attributes)
        right_ctrl2 = Point(P23[0], P23[1], attributes=self.p_crtl2.attributes)

        # 组装 left/right
        left  = Curve(self.p_start, left_end, left_ctrl1, left_ctrl2, attributes=self.attributes)
        right = Curve(right_start, self.p_end, right_ctrl1, right_ctrl2, attributes=self.attributes)

        return left, right


class Path:
    """Collection of curves.

    Manages multiple curves and applies common operations to all of them.

    Attributes:
        curves (list[Curve]): Curves in this path.
        attributes (dict): Custom user metadata for the path.

    Examples:
        >>> path = Path([curve1, curve2, curve3])
        >>> path.color = (1.0, 0.0, 0.0)  # Set all curves to red
        >>> path.thickness = 2.5          # Set all curves to thickness 2.5
        >>> path.curve_num
        3
    """

    def __init__(
        self,
        curves: list[Curve] | None = None,
        label: str | None = None,
        attributes: dict | None = None,
    ):
        """Initialize a Path.

        Args:
            curves (list[Curve] | None, optional): Initial curves. Defaults to None.
            label (str | None, optional): Label for the path. Defaults to None.
        """
        self.curves = curves if curves else []
        self.label = label
        self.attributes = _copy_attributes(attributes)

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
            endpoint colors (RGB arrays) or None.
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
            endpoint pressures or None.
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
            endpoint thicknesses or None.
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
            endpoint opacities or None.
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
        attributes (dict): Custom user metadata for the sketch.

    Examples:
        >>> sketch = Sketch([path1, path2, path3])
        >>> sketch.color = (1.0, 0.0, 0.0)  # Set all paths to red
        >>> sketch.thickness = 2.5          # Set all paths to thickness 2.5
    """

    def __init__(
        self,
        height: float,
        width: float,
        paths: list[Path] | None = None,
        attributes: dict | None = None,
    ):
        """Initialize a Sketch.

        Args:
            height (float, optional): Height of the sketch canvas. Defaults to 512.
            width (float, optional): Width of the sketch canvas. Defaults to 512.
            paths (list[Path] | None, optional): Initial paths. Defaults to None.
        """
        self.height = height
        self.width = width
        self.paths = paths if paths else []
        self.attributes = _copy_attributes(attributes)

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
            colors where each curve has a two-element list of endpoint colors.
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
            pressures where each curve has a two-element list of endpoint pressures.
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
            thicknesses where each curve has a two-element list of endpoint values.
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
            opacities where each curve has a two-element list of endpoint values.
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
                color = (
                    first_curve.color[0]
                    if first_curve.color[0] is not None
                    else (0, 0, 0)
                )
                color_hex = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
                thickness = (
                    first_curve.thickness[0]
                    if first_curve.thickness[0] is not None
                    else 1.0
                )
                opacity = (
                    first_curve.opacity[0]
                    if first_curve.opacity[0] is not None
                    else 1.0
                )

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
        sketch_frames = []

        for i in range(self.path_num):
            base_pathes = [] if i == 0 else self.paths[:i]
            current_path = self.paths[i]
            # print(len(current_path.curves))
            # curve_lengths = current_path.svg_curve_lengths()
            # path_length = np.sum(curve_lengths)
            segment_length = 0
            curves = current_path.curves.copy()
            i = 0
            current_curve = curves[0]
            new_curves = []
            while i < len(curves):
                
                # print(current_curve)
                curve_length = current_curve.svg_length()
                if segment_length + curve_length <= length_per_frame:
                    segment_length += curve_length
                    i += 1
                    new_curves = []
                    if i == len(curves):
                        sketch_frames.append(Sketch(height=self.height, width=self.width, paths=base_pathes + [current_path]))
                    else:
                        current_curve = curves[i]
                else:
                    split_t = (length_per_frame - segment_length) / curve_length
                    left_curve, right_curve = current_curve.split(split_t)
                    current_curve = right_curve
                    new_curves.append(left_curve)
                    sketch_frames.append(Sketch(height=self.height, width=self.width, paths=base_pathes + [Path(curves[:i] + new_curves)]))
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
                p_start = _serialize_point(curve.p_start)
                p_end = _serialize_point(curve.p_end)
                p_crtl1 = _serialize_point(curve.p_crtl1)
                p_crtl2 = _serialize_point(curve.p_crtl2)

                curves_data.append(
                    CurveSchema(
                        p_start=p_start,
                        p_crtl1=p_crtl1,
                        p_crtl2=p_crtl2,
                        p_end=p_end,
                        attributes=_copy_attributes(curve.attributes),
                    )
                )

            paths_data.append(
                PathSchema(
                    curves=curves_data,
                    label=path.label,
                    attributes=_copy_attributes(path.attributes),
                )
            )

        sketch_data = SketchSchema(
            height=float(self.height),
            width=float(self.width),
            paths=paths_data,
            attributes=_copy_attributes(self.attributes),
        )

        with open(file_path, "w") as f:
            f.write(sketch_data.model_dump_json(indent=None))  # Minified JSON

    @classmethod
    def from_json(cls, file_path: str) -> "Sketch":
        """Load a sketch from a JSON file (Compact Format).

        Args:
            json_path (str): Path to the JSON file.

        Returns:
            Sketch: Loaded sketch object.
        """
        with open(file_path, "r") as f:
            json_content = f.read()

        sketch_data = SketchSchema.model_validate_json(json_content)

        paths = []
        for path_data in sketch_data.paths:
            curves = []
            for curve_data in path_data.curves:
                p_start = _deserialize_point(curve_data.p_start)
                p_end = _deserialize_point(curve_data.p_end)
                p_crtl1 = _deserialize_point(curve_data.p_crtl1)
                p_crtl2 = _deserialize_point(curve_data.p_crtl2)

                curves.append(
                    Curve(
                        p_start,
                        p_end,
                        p_crtl1,
                        p_crtl2,
                        attributes=curve_data.attributes,
                    )
                )

            paths.append(
                Path(
                    curves=curves,
                    label=path_data.label,
                    attributes=path_data.attributes,
                )
            )

        return cls(
            height=sketch_data.height,
            width=sketch_data.width,
            paths=paths,
            attributes=sketch_data.attributes,
        )

    @classmethod
    def from_svg(
        cls,
        svg_file: str,
        *,
        default_thickness: float = 1.0,
        default_pressure: float | None = None,
        cast_canvas_to_int: bool = True,
    ) -> "Sketch":
        invalid_svg_shapes = [
            "rect",
            "circle",
            "ellipse",
            "line",
            "polyline",
            "polygon",
        ]

        tree = ET.parse(svg_file)
        root = tree.getroot()

        # --- Canvas size: strictly from viewBox like TU-Berlin ---
        view_box = root.get("viewBox")
        if view_box:
            # raise ValueError(
            #     "TU-Berlin aligned loader requires <svg viewBox='0 0 W H'>."
            # )

            parts = view_box.replace(",", " ").split()
            if len(parts) != 4:
                raise ValueError(f"Invalid viewBox: {view_box!r}")

            view_x, view_y, view_w, view_h = [float(p) for p in parts]
            if int(view_x) != 0 or int(view_y) != 0:
                raise ValueError(
                    f"TU-Berlin aligned loader expects viewBox minX/minY = 0, got {view_x},{view_y}"
                )

            canvas_w = view_w - view_x
            canvas_h = view_h - view_y
            if cast_canvas_to_int:
                canvas_w = int(canvas_w)
                canvas_h = int(canvas_h)
        else:
            canvas_w = int(root.get("width"))
            canvas_h = int(root.get("height"))
            if not canvas_w or not canvas_h:
                raise ValueError(
                    "loader requires <svg width='W' height='H'>."
                )

        # --- Parse paths with optional group transform (TU-Berlin style) ---
        path_objs: List[Path] = []
        matrix: Optional[TransformMatrix] = None

        for elem in root.iter():
            # strip namespace in the same spirit as TU-Berlin
            try:
                _, tag_suffix = elem.tag.split("}")
            except ValueError:
                tag_suffix = elem.tag

            if tag_suffix in invalid_svg_shapes:
                raise ValueError(
                    f"Unsupported SVG shape <{tag_suffix}> (TU-Berlin aligned loader)."
                )

            if tag_suffix == "g":
                if "transform" not in elem.attrib:
                    continue
                matrix = parse_transform(elem.attrib["transform"])

            elif tag_suffix == "path":
                if "transform" in elem.attrib:
                    raise ValueError(
                        "TU-Berlin aligned loader forbids <path transform='...'>. Put transform on <g>."
                    )

                d = elem.attrib.get("d", "").strip()
                if not d:
                    continue

                if matrix is not None:
                    d = apply_transform_to_path(d, matrix)

                subpaths = parse_single_path(d)  # NOTE: now returns multiple subpaths
                if subpaths is None:
                    continue

                for control_points_list in subpaths:
                    segment_num = (len(control_points_list) - 1) // 3
                    curves: List[Curve] = []

                    for i in range(segment_num):
                        raw_curve = control_points_list[i * 3 : i * 3 + 4]  # 4 points

                        p_start = Point(
                            raw_curve[0][0],
                            raw_curve[0][1],
                            thickness=default_thickness,
                            pressure=default_pressure,
                        )
                        p_ctrl1 = Point(raw_curve[1][0], raw_curve[1][1])
                        p_ctrl2 = Point(raw_curve[2][0], raw_curve[2][1])
                        p_end = Point(
                            raw_curve[3][0],
                            raw_curve[3][1],
                            thickness=default_thickness,
                            pressure=default_pressure,
                        )

                        curves.append(Curve(p_start, p_end, p_ctrl1, p_ctrl2))

                    if curves:
                        path_objs.append(Path(curves=curves))

        if not path_objs:
            raise ValueError("No valid <path d='...'> found in SVG.")

        return cls(height=canvas_h, width=canvas_w, paths=path_objs)
