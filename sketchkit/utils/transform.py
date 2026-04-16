import numpy as np
import json
from svg.path import parse_path, path as svg_path
import math
import re

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sketchkit.core.sketch import Sketch, Path, Curve, Point

from typing import TYPE_CHECKING

# def convert_to_curve(sketch: Sketch, maxError=4.0):
#     from sketchkit.utils.fit_curve import fit_curve

#     """Convert a Sketch with polylines to one with Bezier curves.

#     Args:
#         sketch (Sketch): Input sketch with polylines.
#         maxError (float, optional): Maximum fitting error. Defaults to 4.0.

#     Returns:
#         Sketch: New sketch with Bezier curves.
#     """
#     curve_paths = []
#     for path in sketch.paths:
#         if path.curves is None or len(path.curves) < 1:
#             continue
#         # print([[[curve.p_start.x, curve.p_start.y], [curve.p_end.x, curve.p_end.y]] for curve in path.curves])
#         polyline_path = []
#         for curve in path.curves:
#             if len(polyline_path) == 0:
#                 polyline_path.append([curve.p_start.x, curve.p_start.y])
#             elif (
#                 np.linalg.norm(
#                     np.array(polyline_path[-1])
#                     - np.array([curve.p_start.x, curve.p_start.y])
#                 )
#                 > 1
#             ):
#                 polyline_path.append([curve.p_start.x, curve.p_start.y])
#         if (
#             np.linalg.norm(
#                 np.array(polyline_path[-1])
#                 - np.array([path.curves[-1].p_end.x, path.curves[-1].p_end.y])
#             )
#             > 1
#         ):
#             polyline_path.append([path.curves[-1].p_end.x, path.curves[-1].p_end.y])
#         print(polyline_path)
#         # if len(polyline_path) < 4:
#         #     continue
#         curve_path = fit_curve(np.array(polyline_path), maxError)
#         curves = [
#             Curve(
#                 Point(pt[0][0], pt[0][1]),
#                 Point(pt[3][0], pt[3][1]),
#                 Point(pt[1][0], pt[1][1]),
#                 Point(pt[2][0], pt[2][1]),
#             )
#             for pt in curve_path
#         ]
#         curve_paths.append(Path(curves=curves))
#     new_sketch = Sketch(paths=curve_paths)
#     return new_sketch


def hex_to_float_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError("Wrong hex color format.")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


# def load_sketch(json_path: str) -> Sketch:
#     with open(json_path, "r") as f:
#         data = json.load(f)
#         paths = []
#         for path_dict in data:
#             txy = path_dict["txy"]
#             pressure = path_dict["p"]
#             color = hex_to_float_rgb(path_dict["color"])
#             width = int(path_dict["width"])
#             assert len(pressure) * 3 == len(
#                 txy
#             ), "Pressure and txy must have the same length."
#             curves = []
#             for i in range(len(pressure) - 1):
#                 p0 = txy[3 * i + 1 : 3 * i + 3]
#                 p1 = txy[3 * i + 4 : 3 * i + 6]
#                 line = np.array([p0, p1])
#                 segment = Curve.from_line(line)
#                 segment.p_start.pressure = pressure[i]
#                 segment.p_end.pressure = pressure[i + 1]
#                 curves.append(segment)
#             path = Path(curves=curves)
#             path.color = color
#             path.thickness = width
#             paths.append(path)
#         sketch = Sketch(paths=paths)
#         return sketch


# if __name__ == "__main__":
#     json_path = "media/eg_drawing (3)_drawing (1).json"
#     sketch = load_sketch(json_path)
#     svg_str = sketch.to_svg(size=800, is_polyline=True)
#     with open("media/eg_drawing (3)_drawing (1).svg", "w") as f:
#         f.write(svg_str)


# -------------------------
# Parse a single <path d="..."> into multiple subpaths (simple, robust)
# Each subpath is encoded like TU-Berlin:
#   control_points_list = [P0, C1, C2, P1, C1, C2, P2, ...]
# where the first point is Move start, and every segment contributes 3 points.
# Supports:
#   - Move, CubicBezier, Line (converted to cubic), QuadraticBezier (converted to cubic)
#   - Ignores Close (Z) (same spirit as TU-Berlin; dataset typically explicit)
#   - Raises on Arc
# -------------------------
def parse_single_path(path_str: str) -> list[list[tuple[float, float]]] | None:
    try:
        ps = parse_path(path_str)
    except Exception as e:
        print(e)
        return None

    subpaths: list[list[tuple[float, float]]] = []
    control_points_list: list[tuple[float, float]] = []

    for path_item in ps:
        path_type = type(path_item)

        if path_type == svg_path.Move:
            # finalize previous subpath (if it has at least one segment)
            if len(control_points_list) > 1:
                assert (len(control_points_list) - 1) % 3 == 0
                subpaths.append(control_points_list)

            # start a new subpath
            control_points_list = []
            start = path_item.start
            control_points_list.append((start.real, start.imag))

        elif path_type == svg_path.CubicBezier:
            control1 = path_item.control1
            control2 = path_item.control2
            end = path_item.end
            # If SVG is malformed (cubic without a Move), we skip it
            if not control_points_list:
                control_points_list = [(path_item.start.real, path_item.start.imag)]
            control_points_list.append((control1.real, control1.imag))
            control_points_list.append((control2.real, control2.imag))
            control_points_list.append((end.real, end.imag))

        elif path_type == svg_path.Line:
            start, end = path_item.start, path_item.end
            sx, sy = start.real, start.imag
            ex, ey = end.real, end.imag

            # line -> cubic
            c1x = 2.0 / 3.0 * sx + 1.0 / 3.0 * ex
            c1y = 2.0 / 3.0 * sy + 1.0 / 3.0 * ey
            c2x = 1.0 / 3.0 * sx + 2.0 / 3.0 * ex
            c2y = 1.0 / 3.0 * sy + 2.0 / 3.0 * ey

            if not control_points_list:
                control_points_list = [(sx, sy)]
            control_points_list.append((c1x, c1y))
            control_points_list.append((c2x, c2y))
            control_points_list.append((ex, ey))

        elif path_type == svg_path.QuadraticBezier:
            # quadratic -> cubic
            start = path_item.start
            q1 = path_item.control
            end = path_item.end
            x0, y0 = start.real, start.imag
            x1, y1 = q1.real, q1.imag
            x2, y2 = end.real, end.imag

            c1 = (x0 + 2.0 / 3.0 * (x1 - x0), y0 + 2.0 / 3.0 * (y1 - y0))
            c2 = (x2 + 2.0 / 3.0 * (x1 - x2), y2 + 2.0 / 3.0 * (y1 - y2))

            if not control_points_list:
                control_points_list = [(x0, y0)]
            control_points_list.append(c1)
            control_points_list.append(c2)
            control_points_list.append((x2, y2))

        elif path_type == svg_path.Arc:
            raise Exception("Arc is not supported (simple TU-Berlin-aligned loader).")

        elif path_type == svg_path.Close:
            # keep it simple: ignore explicit close
            # If you want: convert to a line-to-cubic back to subpath start.
            pass

        else:
            raise Exception("Unknown path_type", path_type)

    # finalize last subpath
    if len(control_points_list) > 1:
        assert (len(control_points_list) - 1) % 3 == 0
        subpaths.append(control_points_list)

    return subpaths if subpaths else None


# -------------------------
# Transform support (full affine) applied by rewriting d
# -------------------------
class TransformMatrix:
    def __init__(self):
        # identity matrix in SVG affine (a b c d e f)
        self.a, self.b, self.c, self.d, self.e, self.f = 1, 0, 0, 1, 0, 0

    def multiply(self, a, b, c, d, e, f):
        new_a = self.a * a + self.c * b
        new_b = self.b * a + self.d * b
        new_c = self.a * c + self.c * d
        new_d = self.b * c + self.d * d
        new_e = self.a * e + self.c * f + self.e
        new_f = self.b * e + self.d * f + self.f
        self.a, self.b, self.c, self.d, self.e, self.f = (
            new_a,
            new_b,
            new_c,
            new_d,
            new_e,
            new_f,
        )

    def translate(self, tx, ty):
        self.multiply(1, 0, 0, 1, tx, ty)

    def scale(self, sx, sy=None):
        if sy is None:
            sy = sx
        self.multiply(sx, 0, 0, sy, 0, 0)

    def rotate(self, angle, cx=None, cy=None):
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        if cx is not None and cy is not None:
            self.translate(cx, cy)
            self.multiply(cos_a, sin_a, -sin_a, cos_a, 0, 0)
            self.translate(-cx, -cy)
        else:
            self.multiply(cos_a, sin_a, -sin_a, cos_a, 0, 0)

    def skew_x(self, angle):
        rad = math.radians(angle)
        tan_a = math.tan(rad)
        self.multiply(1, 0, tan_a, 1, 0, 0)

    def skew_y(self, angle):
        rad = math.radians(angle)
        tan_a = math.tan(rad)
        self.multiply(1, tan_a, 0, 1, 0, 0)

    def matrix(self, a, b, c, d, e, f):
        self.multiply(a, b, c, d, e, f)

    def apply_to_point(self, x, y):
        new_x = self.a * x + self.c * y + self.e
        new_y = self.b * x + self.d * y + self.f
        return new_x, new_y


def parse_transform(transform_str: str) -> TransformMatrix:
    matrix = TransformMatrix()
    transforms = re.findall(r"(\w+)\(([^)]+)\)", transform_str)

    for transform_type, params_str in transforms:
        params = [float(p) for p in re.split(r"[,\s]+", params_str) if p]

        if transform_type == "translate":
            if len(params) == 1:
                matrix.translate(params[0], 0)
            else:
                matrix.translate(params[0], params[1])

        elif transform_type == "scale":
            if len(params) == 1:
                matrix.scale(params[0])
            else:
                matrix.scale(params[0], params[1])

        elif transform_type == "rotate":
            if len(params) == 1:
                matrix.rotate(params[0])
            else:
                matrix.rotate(params[0], params[1], params[2])

        elif transform_type == "skewX":
            matrix.skew_x(params[0])

        elif transform_type == "skewY":
            matrix.skew_y(params[0])

        elif transform_type == "matrix":
            matrix.matrix(*params)

        else:
            raise ValueError(f"Unsupported transform type: {transform_type}")

    return matrix


def apply_transform_to_path(path_d: str, matrix: TransformMatrix) -> str:
    """
    Rewrite SVG d commands applying affine transform.
    Supports M/L/C/S/Q/T/H/V/A/Z in absolute or relative form.
    """
    path_pattern = re.compile(r"([a-zA-Z])([^a-zA-Z]*)")
    commands = path_pattern.findall(path_d)

    new_d = []
    current_point = (0.0, 0.0)

    for cmd, params_str in commands:
        params = [float(p) for p in re.split(r"[,\s]+", params_str) if p]

        if cmd in "MmLl":
            i = 0
            while i < len(params):
                x, y = params[i], params[i + 1]
                if cmd.islower():
                    x += current_point[0]
                    y += current_point[1]
                new_x, new_y = matrix.apply_to_point(x, y)
                new_d.append(f"{cmd}{new_x:.6f},{new_y:.6f}")
                current_point = (x, y)
                i += 2

        elif cmd in "CcSsQqTt":
            if cmd in "CcSs":
                param_count = 6
            else:
                param_count = 4

            i = 0
            while i < len(params):
                segment_params = params[i : i + param_count]
                transformed_params = []

                for j in range(0, len(segment_params), 2):
                    x, y = segment_params[j], segment_params[j + 1]
                    if cmd.islower():
                        x += current_point[0]
                        y += current_point[1]
                    new_x, new_y = matrix.apply_to_point(x, y)
                    transformed_params.extend([new_x, new_y])

                    if j == len(segment_params) - 2:
                        current_point = (x, y)

                params_joined = ",".join([f"{p:.6f}" for p in transformed_params])
                new_d.append(f"{cmd}{params_joined}")
                i += param_count

        elif cmd in "Aa":
            i = 0
            while i < len(params):
                rx, ry, x_axis_rotation, large_arc_flag, sweep_flag, x, y = params[
                    i : i + 7
                ]

                if cmd.islower():
                    x += current_point[0]
                    y += current_point[1]

                new_x, new_y = matrix.apply_to_point(x, y)

                new_d.append(
                    f"{cmd}{rx:.6f},{ry:.6f},{x_axis_rotation:.6f},{int(large_arc_flag)},{int(sweep_flag)},{new_x:.6f},{new_y:.6f}"
                )
                current_point = (x, y)
                i += 7

        elif cmd in "HhVv":
            is_horizontal = cmd in "Hh"
            for p in params:
                if is_horizontal:
                    x = p
                    y = current_point[1]
                    if cmd.islower():
                        x += current_point[0]
                else:
                    x = current_point[0]
                    y = p
                    if cmd.islower():
                        y += current_point[1]

                new_x, new_y = matrix.apply_to_point(x, y)
                new_d.append(
                    f"{cmd}{new_x:.6f}" if is_horizontal else f"{cmd}{new_y:.6f}"
                )
                current_point = (x, y)

        elif cmd == "Z" or cmd == "z":
            new_d.append(cmd)

        else:
            raise ValueError(f"Unsupported SVG command: {cmd}")

    return " ".join(new_d)
