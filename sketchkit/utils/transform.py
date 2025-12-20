import numpy as np
from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point
from sketchkit.utils.fit_curve import fit_curve
import json


def line_to_cubic(line) -> Curve:
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


def convert_to_curve(sketch: Sketch, maxError=4.0):
    """Convert a Sketch with polylines to one with Bezier curves.

    Args:
        sketch (Sketch): Input sketch with polylines.
        maxError (float, optional): Maximum fitting error. Defaults to 4.0.

    Returns:
        Sketch: New sketch with Bezier curves.
    """
    curve_paths = []
    for path in sketch.paths:
        if path.curves is None or len(path.curves) < 1:
            continue
        # print([[[curve.p_start.x, curve.p_start.y], [curve.p_end.x, curve.p_end.y]] for curve in path.curves])
        polyline_path = []
        for curve in path.curves:
            if len(polyline_path) == 0:
                polyline_path.append([curve.p_start.x, curve.p_start.y])
            elif (
                np.linalg.norm(
                    np.array(polyline_path[-1])
                    - np.array([curve.p_start.x, curve.p_start.y])
                )
                > 1
            ):
                polyline_path.append([curve.p_start.x, curve.p_start.y])
        if (
            np.linalg.norm(
                np.array(polyline_path[-1])
                - np.array([path.curves[-1].p_end.x, path.curves[-1].p_end.y])
            )
            > 1
        ):
            polyline_path.append([path.curves[-1].p_end.x, path.curves[-1].p_end.y])
        print(polyline_path)
        # if len(polyline_path) < 4:
        #     continue
        curve_path = fit_curve(np.array(polyline_path), maxError)
        curves = [
            Curve(
                Vertex(pt[0][0], pt[0][1]),
                Vertex(pt[3][0], pt[3][1]),
                Point(pt[1][0], pt[1][1]),
                Point(pt[2][0], pt[2][1]),
            )
            for pt in curve_path
        ]
        curve_paths.append(Path(curves=curves))
    new_sketch = Sketch(paths=curve_paths)
    return new_sketch

def hex_to_float_rgb(hex_color: str):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError("Wrong hex color format.")
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def load_sketch(json_path: str) -> Sketch:
    with open(json_path, 'r') as f:
        data = json.load(f)
        paths = []
        for path_dict in data:
            txy = path_dict['txy']
            pressure = path_dict['p']
            color = hex_to_float_rgb(path_dict['color'])
            width = int(path_dict['width'])
            assert len(pressure) * 3 == len(txy), "Pressure and txy must have the same length."
            curves = []
            for i in range(len(pressure) - 1):
                p0 = txy[3 * i + 1: 3 * i + 3]
                p1 = txy[3 * i + 4: 3 * i + 6]
                line = np.array([p0, p1])
                segment = line_to_cubic(line)
                segment.p_start.pressure = pressure[i]
                segment.p_end.pressure = pressure[i + 1]
                curves.append(segment)
            path = Path(curves=curves)
            path.color = color
            path.thickness = width
            paths.append(path)
        sketch = Sketch(paths=paths)
        return sketch
    
if __name__ == '__main__':
    json_path = 'media/eg_drawing (3)_drawing (1).json'
    sketch = load_sketch(json_path)
    svg_str = sketch.to_svg(size=800, is_polyline=True)
    with open('media/eg_drawing (3)_drawing (1).svg', 'w') as f:
        f.write(svg_str)