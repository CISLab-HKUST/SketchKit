import numpy as np
from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point
from sketchkit.utils.fit_curve import fit_curve


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
