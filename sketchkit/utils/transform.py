import numpy as np
from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point
from sketchkit.utils.fitCurves import fitCurve

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


def convert_to_curve(sketch: Sketch, datasetName: str, maxError=4.0):
    """Convert a Sketch with polylines to one with Bezier curves.

    Args:
        sketch (Sketch): Input sketch with polylines.
        maxError (float, optional): Maximum fitting error. Defaults to 4.0.

    Returns:
        Sketch: New sketch with Bezier curves.
    """
    if datasetName not in ["DifferSketching", "FSCOCO", "QuickDraw", "SketchXPRIS", "TracingVsFreehand"]:
        raise ValueError(f"Dataset {datasetName} not supported for polyline to curve conversion.")
    curve_paths = []
    for path in sketch.paths:
        if path.curves is None or len(path.curves) < 1:
            continue
        print([[[curve.p_start.x, curve.p_start.y], [curve.end.x, curve.end.y]] for curve in path.curves])
        polyline_path = [[curve.p_start.x, curve.p_start.y] for curve in path.curves]
        polyline_path.append([path.curves[-1].p_end.x, path.curves[-1].p_end.y])
        if len(polyline_path) < 4:
            continue
        curve_path = fitCurve(np.array(polyline_path), maxError)
        curves = [Curve(
            Vertex(pt[0][0], pt[0][1]),
            Vertex(pt[3][0], pt[3][1]),
            Point(pt[1][0], pt[1][1]),
            Point(pt[2][0], pt[2][1]),
        ) for pt in curve_path]
        curve_paths.append(Path(curves=curves))
    new_sketch = Sketch(paths=curve_paths)
    return new_sketch