import numpy as np
from sketchkit.core.sketch import Path, Curve, Vertex, Point


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
