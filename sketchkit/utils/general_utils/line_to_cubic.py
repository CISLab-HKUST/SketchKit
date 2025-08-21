import numpy as np


def line_to_cubic(line):
    """
    Args:
        line: In shape (2, 2), including two points.
        
    Returns:
        cubic: In shape (4, 2), including a starting point, two control points, and an end point.

    """
    p_start, p_end = line[0], line[1]
    start_x, start_y = p_start[0], p_start[1]
    end_x, end_y = p_end[0], p_end[1]
    ctrl1_x = 2.0 / 3.0 * start_x + 1.0 / 3.0 * end_x
    ctrl1_y = 2.0 / 3.0 * start_y + 1.0 / 3.0 * end_y
    ctrl2_x = 1.0 / 3.0 * start_x + 2.0 / 3.0 * end_x
    ctrl2_y = 1.0 / 3.0 * start_y + 2.0 / 3.0 * end_y

    ctrl1 = np.stack([ctrl1_x, ctrl1_y])
    ctrl2 = np.stack([ctrl2_x, ctrl2_y])
    cubic = np.stack([p_start, ctrl1, ctrl2, p_end], axis=0)  # (4, 2)
    return cubic