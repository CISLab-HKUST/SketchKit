from sketchkit.core.sketch import Sketch, Vertex, Point
import numpy as np

def determine_direction(prev: Vertex, start: Vertex, end: Vertex) -> bool:
    """
    Determine the direction of the line segment from prev to end.
    Args:
        prev (Point): The previous point.
        start (Point): The start point of the stroke.
        end (Point): The end point of the stroke.

    Returns:
        bool: True if the direction is start to end, False otherwise.
    """

    # 统一转成 float64 防止类型问题
    p0 = np.array([start.x, start.y], dtype=np.float64)
    p1 = np.array([end.x, end.y], dtype=np.float64)

    # 情况 1：有上一笔 → 优先“连续纸面接触”
    if prev is not None:
        pp = np.array([prev.x, prev.y], dtype=np.float64)
        d0 = np.linalg.norm(p0 - pp)
        d1 = np.linalg.norm(p1 - pp)
        if d0 - d1 < -1e-1:
            return True
        elif d0 - d1 > 1e-1:
            return False

    # 情况 2：第一笔 → 按默认习惯
    vec = p1 - p0
    dx, dy = vec
    tan = abs(dy / (dx + 1e-6))
    if tan > np.tan(np.pi / 12):
        # top to bottom
        return True if dy < 0 else False
    else:
        # left to right
        return True if dx > 0 else False

    