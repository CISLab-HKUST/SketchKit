import numpy as np


def intrinsics_from_fov_xy(
    fov_x: float | np.ndarray, fov_y: float | np.ndarray
) -> np.ndarray:
    """
    Get OpenCV intrinsics matrix from field of view in x and y axis

    Args:
    fov_x (float | np.ndarray): field of view in x axis
    fov_y (float | np.ndarray): field of view in y axis

    Returns:
    (np.ndarray): [3, 3] OpenCV intrinsics matrix
    """
    focal_x = 0.5 / np.tan(fov_x / 2)
    focal_y = 0.5 / np.tan(fov_y / 2)
    cx = cy = 0.5
    return np.array(
        [
            [focal_x, 0, cx],
            [0, focal_y, cy],
            [0, 0, 1],
        ]
    )
