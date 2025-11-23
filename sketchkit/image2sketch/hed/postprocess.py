"""HED post-processing pipeline"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

try:  # optional dependency for thinning
    from skimage.morphology import thin as _skimage_thin  # type: ignore
except Exception:  # pragma: no cover
    _skimage_thin = None


@dataclass(frozen=True)
class PostprocessParams:
    """Configuration mirroring the defaults from ``PostprocessHED.m``."""

    threshold: float = 25.0 / 255.0
    small_edge: int = 5
    nms_radius: int = 1
    nms_border: int = 5
    nms_multiplier: float = 1.01
    smooth_radius: float = 4.0


DEFAULT_POSTPROCESS_PARAMS = PostprocessParams()


def apply_postprocess(
    edge_map: np.ndarray,
    *,
    params: PostprocessParams = DEFAULT_POSTPROCESS_PARAMS,
) -> np.ndarray:
    """Apply the official HED post-processing steps.

    Parameters
    ----------
    edge_map:
        Single-channel array in the range ``[0, 1]`` representing raw HED
        edge probabilities.
    params:
        Tunable parameters. Defaults replicate the MATLAB script.

    Returns
    -------
    ``np.ndarray``
        Float32 array in ``[0, 1]`` where 1 denotes white background and 0
        denotes edge strokes.
    """

    if edge_map.ndim != 2:
        raise ValueError("edge_map must be a single-channel 2D array.")

    working = np.clip(edge_map.astype(np.float32, copy=False), 0.0, 1.0)

    orientation = _compute_orientation(working, smooth_radius=params.smooth_radius)
    suppressed = _edge_nms(
        working,
        orientation,
        radius=params.nms_radius,
        border=params.nms_border,
        multiplier=params.nms_multiplier,
    )

    threshold = max(np.finfo(np.float32).eps, float(params.threshold))
    binary = (suppressed >= threshold).astype(np.uint8)
    thinned = _thin(binary)
    cleaned = _remove_small_objects(thinned, params.small_edge)

    return 1.0 - cleaned.astype(np.float32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_orientation(image: np.ndarray, *, smooth_radius: float) -> np.ndarray:
    smoothed = _conv_tri(image, smooth_radius)
    gx, gy = _gradient2(smoothed)
    gxx, _ = _gradient2(gx)
    gxy, gyy = _gradient2(gy)

    orientation = np.mod(
        np.arctan2(gyy * np.sign(-gxy), gxx + 1e-5),
        np.pi,
    ).astype(np.float32)
    return orientation


def _conv_tri(image: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return image

    if 0 < radius <= 1:
        p = 12.0 / (radius * (radius + 2.0)) - 2.0
        kernel_1d = np.array([1.0, p, 1.0], dtype=np.float32) / (2.0 + p)
        pad = 1
    else:
        r = int(round(radius))
        seq = np.concatenate(
            [np.arange(1, r + 1), np.array([r + 1]), np.arange(r, 0, -1)]
        ).astype(np.float32)
        kernel_1d = seq / float((r + 1) ** 2)
        pad = r

    if pad > 0:
        padded = np.pad(image, ((pad, pad), (pad, pad)), mode="symmetric")
    else:
        padded = image

    kernel = np.outer(kernel_1d, kernel_1d)
    filtered = cv2.filter2D(
        padded,
        -1,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
    )

    if pad > 0:
        filtered = filtered[pad:-pad, pad:-pad]
    return filtered.astype(np.float32, copy=False)


def _gradient2(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gx = np.empty_like(image, dtype=np.float32)
    gy = np.empty_like(image, dtype=np.float32)

    gx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) * 0.5
    gx[:, 0] = image[:, 1] - image[:, 0]
    gx[:, -1] = image[:, -1] - image[:, -2]

    gy[1:-1, :] = (image[2:, :] - image[:-2, :]) * 0.5
    gy[0, :] = image[1, :] - image[0, :]
    gy[-1, :] = image[-1, :] - image[-2, :]

    return gx, gy


def _edge_nms(
    edge_prob: np.ndarray,
    orientation: np.ndarray,
    *,
    radius: int,
    border: int,
    multiplier: float,
) -> np.ndarray:
    h, w = edge_prob.shape
    suppressed = edge_prob.astype(np.float32, copy=True)

    cos_o = np.cos(orientation).astype(np.float32, copy=False)
    sin_o = np.sin(orientation).astype(np.float32, copy=False)

    for y in range(h):
        for x in range(w):
            value = suppressed[y, x]
            if value <= 0.0:
                continue
            value *= multiplier
            cos_v = cos_o[y, x]
            sin_v = sin_o[y, x]

            for d in range(-radius, radius + 1):
                if d == 0:
                    continue
                sample_x = x + d * cos_v
                sample_y = y + d * sin_v
                neighbour = _bilinear(edge_prob, sample_x, sample_y)
                if value < neighbour:
                    suppressed[y, x] = 0.0
                    break

    border = min(border, w // 2, h // 2)
    if border > 0:
        for offset in range(border):
            factor = offset / float(border)
            suppressed[:, offset] *= factor
            suppressed[:, w - 1 - offset] *= factor
            suppressed[offset, :] *= factor
            suppressed[h - 1 - offset, :] *= factor

    return suppressed


def _bilinear(image: np.ndarray, x: float, y: float) -> float:
    h, w = image.shape
    x = np.clip(x, 0.0, w - 1.001)
    y = np.clip(y, 0.0, h - 1.001)

    x0 = int(x)
    y0 = int(y)
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)

    dx = x - x0
    dy = y - y0

    top = image[y0, x0] * (1.0 - dx) + image[y0, x1] * dx
    bottom = image[y1, x0] * (1.0 - dx) + image[y1, x1] * dx

    return float(top * (1.0 - dy) + bottom * dy)


def _thin(binary: np.ndarray) -> np.ndarray:
    if _skimage_thin is not None:
        thinned = _skimage_thin(binary.astype(bool))
        return thinned.astype(np.uint8)
    return _zhang_suen_thinning(binary)


def _zhang_suen_thinning(image: np.ndarray) -> np.ndarray:
    working = image.copy().astype(np.uint8)
    changed = True

    while changed:
        changed = False
        to_remove = []

        for y in range(1, working.shape[0] - 1):
            for x in range(1, working.shape[1] - 1):
                if working[y, x] == 0:
                    continue
                neighbours = _neighbourhood(working, x, y)
                transitions = sum(
                    (neighbours[i] == 0 and neighbours[(i + 1) % 8] == 1)
                    for i in range(8)
                )
                count = sum(neighbours)
                if (
                    2 <= count <= 6
                    and transitions == 1
                    and neighbours[0] * neighbours[2] * neighbours[4] == 0
                    and neighbours[2] * neighbours[4] * neighbours[6] == 0
                ):
                    to_remove.append((y, x))

        if to_remove:
            changed = True
            for y, x in to_remove:
                working[y, x] = 0

        to_remove = []
        for y in range(1, working.shape[0] - 1):
            for x in range(1, working.shape[1] - 1):
                if working[y, x] == 0:
                    continue
                neighbours = _neighbourhood(working, x, y)
                transitions = sum(
                    (neighbours[i] == 0 and neighbours[(i + 1) % 8] == 1)
                    for i in range(8)
                )
                count = sum(neighbours)
                if (
                    2 <= count <= 6
                    and transitions == 1
                    and neighbours[0] * neighbours[2] * neighbours[6] == 0
                    and neighbours[0] * neighbours[4] * neighbours[6] == 0
                ):
                    to_remove.append((y, x))

        if to_remove:
            changed = True
            for y, x in to_remove:
                working[y, x] = 0

    return working


def _neighbourhood(image: np.ndarray, x: int, y: int) -> Tuple[int, ...]:
    return (
        image[y - 1, x],
        image[y - 1, x + 1],
        image[y, x + 1],
        image[y + 1, x + 1],
        image[y + 1, x],
        image[y + 1, x - 1],
        image[y, x - 1],
        image[y - 1, x - 1],
    )


def _remove_small_objects(image: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return image

    num_labels, labels = cv2.connectedComponents(image, connectivity=8)
    if num_labels <= 1:
        return image

    counts = np.bincount(labels.ravel())
    valid = counts >= max(1, int(min_size))
    valid[0] = False

    mask = valid[labels]
    cleaned = np.zeros_like(image, dtype=np.uint8)
    cleaned[mask] = 1
    return cleaned


__all__ = ["PostprocessParams", "DEFAULT_POSTPROCESS_PARAMS", "apply_postprocess"]
