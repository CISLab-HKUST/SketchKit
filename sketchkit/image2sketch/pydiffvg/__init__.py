"""A lightweight subset of the pydiffvg API used by SwiftSketch.

This stub implements just enough functionality for generating SVG content
from cubic Bézier control points.  It is *not* a full replacement for the
original library but provides a pure Python fallback that keeps the runtime
self contained.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path as _Path
from typing import Sequence

import torch


@dataclass
class Path:
    num_control_points: torch.Tensor
    points: torch.Tensor
    stroke_width: torch.Tensor
    is_closed: bool


@dataclass
class ShapeGroup:
    shape_ids: torch.Tensor
    fill_color: torch.Tensor | None
    stroke_color: torch.Tensor


def save_svg(path: str, width: int, height: int, shapes: Sequence[Path], shape_groups: Sequence[ShapeGroup]) -> None:
    svg_content = _build_svg(width, height, shapes, shape_groups)
    _Path(path).write_text(svg_content, encoding="utf-8")


class RenderFunction:
    @staticmethod
    def serialize_scene(width: int, height: int, shapes: Sequence[Path], shape_groups: Sequence[ShapeGroup]):
        return width, height, list(shapes), list(shape_groups)

    @staticmethod
    def apply(width: int, height: int, samples_x: int, samples_y: int, seed: int, background, *scene_args):
        # Render a blank white RGBA tensor. Downstream code converts SVG to an image.
        return torch.ones(height, width, 4, dtype=torch.float32)


def _build_svg(width: int, height: int, shapes: Sequence[Path], shape_groups: Sequence[ShapeGroup]) -> str:
    header = f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"
    paths: list[str] = [header]

    for shape, group in zip(shapes, shape_groups):
        points = shape.points.reshape(-1, 2).cpu().numpy()
        if points.shape[0] < 4:
            continue
        command = [f"M {points[0][0]} {points[0][1]}"]
        for i in range(1, points.shape[0], 3):
            if i + 2 >= points.shape[0]:
                break
            p1, p2, p3 = points[i : i + 3]
            command.append(f"C {p1[0]} {p1[1]}, {p2[0]} {p2[1]}, {p3[0]} {p3[1]}")
        path_data = " ".join(command)
        color = _rgba_to_hex(group.stroke_color)
        paths.append(f"<path d='{path_data}' fill='none' stroke='{color}' stroke-width='{shape.stroke_width.item()}" "' />")

    paths.append("</svg>")
    return "\n".join(paths)


def _rgba_to_hex(color: torch.Tensor) -> str:
    rgba = color.detach().cpu().tolist()
    r, g, b, a = [max(0.0, min(1.0, float(c))) for c in rgba]
    r_i, g_i, b_i = int(r * 255), int(g * 255), int(b * 255)
    if a >= 1.0:
        return f"#{r_i:02x}{g_i:02x}{b_i:02x}"
    return f"rgba({r_i},{g_i},{b_i},{a:.3f})"
