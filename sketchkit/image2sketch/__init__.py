"""Unified image-to-sketch interface and backend classes."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .generator import (
    Image2SketchResult,
    ImageInput,
    ImageSize,
    SketchGenerator,
)

_LAZY_BACKEND_EXPORTS = {
    "HEDModel": ("sketchkit.image2sketch.hed", "HEDModel"),
    "PhotoSketchModel": ("sketchkit.image2sketch.photosketch", "PhotoSketchModel"),
    "SwiftSketchModel": (
        "sketchkit.image2sketch.swiftsketch",
        "SwiftSketchModel",
    ),
    "AniLinesModel": ("sketchkit.image2sketch.anilines", "AniLinesModel"),
}


def __getattr__(name: str) -> Any:
    """Load a backend class only when it is explicitly requested."""
    try:
        module_name, class_name = _LAZY_BACKEND_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), class_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_BACKEND_EXPORTS))


__all__ = [
    "SketchGenerator",
    "Image2SketchResult",
    "ImageInput",
    "ImageSize",
    "HEDModel",
    "PhotoSketchModel",
    "SwiftSketchModel",
    "AniLinesModel",
]
