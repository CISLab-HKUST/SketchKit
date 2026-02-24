"""image2sketch model integrations.

This module exposes a unified API for converting raster images into vector
sketches.  It mirrors the layout of :mod:`sketchkit.vectorization` so that
new ``image2sketch`` methods can be added incrementally without changing the
public entry points.
"""

from __future__ import annotations

from typing import Callable, Dict, MutableMapping, Optional, Protocol, Sequence, Tuple, Union, cast

import numpy as np
from PIL import Image

from sketchkit.core import Sketch
from .hed import HEDModel
from .photosketch import PhotoSketchModel
from .swiftsketch.generate import SwiftSketchModel

ImageInput = Union[Image.Image, np.ndarray, str]
Image2SketchResult = Union[Sketch, Image.Image, list[Sketch], list[Image.Image]]
ImageSize = Optional[Union[int, Tuple[int, int]]]


class Image2SketchModel(Protocol):
    """Protocol describing models capable of turning images into sketches."""

    def generate(
        self,
        image: Union[ImageInput, Sequence[ImageInput]],
        *,
        size: Union[int, Tuple[int, int]] = 512,
    ) -> Image2SketchResult:
        """Produce sketch or raster results for the provided image input(s)."""


MethodFactory = Callable[..., Image2SketchModel]
_METHOD_REGISTRY: MutableMapping[str, MethodFactory] = {}
_DEFAULT_INSTANCES: Dict[str, Image2SketchModel] = {}
_DEFAULT_METHOD = "swiftsketch"
_METHOD_ALIASES: Dict[str, str] = {
    "hed": "hed",
    "HED": "hed",
    "photosketch": "photosketch",
    "PhotoSketch": "photosketch",
    "swiftsketch": "swiftsketch",
    "SwiftSketch": "swiftsketch",
}


def register_method(name: str, factory: MethodFactory) -> None:
    """Register a new ``image2sketch`` method.

    Parameters
    ----------
    name:
        Canonical name for the method. Names are case sensitive.
    factory:
        Callable returning an object that implements :class:`Image2SketchModel`.
    """

    if name in _METHOD_REGISTRY:
        raise ValueError(f"image2sketch method '{name}' is already registered.")
    _METHOD_REGISTRY[name] = factory


def get_method(name: str) -> MethodFactory:
    """Return the factory registered for the given method name."""

    resolved_name = _resolve_method_name(name)
    try:
        return _METHOD_REGISTRY[resolved_name]
    except KeyError as exc:
        available = ", ".join(sorted(_METHOD_REGISTRY)) or "none"
        raise ValueError(
            f"Unknown image2sketch method '{name}'. Available methods: {available}."
        ) from exc


def available_methods() -> list[str]:
    """List the names of registered ``image2sketch`` methods."""

    return list(_METHOD_REGISTRY.keys())


def create_model(method: str = _DEFAULT_METHOD, **kwargs) -> Image2SketchModel:
    """Instantiate a model for the requested method."""

    resolved_method = _resolve_method_name(method)
    factory = get_method(resolved_method)
    instance = factory(**kwargs)
    if not hasattr(instance, "generate"):
        raise TypeError(
            f"image2sketch method '{resolved_method}' produced an object without a 'generate' method."
        )
    return cast(Image2SketchModel, instance)


def _get_default_model(method: str = _DEFAULT_METHOD) -> Image2SketchModel:
    resolved_method = _resolve_method_name(method)
    if resolved_method not in _DEFAULT_INSTANCES:
        _DEFAULT_INSTANCES[resolved_method] = create_model(resolved_method)
    return _DEFAULT_INSTANCES[resolved_method]


def _resolve_method_name(method: str) -> str:
    if method in _METHOD_REGISTRY:
        return method
    if method in _METHOD_ALIASES:
        return _METHOD_ALIASES[method]

    lowered = method.lower()
    if lowered in _METHOD_REGISTRY:
        return lowered
    if lowered in _METHOD_ALIASES:
        return _METHOD_ALIASES[lowered]
    return method


class SketchGenerator:
    """A class for image-to-sketch generation."""

    def __init__(self, method: str = "SwiftSketch", **model_kwargs: object):
        """Initialise the SketchGenerator.

        Args:
            method (str, optional): The image2sketch method name.
            **model_kwargs: Method-specific model arguments.
        """
        self.method = method
        self.model_kwargs = model_kwargs
        self.generator_method = create_model(method, **model_kwargs)

    def run(
        self,
        input: Union[ImageInput, Sequence[ImageInput]],
        size: ImageSize = None,
    ) -> Image2SketchResult:
        """Generate sketch outputs from an input image."""
        if not hasattr(self.generator_method, "generate"):
            raise TypeError("SketchGenerator backend must implement a 'generate' method.")
        return self.generator_method.generate(input, size=size)


def image2sketch(
    image: Union[ImageInput, Sequence[ImageInput]],
    *,
    method: str = _DEFAULT_METHOD,
    size: ImageSize = None,
    **model_kwargs: object,
) -> Image2SketchResult:
    """Generate sketch or raster outputs for the provided image using the requested method."""

    resolved_model: Image2SketchModel
    if model_kwargs:
        resolved_model = create_model(method, **model_kwargs)
    else:
        resolved_model = _get_default_model(method)

    if not hasattr(resolved_model, "generate"):
        raise TypeError("image2sketch requires the model to expose a 'generate' method.")

    return resolved_model.generate(image, size=size)


register_method("hed", HEDModel)
register_method("photosketch", PhotoSketchModel)
register_method("swiftsketch", SwiftSketchModel)
hed = cast(HEDModel, _get_default_model("hed"))
photosketch = cast(PhotoSketchModel, _get_default_model("photosketch"))
swiftsketch = cast(SwiftSketchModel, _get_default_model("swiftsketch"))

__all__ = [
    "SketchGenerator",
    "Image2SketchModel",
    "image2sketch",
    "available_methods",
    "create_model",
    "get_method",
    "register_method",
    "hed",
    "HEDModel",
    "photosketch",
    "PhotoSketchModel",
    "swiftsketch",
    "SwiftSketchModel",
]
