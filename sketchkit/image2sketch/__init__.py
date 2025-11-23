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

    try:
        return _METHOD_REGISTRY[name]
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

    factory = get_method(method)
    instance = factory(**kwargs)
    if not hasattr(instance, "generate"):
        raise TypeError(
            f"image2sketch method '{method}' produced an object without a 'generate' method."
        )
    return cast(Image2SketchModel, instance)


def _get_default_model(method: str = _DEFAULT_METHOD) -> Image2SketchModel:
    if method not in _DEFAULT_INSTANCES:
        _DEFAULT_INSTANCES[method] = create_model(method)
    return _DEFAULT_INSTANCES[method]


def image2sketch(
    image: Union[ImageInput, Sequence[ImageInput]],
    *,
    method: str = _DEFAULT_METHOD,
    size: Union[int, Tuple[int, int]] = 512,
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
