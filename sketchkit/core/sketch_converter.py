"""
Unified protocol and registry for sketch conversion and colorization.

This module provides the common infrastructure for all sketch-to-image
and sketch-colorization methods, following the same pattern as image2sketch.
"""

from __future__ import annotations
from typing import Callable, Dict, MutableMapping, Optional, Protocol, Sequence, Tuple, Union, cast, Any
import threading
import numpy as np
from PIL import Image
from sketchkit.core.sketch import Sketch
from sketchkit.renderer.cairo_renderer import CairoRenderer

# Type definitions for public API
SketchInput = Union[Sketch, Sequence[Sketch]]
ConversionResult = Union[Image.Image, list[Image.Image]]


class SketchConverter(Protocol):
    """Protocol describing models capable of converting sketches to images or colorizing them."""
    
    def convert(
        self,
        sketch: SketchInput,
        prompt: str = "",
        *,
        size: Union[int, Tuple[int, int]] = 512,
        **kwargs: Any,
    ) -> ConversionResult:
        """Convert sketch(es) to image(s) or colorized sketch(es)."""


# Registry infrastructure (similar to image2sketch)
MethodFactory = Callable[..., SketchConverter]
_METHOD_REGISTRY: MutableMapping[str, MethodFactory] = {}
_DEFAULT_INSTANCES: Dict[str, SketchConverter] = {}
_REGISTRY_LOCK = threading.Lock()


def register_method(name: str, factory: MethodFactory) -> None:
    """Register a new sketch conversion/colorization method."""
    with _REGISTRY_LOCK:
        if name in _METHOD_REGISTRY:
            raise ValueError(f"Sketch conversion method '{name}' is already registered.")
        _METHOD_REGISTRY[name] = factory


def get_method(name: str) -> MethodFactory:
    """Return the factory registered for the given method name."""
    try:
        return _METHOD_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_METHOD_REGISTRY)) or "none"
        raise ValueError(
            f"Unknown sketch conversion method '{name}'. Available methods: {available}."
        ) from exc


def available_methods() -> list[str]:
    """List the names of registered sketch conversion methods."""
    return list(_METHOD_REGISTRY.keys())


def create_converter(method: str, **kwargs: Any) -> SketchConverter:
    """Instantiate a converter for the requested method."""
    factory = get_method(method)
    instance = factory(**kwargs)
    if not hasattr(instance, "convert"):
        raise TypeError(
            f"Sketch conversion method '{method}' produced an object without a 'convert' method."
        )
    return cast(SketchConverter, instance)


def _get_default_converter(method: str) -> SketchConverter:
    """Get or create the default singleton instance for a method."""
    with _REGISTRY_LOCK:
        if method not in _DEFAULT_INSTANCES:
            _DEFAULT_INSTANCES[method] = create_converter(method)
        return _DEFAULT_INSTANCES[method]


# Base class with common functionality
class BaseSketchConverter:
    """Base class providing common functionality for all sketch converters."""
    
    def sketch_to_raster(
        self,
        sketch: Sketch,
        size: Union[int, Tuple[int, int]] = 512,
        background_color: Tuple[float, float, float] = (1, 1, 1),
        stroke_width: int = 3,
        fit_canvas: bool = True,
    ) -> np.ndarray:
        """Convert sketch to rasterized image using CairoRenderer."""
        if isinstance(size, int):
            canvas_size = size
        else:
            canvas_size = max(size)
            
        renderer = CairoRenderer(
            canvas_size=canvas_size,
            canvas_color=background_color
        )
        
        raster_image = renderer.render(sketch, width=stroke_width, fit_canvas=fit_canvas)
        
        if isinstance(size, tuple) and size[0] != size[1]:
            pil_image = Image.fromarray(raster_image)
            pil_image = pil_image.resize(size, Image.Resampling.LANCZOS)
            raster_image = np.array(pil_image)
        
        return raster_image
    
    def _prepare_control_image(
        self, 
        raster_sketch: np.ndarray, 
        method_type: str = "default"
    ) -> Image.Image:
        """Prepare control image from rasterized sketch."""
        if len(raster_sketch.shape) == 3:
            if raster_sketch.shape[2] == 3:
                pil_image = Image.fromarray(raster_sketch).convert("RGB")
            else:
                pil_image = Image.fromarray(raster_sketch[:, :, 0]).convert("L")
        else:
            pil_image = Image.fromarray(raster_sketch).convert("L")
        
        return pil_image


# Unified entry point function
def convert_sketch(
    sketch: SketchInput,
    prompt: str = "",
    *,
    method: str,
    size: Union[int, Tuple[int, int]] = 512,
    **model_kwargs: Any,
) -> ConversionResult:
    """Unified entry point for sketch conversion and colorization."""
    
    resolved_converter: SketchConverter
    if model_kwargs:
        resolved_converter = create_converter(method, **model_kwargs)
    else:
        resolved_converter = _get_default_converter(method)
    
    return resolved_converter.convert(sketch, prompt, size=size, **model_kwargs)


__all__ = [
    "SketchConverter",
    "BaseSketchConverter",
    "convert_sketch",
    "available_methods",
    "create_converter",
    "get_method",
    "register_method",
]