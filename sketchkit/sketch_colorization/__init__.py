"""
Sketch Colorization

This module provides a unified API for colorizing vector sketches.
"""

from __future__ import annotations
from typing import Union, Tuple, Optional, List, Any
from PIL import Image

from sketchkit.core.sketch_converter import (
    SketchConverter,
    convert_sketch,
    available_methods,
    create_converter,
    get_method,
    register_method,
)
from sketchkit.core.sketch import Sketch
from .sketch_colorization import SketchColorizationConverter

# Register all colorization methods
register_method("controlnet_lineart", lambda **kwargs: SketchColorizationConverter(method="controlnet_lineart", **kwargs))
register_method("controlnet_scribble", lambda **kwargs: SketchColorizationConverter(method="controlnet_scribble", **kwargs))

# Create default instances
controlnet_lineart = create_converter("controlnet_lineart")
controlnet_scribble = create_converter("controlnet_scribble")

# Backward compatibility function
def colorize_sketch(
    sketch: Sketch,
    prompt: str = "vibrant colors, anime style",
    method: str = "controlnet_lineart",
    size: Union[int, Tuple[int, int]] = 512,
    output_path: Optional[str] = None,
    reference_images: Optional[List[Image.Image]] = None,
    **kwargs: Any,
) -> Image.Image:
    """Legacy function for backward compatibility.
    
    Convert sketch to colorized image using AI methods.
    
    Args:
        sketch: The Sketch object to convert.
        prompt: Text description guiding the colorization. Defaults to 
            "vibrant colors, anime style".
        method: Colorization method to use. Choose from 'controlnet_lineart' or
            'controlnet_scribble'. Defaults to 'controlnet_lineart'.
        size: Output image size. If integer, creates square image. If tuple,
            creates rectangular image (width, height). Defaults to 512.
        output_path: Optional path to save the colorized image. If provided,
            the image will be saved to this location.
        reference_images: Optional list of reference images for color guidance.
            Only used with MagicColor method.
    
    Returns:
        Generated colorized image as PIL Image.
    
    Example:
        >>> # Simple colorization with default parameters
        >>> image = colorize_sketch(sketch, prompt="vibrant anime colors")
        >>>
        >>> # Custom colorization with specific parameters
        >>> image = colorize_sketch(
        ...     sketch, 
        ...     method='controlnet_lineart',
        ...     prompt="watercolor painting style",
        ...     size=(768, 512),
        ...     output_path="colorized.png",
        ...     num_steps=30,
        ...     guidance_scale=8.0
        ... )
        >>>
        >>> # MagicColor with reference images
        >>> refs = [Image.open("ref1.jpg"), Image.open("ref2.jpg")]
        >>> image = colorize_sketch(
        ...     sketch,
        ...     method='magiccolor',
        ...     prompt="consistent colors",
        ...     reference_images=refs
        ... )
    """
    if reference_images and method == "magiccolor":
        kwargs["reference_images"] = reference_images
    
    image = convert_sketch(
        sketch,
        prompt=prompt,
        method=method,
        size=size,
        **kwargs
    )
    
    if output_path and isinstance(image, Image.Image):
        image.save(output_path)
        print(f"Image saved to: {output_path}")
    
    return image


__all__ = [
    "SketchColorizationConverter",
    "colorize_sketch",
    "convert_sketch",
    "controlnet_lineart",
    "controlnet_scribble",
    "magiccolor",
    "available_methods",
    "create_converter",
    "get_method",
    "register_method",
]