"""
Sketch-to-Image Conversions

This module provides a unified API for converting vector sketches into realistic images.
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
from .sketch2img import Sketch2ImageConverter

# Register all sketch2image methods
register_method("controlnet", lambda **kwargs: Sketch2ImageConverter(method="controlnet", **kwargs))
register_method("t2i_adapter", lambda **kwargs: Sketch2ImageConverter(method="t2i_adapter", **kwargs))
register_method("pix2pix", lambda **kwargs: Sketch2ImageConverter(method="pix2pix", **kwargs))

# Create default instances for direct access
controlnet = create_converter("controlnet")
t2i_adapter = create_converter("t2i_adapter")
pix2pix = create_converter("pix2pix")

# Backward compatibility function
def sketch2img(
    sketch: Sketch,
    prompt: str = "a realistic photo",
    method: str = "controlnet",
    size: Union[int, Tuple[int, int]] = 512,
    output_path: Optional[str] = None,
    **kwargs: Any,
) -> Image.Image:
    """Legacy function for backward compatibility.
    
    Convert sketch to realistic image using AI methods.
    
    Args:
        sketch: The Sketch object to convert.
        prompt: Text description guiding the image generation. Defaults to 
            "a realistic photo".
        method: Conversion method to use. Choose from 'controlnet', 't2i_adapter',
            or 'pix2pix'. Defaults to 'controlnet'.
        size: Output image size. If integer, creates square image. If tuple,
            creates rectangular image (width, height). Defaults to 512.
        output_path: Optional path to save the generated image. If provided,
            the image will be saved to this location.
    
    Returns:
        Generated realistic image as PIL Image.
    
    Example:
        >>> # Simple conversion with default parameters
        >>> image = sketch2img(sketch, "a beautiful watercolor painting")
        >>>
        >>> # Custom conversion with specific parameters
        >>> image = sketch2img(
        ...     sketch, 
        ...     method='controlnet',
        ...     prompt="a photorealistic landscape",
        ...     size=(768, 512),
        ...     output_path="output.png",
        ...     num_steps=30,
        ...     guidance_scale=8.0
        ... )
    """
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
    "Sketch2ImageConverter",
    "sketch2img",
    "convert_sketch",
    "controlnet",
    "t2i_adapter",
    "pix2pix",
    "available_methods",
    "create_converter",
    "get_method",
    "register_method",
]