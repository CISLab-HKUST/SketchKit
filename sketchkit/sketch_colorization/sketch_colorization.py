"""
SketchColorizationConverter implementation using the unified architecture.
"""

from typing import Union, Tuple, List, Optional, Dict, Any
import torch
from PIL import Image
import numpy as np

from sketchkit.core.sketch import Sketch
from sketchkit.core.sketch_converter import BaseSketchConverter, SketchConverter
from .models import (
    ControlNetLineartModel,
    ControlNetScribbleModel,
)


class SketchColorizationConverter(BaseSketchConverter, SketchConverter):
    """Convert sketches to colorized images using various AI methods.
    
    This class provides multiple AI-powered methods to colorize sketches.
    Each method is initialized with specific model parameters and can be 
    reused for multiple conversions.
    
    Supported methods:
        - controlnet_lineart: Uses ControlNet with lineart conditioning
        - controlnet_scribble: Uses ControlNet with scribble conditioning
        - magiccolor: Uses reference images for color guidance
    
    Example:
        >>> # Initialize different converters
        >>> lineart_converter = SketchColorizationConverter(method='controlnet_lineart')
        >>> magiccolor_converter = SketchColorizationConverter(method='magiccolor')
        >>>
        >>> # Colorize sketches
        >>> lineart_image = lineart_converter.convert(sketch, "vibrant anime colors")
        >>> magic_image = magiccolor_converter.convert(sketch, "consistent colors", reference_images=refs)
    """
    
    def __init__(
        self, 
        method: str = "controlnet_lineart", 
        **kwargs: Any
    ):
        """Initialize converter with specified method and parameters.
        
        Args:
            method: The colorization method to use. Choose from:
                - 'controlnet_lineart': ControlNet with lineart conditioning
                - 'controlnet_scribble': ControlNet with scribble conditioning
                - 'magiccolor': Reference image-based colorization
        
        Keyword Args for ControlNet:
            controlnet_model (str): ControlNet model name. Defaults vary by method.
            sd_model (str): Stable Diffusion base model. Defaults to 
                "runwayml/stable-diffusion-v1-5".
            torch_dtype: Data type for model weights. Defaults to torch.float16.
            
        Keyword Args for MagicColor:
            magiccolor_root (str): Path to MagicColor installation directory.
        
        Raises:
            ValueError: If an unsupported method is specified.
        """
        super().__init__()
        self.method = method
        self.model_params = kwargs
        self._pipeline = None
        
        self._model_classes = {
            "controlnet_lineart": ControlNetLineartModel,
            "controlnet_scribble": ControlNetScribbleModel,
        }
        
        if method not in self._model_classes:
            raise ValueError(f"Unsupported method: {method}")
    
    def convert(
        self,
        sketch: Union[Sketch, List[Sketch]],
        prompt: str = "vibrant colors, anime style",
        *,
        size: Union[int, Tuple[int, int]] = 512,
        **kwargs: Any,
    ) -> Union[Image.Image, List[Image.Image]]:
        """Convert sketch(es) to colorized image(s) using the initialized method.
        
        This is the main conversion method that takes a vector sketch and
        transforms it into a colorized image using the selected AI method.
        
        Args:
            sketch: The Sketch object(s) to convert.
            prompt: Text description guiding the colorization. This should
                describe the desired colors and style. Defaults to 
                "vibrant colors, anime style".
            size: Output image size. If integer, creates square image. If tuple,
                creates rectangular image (width, height). Defaults to 512.
        
        Keyword Args for Generation Control:
            stroke_width (int): Stroke width for rasterization. Defaults to 2.
            fit_canvas (bool): Whether to fit sketch to canvas. Defaults to True.
            num_steps (int): Number of inference steps. Defaults to 20.
            guidance_scale (float): CFG scale for text guidance. Defaults to 7.5.
            controlnet_scale (float): ControlNet conditioning scale. Defaults to 1.0.
            negative_prompt (str): Negative prompt to avoid unwanted features.
                Defaults to "blurry, low quality, distorted".
            seed (int): Random seed for reproducibility. Optional.
            reference_images: List of reference images for MagicColor method.
        
        Returns:
            Generated colorized image(s) as PIL Image(s).
        
        Raises:
            RuntimeError: If the pipeline failed to initialize.
            ValueError: If method is unsupported.
        """
        if self._pipeline is None:
            model_class = self._model_classes[self.method]
            self._pipeline = model_class(**self.model_params)
        
        if isinstance(sketch, list):
            return [self._convert_single(s, prompt, size, **kwargs) for s in sketch]
        else:
            return self._convert_single(sketch, prompt, size, **kwargs)
    
    def _convert_single(
        self,
        sketch: Sketch,
        prompt: str,
        size: Union[int, Tuple[int, int]],
        **kwargs: Any,
    ) -> Image.Image:
        """Convert a single sketch to colorized image."""
        raster_sketch = self.sketch_to_raster(
            sketch,
            size=size,
            stroke_width=kwargs.get('stroke_width', 2),
            fit_canvas=kwargs.get('fit_canvas', True),
        )
        
        if self.method.startswith('controlnet'):
            return self._controlnet_convert(raster_sketch, prompt, size, **kwargs)
        else:  # magiccolor
            return self._magiccolor_convert(raster_sketch, prompt, **kwargs)
    
    def _controlnet_convert(
        self, 
        raster_sketch: np.ndarray, 
        prompt: str,
        size: Union[int, Tuple[int, int]],
        **kwargs: Any
    ) -> Image.Image:
        """ControlNet conversion implementation."""
        control_image = self._prepare_control_image(raster_sketch, self.method)
        
        # Setup generator for reproducibility
        generator = None
        if 'seed' in kwargs:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(kwargs['seed'])
        
        # Generate colorized image using ControlNet pipeline
        return self._pipeline.generate(
            control_image=control_image,
            prompt=prompt,
            size=size,
            negative_prompt=kwargs.get('negative_prompt', "blurry, low quality, distorted"),
            generator=generator,
            **{k: v for k, v in kwargs.items() if k not in ['stroke_width', 'fit_canvas', 'seed', 'reference_images']}
        )
    
    def _magiccolor_convert(
        self, 
        raster_sketch: np.ndarray, 
        prompt: str,
        **kwargs: Any
    ) -> Image.Image:
        """MagicColor conversion implementation."""
        reference_images = kwargs.get('reference_images')
        if reference_images is None or len(reference_images) == 0:
            raise ValueError("MagicColor requires at least one reference image")
        
        # Use MagicColor model to perform colorization
        return self._pipeline.generate(
            sketch_image=raster_sketch,
            prompt=prompt,
            reference_images=reference_images,
            **{k: v for k, v in kwargs.items() if k not in ['stroke_width', 'fit_canvas', 'reference_images']}
        )