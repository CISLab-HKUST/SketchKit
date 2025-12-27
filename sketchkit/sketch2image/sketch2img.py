"""
Sketch2ImageConverter implementation using the unified architecture.
"""

import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional, Dict, Any, List
import cv2

from sketchkit.core.sketch import Sketch
from sketchkit.core.sketch_converter import BaseSketchConverter, SketchConverter
from .models import (
    ControlNetModel,
    T2IAdapterModel,
    Pix2PixModel,
)


class Sketch2ImageConverter(BaseSketchConverter, SketchConverter):
    """Convert sketches to realistic images using various AI methods.
    
    This class provides multiple AI-powered methods to convert vector sketches
    into photorealistic images. Each method is initialized with specific model
    parameters and can be reused for multiple conversions.
    
    Supported methods:
        - controlnet: Uses ControlNet with scribble conditioning
        - t2i_adapter: Uses T2I-Adapter with edge conditioning  
        - pix2pix: Uses InstructPix2Pix for image-to-image translation
    
    Example:
        >>> # Initialize different converters
        >>> controlnet_converter = Sketch2ImageConverter(method='controlnet')
        >>> pix2pix_converter = Sketch2ImageConverter(method='pix2pix')
        >>>
        >>> # Convert sketches
        >>> controlnet_image = controlnet_converter.convert(sketch, "a beautiful landscape")
        >>> pix2pix_image = pix2pix_converter.convert(sketch, "a realistic photo")
    """
    
    def __init__(
        self, 
        method: str = "controlnet",
        **kwargs: Any
    ):
        """Initialize converter with specified method and parameters.
        
        Args:
            method: The conversion method to use. Choose from:
                - 'controlnet': ControlNet with scribble conditioning
                - 't2i_adapter': T2I-Adapter with edge conditioning
                - 'pix2pix': InstructPix2Pix for direct translation
        
        Keyword Args for ControlNet:
            controlnet_model (str): ControlNet model name. Defaults to 
                "lllyasviel/sd-controlnet-scribble".
            sd_model (str): Stable Diffusion base model. Defaults to 
                "runwayml/stable-diffusion-v1-5".
            torch_dtype: Data type for model weights. Defaults to torch.float16.
            
        Keyword Args for T2I-Adapter:
            adapter_model (str): T2I-Adapter model name. Defaults to 
                "TencentARC/t2iadapter_canny_sd15v2".
            sd_model (str): Stable Diffusion base model. Defaults to 
                "runwayml/stable-diffusion-v1-5".
            torch_dtype: Data type for model weights. Defaults to torch.float16.
            
        Keyword Args for Pix2Pix:
            model_name (str): InstructPix2Pix model name. Defaults to 
                "timbrooks/instruct-pix2pix".
            torch_dtype: Data type for model weights. Defaults to torch.float16.
        
        Raises:
            ValueError: If an unsupported method is specified.
        """
        super().__init__()
        self.method = method
        self.model_params = kwargs
        self._pipeline = None
        
        # Map method names to model classes
        self._model_classes = {
            "controlnet": ControlNetModel,
            "t2i_adapter": T2IAdapterModel,
            "pix2pix": Pix2PixModel,
        }
        
        if method not in self._model_classes:
            raise ValueError(f"Unsupported method: {method}")
    
    def _lazy_load_pipeline(self) -> None:
        """Lazy load the pipeline on first use."""
        if self._pipeline is None:
            model_class = self._model_classes[self.method]
            self._pipeline = model_class(**self.model_params)
    
    def convert(
        self,
        sketch: Union[Sketch, List[Sketch]],
        prompt: str = "a realistic photo",
        *,
        size: Union[int, Tuple[int, int]] = 512,
        **kwargs: Any,
    ) -> Union[Image.Image, List[Image.Image]]:
        """Convert sketch(es) to realistic image(s) using the initialized method.
        
        This is the main conversion method that takes a vector sketch and
        transforms it into a photorealistic image using the selected AI method.
        
        Args:
            sketch: The Sketch object(s) to convert.
            prompt: Text description guiding the image generation. This should
                describe the desired output. Defaults to "a realistic photo".
            size: Output image size. If integer, creates square image. If tuple,
                creates rectangular image (width, height). Defaults to 512.
        
        Keyword Args for Generation Control:
            stroke_width (int): Stroke width for rasterization. Defaults to 3.
            fit_canvas (bool): Whether to fit sketch to canvas. Defaults to True.
            num_steps (int): Number of inference steps. Varies by method:
                - ControlNet: Defaults to 20
                - T2I-Adapter: Defaults to 50  
                - Pix2Pix: Defaults to 10
            guidance_scale (float): CFG scale for text guidance. Defaults to 7.5.
            controlnet_scale (float): ControlNet conditioning scale. Defaults to 1.0.
            adapter_scale (float): T2I-Adapter conditioning scale. Defaults to 1.0.
            image_guidance_scale (float): Pix2Pix image guidance scale. Defaults to 1.0.
        
        Returns:
            Generated realistic image(s) as PIL Image(s).
        
        Raises:
            RuntimeError: If the pipeline failed to initialize.
            ValueError: If an unsupported method is used.
        """
        self._lazy_load_pipeline()
        
        # Handle single sketch or batch
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
        """Convert a single sketch to image."""
        # Convert sketch to raster
        raster_sketch = self.sketch_to_raster(
            sketch,
            size=size,
            stroke_width=kwargs.get('stroke_width', 3),
            fit_canvas=kwargs.get('fit_canvas', True),
        )
        
        # Prepare control image based on method
        if self.method == 't2i_adapter':
            control_image = self._prepare_t2i_adapter_image(raster_sketch)
        else:
            control_image = self._prepare_control_image(raster_sketch, self.method)
        
        # Generate image using the appropriate pipeline
        return self._pipeline.generate(
            control_image=control_image,
            prompt=prompt,
            size=size,
            **{k: v for k, v in kwargs.items() if k not in ['stroke_width', 'fit_canvas']}
        )
    
    def _prepare_t2i_adapter_image(self, raster_sketch: np.ndarray) -> Image.Image:
        """Special preparation for T2I-Adapter edge detection."""
        pil_image = self._prepare_control_image(raster_sketch, "default")
        
        if pil_image.mode == 'RGB':
            gray_array = np.array(pil_image.convert('L'))
        else:
            gray_array = np.array(pil_image)
        
        edges = cv2.Canny(gray_array, 50, 150)
        return Image.fromarray(edges).convert("L")