"""
Colorizer for sketch colorization tasks.

This class provides functionality to colorize vector sketches using various AI methods.
"""

import numpy as np
from PIL import Image
from typing import Union, Tuple, Dict, Any
import torch

from sketchkit.core.sketch import Sketch
from sketchkit.renderer.cairo_renderer import CairoRenderer
from .models import ControlNetLineartModel, ControlNetScribbleModel


class Colorizer:
    """
    A class for sketch colorization.

    This class provides functionality to colorize vector sketches using various AI methods.

    Attributes:
        method (str): The colorization method that is used.
        device (str): The device that is used.
        _model: The loaded AI model instance.
    """
    
    def __init__(
        self, 
        method: str = "controlnet_lineart", 
        device: str = "cuda",
        **kwargs: Any
    ):
        """
        Initializes the Colorizer with specified method and parameters.
        
        Args:
            method: The colorization method to use. Choose from:
                - 'controlnet_lineart': ControlNet with lineart conditioning
                - 'controlnet_scribble': ControlNet with scribble conditioning
            device: The device to use (e.g., 'cuda' or 'cpu')
            **kwargs: Additional parameters for model initialization.
        """
        self.method = method
        self.device = device
        
        # Map method names to model classes
        if method == "controlnet_lineart":
            self._model = ControlNetLineartModel(device=device, **kwargs)
        elif method == "controlnet_scribble":
            self._model = ControlNetScribbleModel(device=device, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}. Available methods: ['controlnet_lineart', 'controlnet_scribble']")
    
    def run(
        self,
        sketch: Sketch,
        prompt: str = "vibrant colors, anime style",
        size: Union[int, Tuple[int, int]] = 512,
        **kwargs: Any,
    ) -> Image.Image:
        """
        Colorize the input sketch.

        Args:
            sketch: Input vector sketch.
            prompt: Text description guiding the colorization.
            size: Output image size (int for square, tuple for rectangle).
            
        Keyword Args for Generation Control:
            stroke_width (int): Stroke width for rasterization. Defaults to 2.
            fit_canvas (bool): Whether to fit sketch to canvas. Defaults to True.
            num_inference_steps (int): Number of inference steps. Defaults to 20.
            guidance_scale (float): CFG scale for text guidance. Defaults to 7.5.
            controlnet_conditioning_scale (float): ControlNet conditioning scale. Defaults to 1.0.
            negative_prompt (str): Negative prompt to avoid unwanted features.
                Defaults to "blurry, low quality, distorted".
            seed (int): Random seed for reproducibility. Optional.

        Returns:
            Colorized image as PIL Image.
        """
        if self.method == "controlnet_lineart":
            return self._run_controlnet(sketch, prompt, size, **kwargs)
        elif self.method == "controlnet_scribble":
            return self._run_controlnet(sketch, prompt, size, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def _run_controlnet(
        self,
        sketch: Sketch,
        prompt: str,
        size: Union[int, Tuple[int, int]],
        **kwargs: Any,
    ) -> Image.Image:
        """Run ControlNet-based colorization."""
        # Convert sketch to raster
        raster_sketch = self._sketch_to_raster(
            sketch,
            size=size,
            stroke_width=kwargs.get('stroke_width', 2),
            fit_canvas=kwargs.get('fit_canvas', True),
        )
        
        # Prepare control image
        control_image = self._prepare_control_image(raster_sketch)
        
        # Setup generator for reproducibility
        generator = None
        if 'seed' in kwargs:
            device = self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(kwargs['seed'])
        
        # Prepare generation parameters
        gen_params = {
            'control_image': control_image,
            'prompt': prompt,
            'size': size,
            'negative_prompt': kwargs.get('negative_prompt', "blurry, low quality, distorted"),
            'num_inference_steps': kwargs.get('num_inference_steps', 20),
            'guidance_scale': kwargs.get('guidance_scale', 7.5),
            'controlnet_conditioning_scale': kwargs.get('controlnet_conditioning_scale', 1.0),
            'seed': generator,
        }
        
        # Remove already used parameters
        used_keys = ['stroke_width', 'fit_canvas', 'num_inference_steps', 'guidance_scale',
                    'controlnet_conditioning_scale', 'negative_prompt', 'seed']
        for key in used_keys:
            kwargs.pop(key, None)
        
        # Add remaining parameters
        gen_params.update(kwargs)
        
        # Generate colorized image
        return self._model.generate(**gen_params)
    
    def _sketch_to_raster(
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
    
    def _prepare_control_image(self, raster_sketch: np.ndarray) -> Image.Image:
        """Prepare control image from rasterized sketch."""
        if len(raster_sketch.shape) == 3:
            if raster_sketch.shape[2] == 3:
                pil_image = Image.fromarray(raster_sketch).convert("RGB")
            else:
                pil_image = Image.fromarray(raster_sketch[:, :, 0]).convert("L")
        else:
            pil_image = Image.fromarray(raster_sketch).convert("L")
        
        return pil_image