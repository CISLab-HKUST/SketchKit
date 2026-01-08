"""
Colorizer for sketch colorization tasks.

This class provides functionality to colorize vector sketches using various AI methods.
"""

import numpy as np
from PIL import Image
from typing import Union, Tuple, Dict, Any, Optional
import torch

from sketchkit.core.sketch import Sketch
from sketchkit.renderer.cairo_renderer import CairoRenderer, CairoRenderOptions
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
        input_data: Union[Sketch, Image.Image, np.ndarray],
        prompt: str = "vibrant colors, anime style",
        size: Union[int, Tuple[int, int]] = 512,
        input_size: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """
        Colorize the input sketch or image.

        Args:
            input_data: Input can be:
                - Sketch: Vector sketch object
                - Image.Image: PIL Image object
                - np.ndarray: Numpy array image
            prompt: Text description guiding the colorization.
            size: Output image size (int for square, tuple for rectangle).
            input_size: Optional input resize dimensions (height, width). If provided,
                the input will be resized before processing.
            
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
        # Determine input type and process accordingly
        if isinstance(input_data, Sketch):
            return self._run_with_sketch(input_data, prompt, size, input_size, **kwargs)
        elif isinstance(input_data, (Image.Image, np.ndarray)):
            return self._run_with_image(input_data, prompt, size, input_size, **kwargs)
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}. "
                          f"Expected Sketch, Image.Image, or np.ndarray.")
    
    def _run_with_sketch(
        self,
        sketch: Sketch,
        prompt: str,
        size: Union[int, Tuple[int, int]],
        input_size: Optional[Tuple[int, int]],
        **kwargs: Any,
    ) -> Image.Image:
        """Run colorization with Sketch input."""
        # Resize sketch if requested 
        original_size = None
        if input_size is not None:
            original_size = (sketch.height, sketch.width)
            sketch.resize(input_size[0], input_size[1])
        
        try:
            # Convert sketch to raster
            raster_sketch = self._sketch_to_raster(
                sketch,
                size=size,
                stroke_width=kwargs.get('stroke_width', 2),
                fit_canvas=kwargs.get('fit_canvas', True),
            )
            
            # Prepare control image
            control_image = self._prepare_control_image(raster_sketch)
            
            return self._generate_with_control_image(control_image, prompt, size, **kwargs)
        
        finally:
            if original_size is not None:
                sketch.resize(original_size[0], original_size[1])
    
    def _run_with_image(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: str,
        size: Union[int, Tuple[int, int]],
        input_size: Optional[Tuple[int, int]],
        **kwargs: Any,
    ) -> Image.Image:
        """Run colorization with Image input."""
        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
            elif image.ndim == 2:
                pil_image = Image.fromarray(image.astype(np.uint8)).convert("L")
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        else:
            pil_image = image
        
        # Resize image if requested
        if input_size is not None:
            pil_image = pil_image.resize((input_size[1], input_size[0]), Image.Resampling.LANCZOS)
        
        # Prepare control image
        control_image = self._prepare_control_image(pil_image)
        
        return self._generate_with_control_image(control_image, prompt, size, **kwargs)
    
    def _generate_with_control_image(
        self,
        control_image: Image.Image,
        prompt: str,
        size: Union[int, Tuple[int, int]],
        **kwargs: Any,
    ) -> Image.Image:
        """Generate colorized image using control image."""
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
            canvas_size = (size, size)
        else:
            canvas_size = size
                
        render_options = CairoRenderOptions(
            canvas_size=canvas_size,
            background_color=background_color,
            stroke_width=stroke_width
        )
        
        renderer = CairoRenderer(render_options=render_options)
        
        pil_image = renderer.render(sketch, render_options=render_options)
        raster_image = np.array(pil_image)
        
        return raster_image
    
    def _prepare_control_image(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """Prepare control image from rasterized sketch or PIL image."""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    pil_image = Image.fromarray(image).convert("RGB")
                else:
                    pil_image = Image.fromarray(image[:, :, 0]).convert("L")
            else:
                pil_image = Image.fromarray(image).convert("L")
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        return pil_image