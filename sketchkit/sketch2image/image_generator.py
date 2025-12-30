"""
ImageGenerator for sketch-to-image conversion tasks.

This class provides functionality to convert vector sketches to realistic images using various AI methods.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Union, Tuple, Dict, Any
import torch

from sketchkit.core.sketch import Sketch
from sketchkit.renderer.cairo_renderer import CairoRenderer
from .models import ControlNetModel, T2IAdapterModel, Pix2PixModel


class ImageGenerator:
    """
    A class for sketch-to-image conversion.

    This class provides functionality to convert vector sketches to realistic images
    using various AI methods.

    Attributes:
        method (str): The conversion method that is used.
        device (str): The device that is used.
        _model: The loaded AI model instance.
    """
    
    def __init__(
        self, 
        method: str = "controlnet",
        device: str = "cuda",
        **kwargs: Any
    ):
        """
        Initializes the ImageGenerator with specified method and parameters.
        
        Args:
            method: The conversion method to use. Choose from:
                - 'controlnet': ControlNet with scribble conditioning
                - 't2i_adapter': T2I-Adapter with edge conditioning
                - 'pix2pix': InstructPix2Pix for direct translation
            device: The device to use (e.g., 'cuda' or 'cpu')
            **kwargs: Additional parameters for model initialization.
        """
        self.method = method
        self.device = device
        
        # Map method names to model classes
        if method == "controlnet":
            self._model = ControlNetModel(device=device, **kwargs)
        elif method == "t2i_adapter":
            self._model = T2IAdapterModel(device=device, **kwargs)
        elif method == "pix2pix":
            self._model = Pix2PixModel(device=device, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}. Available methods: ['controlnet', 't2i_adapter', 'pix2pix']")
    
    def run(
        self,
        sketch: Sketch,
        prompt: str = "a realistic photo",
        size: Union[int, Tuple[int, int]] = 512,
        **kwargs: Any,
    ) -> Image.Image:
        """
        Convert the input sketch to realistic image.

        Args:
            sketch: Input vector sketch.
            prompt: Text description guiding the image generation.
            size: Output image size (int for square, tuple for rectangle).
            
        Keyword Args for Generation Control:
            stroke_width (int): Stroke width for rasterization. Defaults to 3.
            fit_canvas (bool): Whether to fit sketch to canvas. Defaults to True.
            num_inference_steps (int): Number of inference steps. Defaults to 20.
            guidance_scale (float): CFG scale for text guidance. Defaults to 7.5.
            controlnet_conditioning_scale (float): ControlNet conditioning scale. Defaults to 1.0.
            adapter_conditioning_scale (float): T2I-Adapter conditioning scale. Defaults to 1.0.
            image_guidance_scale (float): Pix2Pix image guidance scale. Defaults to 1.0.
            seed (int): Random seed for reproducibility. Optional.

        Returns:
            Generated image as PIL Image.
        """
        if self.method == "controlnet":
            return self._run_controlnet(sketch, prompt, size, **kwargs)
        elif self.method == "t2i_adapter":
            return self._run_t2i_adapter(sketch, prompt, size, **kwargs)
        elif self.method == "pix2pix":
            return self._run_pix2pix(sketch, prompt, size, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def _run_controlnet(
        self,
        sketch: Sketch,
        prompt: str,
        size: Union[int, Tuple[int, int]],
        **kwargs: Any,
    ) -> Image.Image:
        """Run ControlNet-based sketch-to-image conversion."""
        # Convert sketch to raster
        raster_sketch = self._sketch_to_raster(
            sketch,
            size=size,
            stroke_width=kwargs.get('stroke_width', 3),
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
            'negative_prompt': kwargs.get('negative_prompt', ""),
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
        
        # Generate image
        return self._model.generate(**gen_params)
    
    def _run_t2i_adapter(
        self,
        sketch: Sketch,
        prompt: str,
        size: Union[int, Tuple[int, int]],
        **kwargs: Any,
    ) -> Image.Image:
        """Run T2I-Adapter-based sketch-to-image conversion."""
        # Convert sketch to raster
        raster_sketch = self._sketch_to_raster(
            sketch,
            size=size,
            stroke_width=kwargs.get('stroke_width', 3),
            fit_canvas=kwargs.get('fit_canvas', True),
        )
        
        # Prepare edge image for T2I-Adapter
        control_image = self._prepare_t2i_adapter_image(raster_sketch)
        
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
            'negative_prompt': kwargs.get('negative_prompt', ""),
            'num_inference_steps': kwargs.get('num_inference_steps', 50),  # T2I-Adapter default is 50
            'guidance_scale': kwargs.get('guidance_scale', 7.5),
            'adapter_conditioning_scale': kwargs.get('adapter_conditioning_scale', 1.0),
            'seed': generator,
        }
        
        # Remove already used parameters
        used_keys = ['stroke_width', 'fit_canvas', 'num_inference_steps', 'guidance_scale',
                    'adapter_conditioning_scale', 'negative_prompt', 'seed']
        for key in used_keys:
            kwargs.pop(key, None)
        
        # Add remaining parameters
        gen_params.update(kwargs)
        
        # Generate image
        return self._model.generate(**gen_params)
    
    def _run_pix2pix(
        self,
        sketch: Sketch,
        prompt: str,
        size: Union[int, Tuple[int, int]],
        **kwargs: Any,
    ) -> Image.Image:
        """Run Pix2Pix-based sketch-to-image conversion."""
        # Convert sketch to raster
        raster_sketch = self._sketch_to_raster(
            sketch,
            size=size,
            stroke_width=kwargs.get('stroke_width', 3),
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
            'negative_prompt': kwargs.get('negative_prompt', ""),
            'num_inference_steps': kwargs.get('num_inference_steps', 10),  # Pix2Pix default is 10
            'guidance_scale': kwargs.get('guidance_scale', 7.5),
            'image_guidance_scale': kwargs.get('image_guidance_scale', 1.0),
            'seed': generator,
        }
        
        # Remove already used parameters
        used_keys = ['stroke_width', 'fit_canvas', 'num_inference_steps', 'guidance_scale',
                    'image_guidance_scale', 'negative_prompt', 'seed']
        for key in used_keys:
            kwargs.pop(key, None)
        
        # Add remaining parameters
        gen_params.update(kwargs)
        
        # Generate image
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
    
    def _prepare_t2i_adapter_image(self, raster_sketch: np.ndarray) -> Image.Image:
        """Special preparation for T2I-Adapter edge detection."""
        pil_image = self._prepare_control_image(raster_sketch)
        
        if pil_image.mode == 'RGB':
            gray_array = np.array(pil_image.convert('L'))
        else:
            gray_array = np.array(pil_image)
        
        edges = cv2.Canny(gray_array, 50, 150)
        return Image.fromarray(edges).convert("L")