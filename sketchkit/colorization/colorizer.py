"""
Colorizer for sketch colorization tasks.
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
        model_params (dict): Parameters for model initialization.
        _pipeline: The loaded AI model pipeline.
    """
    
    def __init__(
        self, 
        method: str = "controlnet_lineart", 
        **kwargs: Any
    ):
        """
        Initializes the Colorizer with specified method and parameters.
        
        Args:
            method: The colorization method to use. Choose from:
                - 'controlnet_lineart': ControlNet with lineart conditioning
                - 'controlnet_scribble': ControlNet with scribble conditioning
        
        Keyword Args for ControlNet:
            controlnet_model (str): ControlNet model name. Defaults vary by method.
            sd_model (str): Stable Diffusion base model. Defaults to 
                "runwayml/stable-diffusion-v1-5".
            torch_dtype: Data type for model weights. Defaults to torch.float16.
        """
        self.method = method
        self.model_params = kwargs
        self._pipeline = None
        
        # Map method names to model classes
        self._model_classes = {
            "controlnet_lineart": ControlNetLineartModel,
            "controlnet_scribble": ControlNetScribbleModel,
        }
        
        if method not in self._model_classes:
            raise ValueError(f"Unsupported method: {method}. Available methods: {list(self._model_classes.keys())}")
    
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
            sketch (Sketch): Input vector sketch.
            prompt (str): Text description guiding the colorization.
            size (Union[int, Tuple[int, int]]): Output image size.
            
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
            Image.Image: Colorized image as PIL Image.
        """
        # Lazy load pipeline
        if self._pipeline is None:
            model_class = self._model_classes[self.method]
            self._pipeline = model_class(**self.model_params)
        
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
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(kwargs['seed'])
        
        # 根据方法设置默认参数
        if self.method == 'controlnet_lineart' or self.method == 'controlnet_scribble':
            default_steps = 20
            scale_param = 'controlnet_conditioning_scale'
            scale_value = kwargs.get('controlnet_conditioning_scale', 1.0)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        # 准备生成参数
        gen_params = {
            'control_image': control_image,
            'prompt': prompt,
            'size': size,
            'negative_prompt': kwargs.get('negative_prompt', "blurry, low quality, distorted"),
            'num_inference_steps': kwargs.get('num_inference_steps', default_steps),
            'guidance_scale': kwargs.get('guidance_scale', 7.5),
            'seed': generator,
            scale_param: scale_value,
        }
        
        # 移除已经使用的参数
        used_keys = ['stroke_width', 'fit_canvas', 'num_inference_steps', 'guidance_scale', 
                    'controlnet_conditioning_scale', 'negative_prompt', 'seed']
        for key in used_keys:
            kwargs.pop(key, None)
        
        # 添加剩余的参数
        gen_params.update(kwargs)
        
        # Generate colorized image
        return self._pipeline.generate(**gen_params)
    
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