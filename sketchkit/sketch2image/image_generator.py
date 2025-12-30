"""
ImageGenerator for sketch-to-image conversion tasks.
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
        model_params (dict): Parameters for model initialization.
        _pipeline: The loaded AI model pipeline.
    """
    
    def __init__(
        self, 
        method: str = "controlnet",
        **kwargs: Any
    ):
        """
        Initializes the ImageGenerator with specified method and parameters.
        
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
        """
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
            raise ValueError(f"Unsupported method: {method}. Available methods: {list(self._model_classes.keys())}")
    
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
            sketch (Sketch): Input vector sketch.
            prompt (str): Text description guiding the image generation.
            size (Union[int, Tuple[int, int]]): Output image size.
            
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
            Image.Image: Generated image as PIL Image.
        """
        # Lazy load pipeline
        if self._pipeline is None:
            model_class = self._model_classes[self.method]
            self._pipeline = model_class(**self.model_params)
        
        # Convert sketch to raster
        raster_sketch = self._sketch_to_raster(
            sketch,
            size=size,
            stroke_width=kwargs.get('stroke_width', 3),
            fit_canvas=kwargs.get('fit_canvas', True),
        )
        
        # Prepare control image based on method
        if self.method == 't2i_adapter':
            control_image = self._prepare_t2i_adapter_image(raster_sketch)
        else:
            control_image = self._prepare_control_image(raster_sketch)
        
        # 根据方法设置默认参数
        if self.method == 'controlnet':
            default_steps = 20
            scale_param = 'controlnet_conditioning_scale'
            scale_value = kwargs.get('controlnet_conditioning_scale', 1.0)
        elif self.method == 't2i_adapter':
            default_steps = 50
            scale_param = 'adapter_conditioning_scale'
            scale_value = kwargs.get('adapter_conditioning_scale', 1.0)
        elif self.method == 'pix2pix':
            default_steps = 10
            scale_param = 'image_guidance_scale'
            scale_value = kwargs.get('image_guidance_scale', 1.0)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        # Setup generator for reproducibility
        generator = None
        if 'seed' in kwargs:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(kwargs['seed'])
        
        # 准备生成参数
        gen_params = {
            'control_image': control_image,
            'prompt': prompt,
            'size': size,
            'negative_prompt': kwargs.get('negative_prompt', ""),
            'num_inference_steps': kwargs.get('num_inference_steps', default_steps),
            'guidance_scale': kwargs.get('guidance_scale', 7.5),
            'seed': generator,
            scale_param: scale_value,
        }
        
        # 移除已经使用的参数
        used_keys = ['stroke_width', 'fit_canvas', 'num_inference_steps', 'guidance_scale',
                    'controlnet_conditioning_scale', 'adapter_conditioning_scale',
                    'image_guidance_scale', 'negative_prompt', 'seed']
        for key in used_keys:
            kwargs.pop(key, None)
        
        # 添加剩余的参数
        gen_params.update(kwargs)
        
        # Generate image using the appropriate pipeline
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
    
    def _prepare_t2i_adapter_image(self, raster_sketch: np.ndarray) -> Image.Image:
        """Special preparation for T2I-Adapter edge detection."""
        pil_image = self._prepare_control_image(raster_sketch)
        
        if pil_image.mode == 'RGB':
            gray_array = np.array(pil_image.convert('L'))
        else:
            gray_array = np.array(pil_image)
        
        edges = cv2.Canny(gray_array, 50, 150)
        return Image.fromarray(edges).convert("L")