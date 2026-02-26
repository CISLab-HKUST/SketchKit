"""
Colorizer for sketch colorization tasks.
"""

import numpy as np
from PIL import Image
from typing import Union, Tuple, Dict, Any, Optional
import torch

from sketchkit.core.sketch import Sketch
from sketchkit.renderer.cairo_renderer import CairoRenderer, CairoRenderOptions
from .models import ControlNetLineartModel, ControlNetScribbleModel


class Colorizer:
    """A class for sketch colorization using various AI methods.
    
    This class provides a unified interface to colorize vector sketches or 
    rasterized linearts using ControlNet-based models.
    
    Attributes:
        method (str): The colorization method being used (e.g., 'controlnet_lineart').
        device (str): The computation device ('cuda' or 'cpu').
        _model: The underlying AI model instance initialized based on the method.
    """
    
    def __init__(
        self, 
        method: str = "controlnet_lineart", 
        device: str = "cuda",
        **kwargs: Any
    ):
        """Initializes the Colorizer with a specified method and device.

        Args:
            method (str): The colorization method to use. Options are 
                'controlnet_lineart' or 'controlnet_scribble'. Defaults to "controlnet_lineart".
            device (str): The device to run the model on. Defaults to "cuda".
            **kwargs (Any): Additional keyword arguments passed to the model initialization 
                (e.g., sd_model, controlnet_model).

        Raises:
            ValueError: If an unsupported method is provided.
        """
        self.method = method
        self.device = device
        
        if method == "controlnet_lineart":
            self._model = ControlNetLineartModel(device=device, **kwargs)
        elif method == "controlnet_scribble":
            self._model = ControlNetScribbleModel(device=device, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}. Available methods: ['controlnet_lineart', 'controlnet_scribble']")
    
    def run(
        self,
        input_data: Union[Sketch, Image.Image, np.ndarray],
        prompt: str = "masterpiece, best quality, vibrant colors, flat color, highly detailed, anime illustration style", 
        size: Optional[Union[int, Tuple[int, int]]] = None,
        input_size: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Executes the colorization process on the input sketch or image.

        Args:
            input_data (Union[Sketch, Image.Image, np.ndarray]): The input lineart. 
                Can be a vector Sketch, a PIL Image, or a numpy array.
            prompt (str): Text description to guide the colorization. 
                Defaults to a high-quality anime illustration prompt.
            size (Optional[Union[int, Tuple[int, int]]]): The desired output size. 
                If None, it infers the size from the input.
            input_size (Optional[Tuple[int, int]]): Optional resolution to resize 
                the input to before processing.
            **kwargs (Any): Additional generation parameters (e.g., num_inference_steps, 
                guidance_scale, seed, negative_prompt).

        Returns:
            Image.Image: The resulting colorized image.

        Raises:
            TypeError: If the input_data type is not supported.
        """
        if isinstance(input_data, Sketch):
            return self._run_with_sketch(input_data, prompt, size, input_size, **kwargs)
        elif isinstance(input_data, (Image.Image, np.ndarray)):
            return self._run_with_image(input_data, prompt, size, input_size, **kwargs)
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}. Expected Sketch, Image.Image, or np.ndarray.")
    
    def _run_with_sketch(
            self,
            sketch: Sketch,
            prompt: str,
            size: Optional[Union[int, Tuple[int, int]]],
            input_size: Optional[Tuple[int, int]],
            **kwargs: Any,
        ) -> Image.Image:
            """Processes a vector Sketch object for colorization.

            Args:
                sketch (Sketch): The input vector sketch.
                prompt (str): Textual prompt for colorization.
                size (Optional[Union[int, Tuple[int, int]]]): Output image size.
                input_size (Optional[Tuple[int, int]]): Target size for rasterization.
                **kwargs (Any): Additional generation parameters.

            Returns:
                Image.Image: Generated colorized image.
            """
            original_w, original_h = sketch.width, sketch.height

            if input_size is not None:
                raster_w, raster_h = input_size[0], input_size[1]
            elif size is not None:
                raster_w = size if isinstance(size, int) else size[0]
                raster_h = size if isinstance(size, int) else size[1]
            else:
                raster_w, raster_h = int(original_w), int(original_h)
                size = (raster_w, raster_h)

            sketch.resize(raster_w, raster_h)
            
            try:
                raster_sketch = self._sketch_to_raster(
                    sketch,
                    size=(raster_w, raster_h),
                    stroke_width=kwargs.get('stroke_width', 2), 
                    fit_canvas=kwargs.get('fit_canvas', True),
                )
                
                control_image = self._prepare_control_image(raster_sketch)
                return self._generate_with_control_image(control_image, prompt, size, **kwargs)
                
            finally:
                sketch.resize(original_w, original_h)
    
    def _run_with_image(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: str,
        size: Optional[Union[int, Tuple[int, int]]],
        input_size: Optional[Tuple[int, int]],
        **kwargs: Any,
    ) -> Image.Image:
        """Processes a raster image (PIL or numpy array) for colorization.

        Args:
            image (Union[Image.Image, np.ndarray]): The input raster image.
            prompt (str): Textual prompt for colorization.
            size (Optional[Union[int, Tuple[int, int]]]): Output image size.
            input_size (Optional[Tuple[int, int]]): Target size to resize the input image.
            **kwargs (Any): Additional generation parameters.

        Returns:
            Image.Image: Generated colorized image.
            
        Raises:
            ValueError: If the numpy array shape is invalid.
        """
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
            elif image.ndim == 2:
                pil_image = Image.fromarray(image.astype(np.uint8)).convert("L")
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        else:
            pil_image = image
        
        if input_size is not None:
            pil_image = pil_image.resize((input_size[1], input_size[0]), Image.Resampling.LANCZOS)
        
        control_image = self._prepare_control_image(pil_image)
        return self._generate_with_control_image(control_image, prompt, size, **kwargs)
    
    def _generate_with_control_image(
        self,
        control_image: Image.Image,
        prompt: str,
        size: Union[int, Tuple[int, int], None] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Generates the final image using the underlying ControlNet model.

        Args:
            control_image (Image.Image): The processed conditioning image.
            prompt (str): Textual prompt.
            size (Optional[Union[int, Tuple[int, int]]]): Final output size.
            **kwargs (Any): Additional control parameters (e.g., negative_prompt, steps).

        Returns:
            Image.Image: The generated image.
        """
        generator = None
        if 'seed' in kwargs:
            device = self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(kwargs['seed'])

        default_negative = "monochrome, greyscale, lineart, sketch, blurry, lowres, bad anatomy, bad hands, cropped, worst quality"
        
        gen_params = {
            'control_image': control_image,
            'prompt': prompt,
            'size': size,
            'negative_prompt': kwargs.get('negative_prompt', default_negative),
            'num_inference_steps': kwargs.get('num_inference_steps', 30), 
            'guidance_scale': kwargs.get('guidance_scale', 7.5),
            'controlnet_conditioning_scale': kwargs.get('controlnet_conditioning_scale', 1.0),
            'seed': generator,
        }
        
        used_keys = ['stroke_width', 'fit_canvas', 'num_inference_steps', 'guidance_scale',
                    'controlnet_conditioning_scale', 'negative_prompt', 'seed']
        for key in used_keys:
            kwargs.pop(key, None)
        
        gen_params.update(kwargs)
        return self._model.generate(**gen_params)
    
    def _sketch_to_raster(
        self,
        sketch: Sketch,
        size: Union[int, Tuple[int, int]] = 1024, 
        background_color: Tuple[float, float, float] = (1, 1, 1),
        stroke_width: int = 3,
        fit_canvas: bool = True,
    ) -> np.ndarray:
        """Converts a vector Sketch to a rasterized numpy array.

        Args:
            sketch (Sketch): The vector sketch to render.
            size (Union[int, Tuple[int, int]]): Canvas dimensions. Defaults to 1024.
            background_color (Tuple[float, float, float]): RGB background color. Defaults to white.
            stroke_width (int): Thickness of the rendered strokes. Defaults to 3.
            fit_canvas (bool): Whether to scale paths to fit the canvas. Defaults to True.

        Returns:
            np.ndarray: The rasterized image as a numpy array.
        """
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
        return np.array(pil_image)
    
    def _prepare_control_image(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """Standardizes the control image format to a PIL Image.

        Args:
            image (Union[np.ndarray, Image.Image]): The raw input image array or PIL object.

        Returns:
            Image.Image: The standardized PIL Image in the correct color mode.
            
        Raises:
            TypeError: If the input type is unsupported.
        """
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