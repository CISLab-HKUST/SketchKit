"""
ImageGenerator for sketch-to-image conversion tasks.

This module provides the core `ImageGenerator` class, which acts as a unified 
interface for converting vector sketches or rasterized images into realistic 
images using various AI models (ControlNet, T2I-Adapter, InstructPix2Pix).
"""

import numpy as np
import cv2
from PIL import Image
from typing import Union, Tuple, Dict, Any, Optional
import torch

from sketchkit.core.sketch import Sketch
from sketchkit.renderer.cairo_renderer import CairoRenderer, CairoRenderOptions
from .models import ControlNetModel, T2IAdapterModel, Pix2PixModel


class ImageGenerator:
    """A class for sketch-to-image conversion.

    This class provides functionality to convert vector sketches to realistic images
    using various AI methods. It automatically handles the preprocessing of inputs 
    (such as rasterizing vector sketches or applying edge detection) and routes 
    the request to the appropriate generative model.

    Attributes:
        method (str): The conversion method that is used (e.g., 'controlnet').
        device (str): The computation device ('cuda' or 'cpu').
        _model: The loaded AI model instance.
    """
    
    def __init__(
        self, 
        method: str = "controlnet",
        device: str = "cuda",
        **kwargs: Any
    ):
        """Initializes the ImageGenerator with a specified method and parameters.
        
        Args:
            method (str): The conversion method to use. Options are:
                - 'controlnet': ControlNet with scribble conditioning.
                - 't2i_adapter': T2I-Adapter with sketch/edge conditioning.
                - 'pix2pix': InstructPix2Pix for direct image translation.
                Defaults to "controlnet".
            device (str): The device to run the model on. Defaults to "cuda".
            **kwargs (Any): Additional parameters for model initialization 
                (e.g., `sd_model`, `vae_model`).

        Raises:
            ValueError: If an unsupported method is provided.
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
        input_data: Union[Sketch, Image.Image, np.ndarray],
        prompt: str = "a realistic photo",
        size: Optional[Union[int, Tuple[int, int]]] = None,
        input_size: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Converts the input sketch or image to a realistic image.

        Args:
            input_data (Union[Sketch, Image.Image, np.ndarray]): The input representation. Can be:
                - `Sketch`: A vector sketch object.
                - `Image.Image`: A PIL Image object.
                - `np.ndarray`: A Numpy array representing an image.
            prompt (str): Text description guiding the image generation. Defaults to "a realistic photo".
            size (Optional[Union[int, Tuple[int, int]]]): Output image size (int for square, 
                tuple for rectangle). If None, it infers the target size from the input.
            input_size (Optional[Tuple[int, int]]): Target dimensions (width, height) to 
                resize the input to before processing.
            
            **kwargs (Any): Keyword arguments for Generation Control:
                - `stroke_width` (int): Stroke width for rasterization. Defaults to 3.
                - `fit_canvas` (bool): Whether to fit sketch to canvas. Defaults to True.
                - `num_inference_steps` (int): Number of denoising steps. Defaults to 30.
                - `guidance_scale` (float): CFG scale for text guidance. Defaults to 7.5.
                - `controlnet_conditioning_scale` (float): Strength of ControlNet. Defaults to 1.0.
                - `adapter_conditioning_scale` (float): Strength of T2I-Adapter. Defaults to 1.0.
                - `image_guidance_scale` (float): Strength of InstructPix2Pix image guidance. Defaults to 1.5.
                - `negative_prompt` (str): Prompt detailing what to avoid.
                - `seed` (int): Random seed for reproducibility.

        Returns:
            Image.Image: The generated realistic image.

        Raises:
            TypeError: If the `input_data` is not of a supported type.
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
            size: Optional[Union[int, Tuple[int, int]]],
            input_size: Optional[Tuple[int, int]],
            **kwargs: Any,
        ) -> Image.Image:
            """Processes a vector Sketch object for sketch-to-image conversion.

            Args:
                sketch (Sketch): The input vector sketch.
                prompt (str): Textual prompt guiding the generation.
                size (Optional[Union[int, Tuple[int, int]]]): Target output dimensions.
                input_size (Optional[Tuple[int, int]]): Resolution for rasterizing the sketch.
                **kwargs (Any): Additional pipeline parameters.

            Returns:
                Image.Image: The generated image.
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
                    stroke_width=kwargs.get('stroke_width', 3),
                    fit_canvas=kwargs.get('fit_canvas', True),
                )
                
                control_image = self._prepare_control_image(raster_sketch)
                
                if self.method == "controlnet":
                    return self._run_controlnet(control_image, prompt, size, **kwargs)
                elif self.method == "t2i_adapter":
                    control_image = self._prepare_t2i_adapter_image(control_image)
                    return self._run_t2i_adapter(control_image, prompt, size, **kwargs)
                elif self.method == "pix2pix":
                    return self._run_pix2pix(control_image, prompt, size, **kwargs)
                else:
                    raise ValueError(f"Unsupported method: {self.method}")
            
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
        """Processes a raster image (PIL or numpy array) for sketch-to-image conversion.

        Args:
            image (Union[Image.Image, np.ndarray]): The input raster image.
            prompt (str): Textual prompt guiding the generation.
            size (Optional[Union[int, Tuple[int, int]]]): Output image size.
            input_size (Optional[Tuple[int, int]]): Target size to resize the input image.
            **kwargs (Any): Additional generation parameters.

        Returns:
            Image.Image: The generated image.
            
        Raises:
            ValueError: If the numpy array shape is not supported.
            TypeError: If the input type is unknown.
        """
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
        
        # If size is not provided, use image's dimensions
        if size is None:
            size = (pil_image.height, pil_image.width)
        
        # Prepare control image
        control_image = self._prepare_control_image(pil_image)
        
        if self.method == "controlnet":
            return self._run_controlnet(control_image, prompt, size, **kwargs)
        elif self.method == "t2i_adapter":
            # For T2I-Adapter, we need to do edge detection on the image
            control_image = self._prepare_t2i_adapter_image(control_image)
            return self._run_t2i_adapter(control_image, prompt, size, **kwargs)
        elif self.method == "pix2pix":
            return self._run_pix2pix(control_image, prompt, size, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def _run_controlnet(
            self,
            control_image: Image.Image,
            prompt: str,
            size: Union[int, Tuple[int, int], None] = None, 
            **kwargs: Any,
        ) -> Image.Image:
            """Executes the ControlNet model for image generation.

            Args:
                control_image (Image.Image): The conditioning sketch image.
                prompt (str): Text prompt.
                size (Union[int, Tuple[int, int], None]): Output size.
                **kwargs (Any): Additional parameters.

            Returns:
                Image.Image: The generated image.
            """
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
                'negative_prompt': kwargs.get('negative_prompt', "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"),
                'num_inference_steps': kwargs.get('num_inference_steps', 30), 
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
            control_image: Image.Image,
            prompt: str,
            size: Union[int, Tuple[int, int], None] = None,
            **kwargs: Any,
        ) -> Image.Image:
            """Executes the T2I-Adapter model for image generation.

            Args:
                control_image (Image.Image): The conditioning sketch/edge image.
                prompt (str): Text prompt.
                size (Union[int, Tuple[int, int], None]): Output size.
                **kwargs (Any): Additional parameters.

            Returns:
                Image.Image: The generated image.
            """
            generator = None
            if 'seed' in kwargs:
                device = self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device=device).manual_seed(kwargs['seed'])
            
            gen_params = {
                'control_image': control_image,
                'prompt': prompt,
                'size': size,
                'negative_prompt': kwargs.get('negative_prompt', "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"),
                'num_inference_steps': kwargs.get('num_inference_steps', 30),  
                'guidance_scale': kwargs.get('guidance_scale', 7.5),
                'adapter_conditioning_scale': kwargs.get('adapter_conditioning_scale', 1.0),
                'seed': generator,
            }
            
            used_keys = ['stroke_width', 'fit_canvas', 'num_inference_steps', 'guidance_scale',
                        'adapter_conditioning_scale', 'negative_prompt', 'seed']
            for key in used_keys:
                kwargs.pop(key, None)
            
            gen_params.update(kwargs)
            return self._model.generate(**gen_params)
    
    def _run_pix2pix(
        self,
        control_image: Image.Image,
        prompt: str,
        size: Union[int, Tuple[int, int], None] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Executes the InstructPix2Pix model for direct image translation.

        Args:
            control_image (Image.Image): The input image to be translated.
            prompt (str): Instruction prompt (e.g., "make it a photo").
            size (Union[int, Tuple[int, int], None]): Output size.
            **kwargs (Any): Additional parameters.

        Returns:
            Image.Image: The generated image.
        """
        generator = None
        if 'seed' in kwargs:
            device = self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(kwargs['seed'])
        
        gen_params = {
            'control_image': control_image,
            'prompt': prompt,
            'size': size,
            'negative_prompt': kwargs.get('negative_prompt', "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"),
            'num_inference_steps': kwargs.get('num_inference_steps', 30), 
            'guidance_scale': kwargs.get('guidance_scale', 7.5),
            'image_guidance_scale': kwargs.get('image_guidance_scale', 1.5), 
            'seed': generator,
        }
        
        used_keys = ['stroke_width', 'fit_canvas', 'num_inference_steps', 'guidance_scale',
                    'image_guidance_scale', 'negative_prompt', 'seed']
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
        raster_image = np.array(pil_image)
        
        return raster_image
    
    def _prepare_control_image(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """Standardizes the control image format to a PIL Image.

        Args:
            image (Union[np.ndarray, Image.Image]): The raw input image array or PIL object.

        Returns:
            Image.Image: The standardized PIL Image in grayscale ('L') or RGB mode depending on input.
            
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
    
    def _prepare_t2i_adapter_image(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """Applies Canny edge detection specifically for the T2I-Adapter model.

        Args:
            image (Union[np.ndarray, Image.Image]): The input raster image.

        Returns:
            Image.Image: A PIL Image containing the extracted edges in grayscale ('L') mode.
        """
        pil_image = self._prepare_control_image(image)
        
        if pil_image.mode == 'RGB':
            gray_array = np.array(pil_image.convert('L'))
        else:
            gray_array = np.array(pil_image)
        
        edges = cv2.Canny(gray_array, 50, 150)
        return Image.fromarray(edges).convert("L")