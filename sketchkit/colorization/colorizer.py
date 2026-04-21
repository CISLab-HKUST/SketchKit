"""
Colorizer for sketch colorization tasks.
"""

import numpy as np
from PIL import Image
from typing import Union, Tuple, Dict, Any, Optional
import torch

from sketchkit.core.sketch import Sketch
from sketchkit.renderer.cairo_renderer import CairoRenderer, CairoRenderOptions
from .controlnet_lineart import ControlNetLineartModel
from .controlnet_scribble import ControlNetScribbleModel
from .manga_ninja import MangaNinjaModel


class Colorizer:
    """A class for sketch colorization using various AI methods.
    
    This class provides a unified interface to colorize vector sketches or 
    rasterized linearts using ControlNet-based models or reference-based
    methods like MangaNinja.
    
    Attributes:
        method (str): The colorization method being used (e.g., 'controlnet_lineart').
        device (str): The computation device ('cuda' or 'cpu').
        _model: The underlying AI model instance initialized based on the method.
    """
    
    AVAILABLE_METHODS = ["controlnet_lineart", "controlnet_scribble", "manga_ninja"]

    def __init__(
        self, 
        method: str = "controlnet_lineart", 
        device: str = "cuda",
        **kwargs: Any
    ):
        """Initializes the Colorizer with a specified method and device.

        Args:
            method (str): The colorization method to use. Options are 
                'controlnet_lineart', 'controlnet_scribble', or 'manga_ninja'. 
                Defaults to "controlnet_lineart".
            device (str): The device to run the model on. Defaults to "cuda".
            **kwargs (Any): Additional keyword arguments passed to the model initialization.

        Raises:
            ValueError: If an unsupported method is provided.
        """
        self.method = method
        self.device = device
        
        if method == "controlnet_lineart":
            self._model = ControlNetLineartModel(device=device, **kwargs)
        elif method == "controlnet_scribble":
            self._model = ControlNetScribbleModel(device=device, **kwargs)
        elif method == "manga_ninja":
            self._model = MangaNinjaModel(device=device, **kwargs)
        else:
            raise ValueError(
                f"Unsupported method: {method}. Available methods: {self.AVAILABLE_METHODS}"
            )
    
    def run(
        self,
        input_data: Union[Sketch, Image.Image, np.ndarray],
        prompt: str = "masterpiece, best quality, vibrant colors, flat color, highly detailed, anime illustration style", 
        size: Optional[Union[int, Tuple[int, int]]] = None,
        input_size: Optional[Tuple[int, int]] = None,
        reference_image: Optional[Union[Image.Image, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Executes the colorization process on the input sketch or image.

        Args:
            input_data (Union[Sketch, Image.Image, np.ndarray]): The input lineart. 
                Can be a vector Sketch, a PIL Image, or a numpy array.
            prompt (str): Text description to guide the colorization. 
                Defaults to a high-quality anime illustration prompt.
                Note: Not used by 'manga_ninja' method (reference image is used instead).
            size (Optional[Union[int, Tuple[int, int]]]): The desired output size. 
                If None, it infers the size from the input.
                For 'manga_ninja', the internal processing resolution is 512x512.
            input_size (Optional[Tuple[int, int]]): Optional resolution to resize 
                the input to before processing.
            reference_image (Optional[Union[Image.Image, np.ndarray]]): Reference image 
                for colorization guidance. Required for 'manga_ninja' method.
                Ignored for ControlNet-based methods.
            **kwargs (Any): Additional generation parameters. Common parameters include:
                - num_inference_steps (int): Number of denoising steps.
                - guidance_scale (float): CFG scale for text guidance.
                - seed (int): Random seed for deterministic generation.
                - negative_prompt (str): Negative text prompt.
                For 'manga_ninja' method, additional parameters include:
                - is_lineart (bool): If True, input is already line art.
                - guidance_scale_ref (float): Guidance scale for reference image influence.
                - guidance_scale_point (float): Guidance scale for point control influence.
                - point_ref (torch.Tensor): Point map on reference image (1,1,H,W).
                - point_main (torch.Tensor): Point map on line art image (1,1,H,W).

        Returns:
            Image.Image: The resulting colorized image.

        Raises:
            TypeError: If the input_data type is not supported.
            ValueError: If 'manga_ninja' method is used without a reference_image.
        """
        if self.method == "manga_ninja" and reference_image is None:
            raise ValueError(
                "reference_image is required for the 'manga_ninja' method. "
                "Please provide a reference image to guide the colorization."
            )

        if isinstance(input_data, Sketch):
            return self._run_with_sketch(input_data, prompt, size, input_size, reference_image, **kwargs)
        elif isinstance(input_data, (Image.Image, np.ndarray)):
            return self._run_with_image(input_data, prompt, size, input_size, reference_image, **kwargs)
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}. Expected Sketch, Image.Image, or np.ndarray.")
    
    def _run_with_sketch(
            self,
            sketch: Sketch,
            prompt: str,
            size: Optional[Union[int, Tuple[int, int]]],
            input_size: Optional[Tuple[int, int]],
            reference_image: Optional[Union[Image.Image, np.ndarray]] = None,
            **kwargs: Any,
        ) -> Image.Image:
            """Processes a vector Sketch object for colorization.

            Args:
                sketch (Sketch): The input vector sketch.
                prompt (str): Textual prompt for colorization.
                size (Optional[Union[int, Tuple[int, int]]]): Output image size.
                input_size (Optional[Tuple[int, int]]): Target size for rasterization.
                reference_image (Optional[Union[Image.Image, np.ndarray]]): Reference image 
                    for MangaNinja method.
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
                ref_image = self._prepare_reference_image(reference_image) if reference_image is not None else None
                return self._generate_with_control_image(control_image, prompt, size, reference_image=ref_image, **kwargs)
                
            finally:
                sketch.resize(original_w, original_h)
    
    def _run_with_image(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: str,
        size: Optional[Union[int, Tuple[int, int]]],
        input_size: Optional[Tuple[int, int]],
        reference_image: Optional[Union[Image.Image, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Processes a raster image (PIL or numpy array) for colorization.

        Args:
            image (Union[Image.Image, np.ndarray]): The input raster image.
            prompt (str): Textual prompt for colorization.
            size (Optional[Union[int, Tuple[int, int]]]): Output image size.
            input_size (Optional[Tuple[int, int]]): Target size to resize the input image.
            reference_image (Optional[Union[Image.Image, np.ndarray]]): Reference image 
                for MangaNinja method.
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
        ref_image = self._prepare_reference_image(reference_image) if reference_image is not None else None
        return self._generate_with_control_image(control_image, prompt, size, reference_image=ref_image, **kwargs)
    
    def _generate_with_control_image(
        self,
        control_image: Image.Image,
        prompt: str,
        size: Union[int, Tuple[int, int], None] = None,
        reference_image: Optional[Image.Image] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Generates the final image using the underlying model.

        Args:
            control_image (Image.Image): The processed conditioning image.
            prompt (str): Textual prompt.
            size (Optional[Union[int, Tuple[int, int]]]): Final output size.
            reference_image (Optional[Image.Image]): Reference image for MangaNinja.
            **kwargs (Any): Additional control parameters.

        Returns:
            Image.Image: The generated image.
        """
        if self.method == "manga_ninja":
            return self._generate_manga_ninja(control_image, reference_image, size, **kwargs)
        return self._generate_controlnet(control_image, prompt, size, **kwargs)

    def _generate_controlnet(
        self,
        control_image: Image.Image,
        prompt: str,
        size: Union[int, Tuple[int, int], None] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Generates an image using ControlNet-based models.

        Args:
            control_image (Image.Image): The processed conditioning image.
            prompt (str): Textual prompt.
            size (Optional[Union[int, Tuple[int, int]]]): Final output size.
            **kwargs (Any): Additional control parameters.

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

    def _generate_manga_ninja(
        self,
        control_image: Image.Image,
        reference_image: Image.Image,
        size: Union[int, Tuple[int, int], None] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Generates a colorized image using MangaNinja.

        Args:
            control_image (Image.Image): The input line art image.
            reference_image (Image.Image): The reference image for color guidance.
            size (Optional[Union[int, Tuple[int, int]]]): Processing resolution.
            **kwargs (Any): MangaNinja-specific parameters including:
                - is_lineart (bool): Whether input is already line art.
                - guidance_scale_ref (float): Reference guidance scale.
                - guidance_scale_point (float): Point guidance scale.
                - num_inference_steps (int): Denoising steps.
                - seed (int): Random seed.
                - point_ref (torch.Tensor): Point map on reference image.
                - point_main (torch.Tensor): Point map on line art image.

        Returns:
            Image.Image: The colorized image.
        """
        gen_params = {
            'control_image': control_image,
            'reference_image': reference_image,
            'size': size,
            'is_lineart': kwargs.get('is_lineart', False),
            'num_inference_steps': kwargs.get('num_inference_steps', 50),
            'guidance_scale_ref': kwargs.get('guidance_scale_ref', 9.0),
            'guidance_scale_point': kwargs.get('guidance_scale_point', 15.0),
            'seed': kwargs.get('seed', None),
            'point_ref': kwargs.get('point_ref', None),
            'point_main': kwargs.get('point_main', None),
        }

        used_keys = ['stroke_width', 'fit_canvas', 'is_lineart', 'num_inference_steps',
                     'guidance_scale_ref', 'guidance_scale_point', 'seed',
                     'point_ref', 'point_main']
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

    def _prepare_reference_image(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Standardizes the reference image format to a PIL RGB Image.

        Args:
            image (Union[Image.Image, np.ndarray]): The reference image.

        Returns:
            Image.Image: The standardized PIL Image in RGB mode.
        """
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
            elif image.ndim == 2:
                pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
            else:
                pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        return pil_image
