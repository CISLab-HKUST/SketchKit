"""
ControlNet Lineart Model for Sketch Colorization (SDXL Version)
"""

import torch
from diffusers import (
    ControlNetModel as DiffusersControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler
)
from PIL import Image
import numpy as np
from typing import Tuple, Union, Dict, Any, Optional
import warnings


class ControlNetLineartModel:
    """ControlNet model with lineart conditioning for sketch colorization using SDXL.
    
    Attributes:
        pipeline (StableDiffusionXLControlNetPipeline): The loaded diffusers pipeline.
        device (str): Computation device.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        controlnet_model: str = "ShermanG/ControlNet-Standard-Lineart-for-SDXL", 
        sd_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        vae_model: str = "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype: torch.dtype = torch.float16,
        safety_checker: bool = False,
        **kwargs: Any,
    ):
        """Initializes the SDXL ControlNet Lineart pipeline.

        Args:
            device (str): Hardware device to run the model on. Defaults to "cuda".
            controlnet_model (str): HuggingFace hub path or local path for the ControlNet.
            sd_model (str): HuggingFace hub path or local path for the base SDXL model.
            vae_model (str): HuggingFace hub path or local path for the VAE model.
            torch_dtype (torch.dtype): PyTorch data type for model weights.
            safety_checker (bool): Whether to enable the NSFW safety checker.
            **kwargs (Any): Additional kwargs passed to diffusers loading methods.

        Raises:
            RuntimeError: If model initialization fails.
        """
        print(f"Loading SDXL Lineart ControlNet model: {controlnet_model}")
        
        try:
            scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                sd_model, subfolder="scheduler", **kwargs
            )

            controlnet = DiffusersControlNetModel.from_pretrained(
                controlnet_model, torch_dtype=torch_dtype, **kwargs
            )
            
            vae = AutoencoderKL.from_pretrained(
                vae_model, torch_dtype=torch_dtype, **kwargs
            )

            if safety_checker:
                self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                    sd_model, controlnet=controlnet, vae=vae, scheduler=scheduler,
                    torch_dtype=torch_dtype, **kwargs
                )
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                        sd_model, controlnet=controlnet, vae=vae, scheduler=scheduler,
                        torch_dtype=torch_dtype, safety_checker=None, requires_safety_checker=False, **kwargs
                    )
            
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()
            
            if device == "cuda" and torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
            elif device == "cpu":
                self.pipeline = self.pipeline.to("cpu")
            
            self.device = device
            print(f"✅ SDXL Lineart model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load SDXL Lineart model: {e}") from e
    
    def generate(
        self,
        control_image: Image.Image,
        prompt: str,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Generates an image from a lineart sketch using SDXL ControlNet.

        Args:
            control_image (Image.Image): The input conditioning lineart image.
            prompt (str): Text prompt describing the desired output.
            size (Optional[Union[int, Tuple[int, int]]]): Target resolution. If None, 
                it auto-scales proportionally based on a 1024x1024 baseline.
            negative_prompt (str): Text prompt detailing what to avoid.
            num_inference_steps (int): Number of denoising steps. Defaults to 30.
            guidance_scale (float): CFG scale for text guidance. Defaults to 7.5.
            controlnet_conditioning_scale (float): Strength of the ControlNet. Defaults to 1.0.
            seed (Optional[int]): Random seed for deterministic generation.
            **kwargs (Any): Additional arguments for the pipeline.

        Returns:
            Image.Image: The generated image.

        Raises:
            RuntimeError: If image generation fails during pipeline execution.
        """
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('num_steps', None)
        kwargs_copy.pop('guidance_scale', None)
        kwargs_copy.pop('controlnet_scale', None)
        
        generator = None
        if seed is not None:
            device = self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(seed)
        
        width, height = control_image.size
        if size is None:
            ratio = np.sqrt(1024.0 * 1024.0 / (width * height))
            new_width, new_height = int(width * ratio), int(height * ratio)
        else:
            new_width = size if isinstance(size, int) else size[0]
            new_height = size if isinstance(size, int) else size[1]

        new_width, new_height = (new_width // 8) * 8, (new_height // 8) * 8
        
        if (width, height) != (new_width, new_height):
            control_image = control_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        try:
            result = self.pipeline(
                prompt=prompt,
                image=control_image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
                width=new_width,
                height=new_height,
                **kwargs_copy,
            )
            return result.images[0]
        except Exception as e:
            raise RuntimeError(f"SDXL Lineart generation failed: {e}") from e