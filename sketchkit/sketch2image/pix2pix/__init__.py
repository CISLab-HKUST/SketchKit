"""
Pix2Pix Model Implementation for Sketch-to-Image Conversion (SDXL Version)

This module encapsulates the diffusers `StableDiffusionXLInstructPix2PixPipeline` 
for generating images via instructional prompts.
"""

import torch
from diffusers import (
    StableDiffusionXLInstructPix2PixPipeline,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler
)
from PIL import Image
import numpy as np
from typing import Tuple, Union, Dict, Any, Optional
import warnings


class Pix2PixModel:
    """InstructPix2Pix model for direct sketch-to-image translation using SDXL.
    
    Attributes:
        pipeline (StableDiffusionXLInstructPix2PixPipeline): The loaded diffusers pipeline.
        model_name (str): The name/path of the loaded InstructPix2Pix model.
        device (str): Computation device ('cuda' or 'cpu').
    """
    
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "diffusers/sdxl-instructpix2pix-768", 
        vae_model: str = "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype: torch.dtype = torch.float16,
        safety_checker: bool = False,
        **kwargs: Any,
    ):
        """Initializes the SDXL Pix2Pix model.

        Args:
            device (str): Hardware device to run the model on. Defaults to "cuda".
            model_name (str): HuggingFace hub path for the InstructPix2Pix SDXL model.
            vae_model (str): HuggingFace hub path or local path for the VAE model.
            torch_dtype (torch.dtype): PyTorch data type for model weights.
            safety_checker (bool): Whether to enable the NSFW safety checker.
            **kwargs (Any): Additional kwargs passed to diffusers loading methods.

        Raises:
            ImportError: If required `diffusers` library versions are missing.
            RuntimeError: If model initialization fails.
        """
        print(f"Loading SDXL Pix2Pix model: {model_name}")
        
        try:
            scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                model_name,
                subfolder="scheduler",
                **kwargs
            )

            vae = AutoencoderKL.from_pretrained(
                vae_model, 
                torch_dtype=torch_dtype,
                **kwargs
            )

            if safety_checker:
                self.pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
                    model_name,
                    vae=vae,
                    scheduler=scheduler,
                    torch_dtype=torch_dtype,
                    **kwargs
                )
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
                        model_name,
                        vae=vae,
                        scheduler=scheduler,
                        torch_dtype=torch_dtype,
                        safety_checker=None,
                        requires_safety_checker=False,
                        **kwargs
                    )
            
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()
            
            if device == "cuda" and torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
            elif device == "cpu":
                self.pipeline = self.pipeline.to("cpu")
            
            self.model_name = model_name
            self.torch_dtype = torch_dtype
            self.device = device
            
            print(f"✅ SDXL Pix2Pix model loaded successfully")
            print(f"   - Model: {model_name}")
            
        except ImportError as e:
            raise ImportError(
                "SDXL InstructPix2Pix requires diffusers>=0.19.0. "
                "Install with: pip install diffusers transformers accelerate"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load Pix2Pix model: {e}") from e
    
    def generate(
        self,
        control_image: Image.Image,
        prompt: str,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        negative_prompt: str = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        num_inference_steps: int = 30, 
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5, 
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Generates an image from a sketch using instruction-based translation with SDXL.

        Args:
            control_image (Image.Image): The input image to be translated.
            prompt (str): Text instruction (e.g., "turn it into a realistic photo").
            size (Optional[Union[int, Tuple[int, int]]]): Target resolution.
            negative_prompt (str): Text prompt detailing what to avoid.
            num_inference_steps (int): Number of denoising steps. Defaults to 30.
            guidance_scale (float): CFG scale for text guidance. Defaults to 7.5.
            image_guidance_scale (float): Scale for how closely to follow the input image structure. 
                Defaults to 1.5.
            seed (Optional[int]): Random seed for deterministic generation.
            **kwargs (Any): Additional arguments for the pipeline.

        Returns:
            Image.Image: The generated translated image.

        Raises:
            RuntimeError: If image generation fails during pipeline execution.
        """
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('num_steps', None)
        kwargs_copy.pop('guidance_scale', None)
        kwargs_copy.pop('image_guidance_scale', None)
        
        # Ensure RGB format for Pix2Pix
        if control_image.mode != 'RGB':
            control_image = control_image.convert('RGB')
        
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

        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        
        if (width, height) != (new_width, new_height):
            control_image = control_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        try:
            result = self.pipeline(
                prompt=prompt,
                image=control_image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                generator=generator,
                **kwargs_copy,
            )
            
            return result.images[0]
            
        except Exception as e:
            raise RuntimeError(f"Pix2Pix SDXL generation failed: {e}") from e