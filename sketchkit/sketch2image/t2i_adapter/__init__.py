"""
T2I-Adapter Model Implementation for Sketch-to-Image Conversion (SDXL Version)

This module encapsulates the diffusers `StableDiffusionXLAdapterPipeline` 
for generating images conditioned on edge maps or sketches.
"""

import torch
from diffusers import (
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler
)
from PIL import Image
import numpy as np
from typing import Tuple, Union, Dict, Any, Optional
import warnings


class T2IAdapterModel:
    """T2I-Adapter model for edge/sketch-conditioned sketch-to-image conversion using SDXL.
    
    Attributes:
        pipeline (StableDiffusionXLAdapterPipeline): The loaded diffusers pipeline.
        adapter_model (str): The name/path of the loaded T2I-Adapter.
        sd_model (str): The name/path of the base SDXL model.
        device (str): Computation device ('cuda' or 'cpu').
    """
    
    def __init__(
        self,
        device: str = "cuda",
        adapter_model: str = "TencentARC/t2i-adapter-sketch-sdxl-1.0", 
        sd_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        vae_model: str = "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype: torch.dtype = torch.float16,
        safety_checker: bool = False,
        **kwargs: Any,
    ):
        """Initializes the T2I-Adapter model with SDXL, VAE, and Euler Ancestral Scheduler.

        Args:
            device (str): Hardware device to run the model on. Defaults to "cuda".
            adapter_model (str): HuggingFace hub path for the T2I-Adapter.
            sd_model (str): HuggingFace hub path or local path for the base SDXL model.
            vae_model (str): HuggingFace hub path or local path for the VAE model.
            torch_dtype (torch.dtype): PyTorch data type for model weights.
            safety_checker (bool): Whether to enable the NSFW safety checker.
            **kwargs (Any): Additional kwargs passed to diffusers loading methods.

        Raises:
            ImportError: If required `diffusers` library versions are missing.
            RuntimeError: If model initialization fails.
        """
        print(f"Loading SDXL T2I-Adapter model: {adapter_model}")
        
        try:
            scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                sd_model,
                subfolder="scheduler",
                **kwargs
            )

            adapter = T2IAdapter.from_pretrained(
                adapter_model,
                torch_dtype=torch_dtype,
                **kwargs
            )

            vae = AutoencoderKL.from_pretrained(
                vae_model, 
                torch_dtype=torch_dtype,
                **kwargs
            )

            if safety_checker:
                self.pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
                    sd_model,
                    adapter=adapter,
                    vae=vae,
                    scheduler=scheduler,
                    torch_dtype=torch_dtype,
                    **kwargs
                )
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
                        sd_model,
                        adapter=adapter,
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
            
            self.adapter_model = adapter_model
            self.sd_model = sd_model
            self.torch_dtype = torch_dtype
            self.device = device
            
            print(f"✅ SDXL T2I-Adapter loaded successfully")
            print(f"   - Adapter: {adapter_model}")
            print(f"   - Base model: {sd_model}")
            
        except ImportError as e:
            raise ImportError(
                "SDXL T2I-Adapter requires diffusers>=0.19.0. "
                "Install with: pip install diffusers transformers accelerate"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load T2I-Adapter model: {e}") from e
    
    def generate(
        self,
        control_image: Image.Image,
        prompt: str,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        negative_prompt: str = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        num_inference_steps: int = 30, 
        guidance_scale: float = 7.5,
        adapter_conditioning_scale: float = 1.0,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Generates an image from an edge or sketch conditioned image using SDXL T2I-Adapter.

        Args:
            control_image (Image.Image): The input conditioning sketch/edge map.
            prompt (str): Text prompt describing the desired output.
            size (Optional[Union[int, Tuple[int, int]]]): Target resolution.
            negative_prompt (str): Text prompt detailing what to avoid.
            num_inference_steps (int): Number of denoising steps. Defaults to 30.
            guidance_scale (float): CFG scale for text guidance. Defaults to 7.5.
            adapter_conditioning_scale (float): Strength of the T2I-Adapter influence. Defaults to 1.0.
            seed (Optional[int]): Random seed for deterministic generation.
            **kwargs (Any): Additional arguments for the pipeline.

        Returns:
            Image.Image: The generated image.

        Raises:
            RuntimeError: If image generation fails during pipeline execution.
        """
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('controlnet_conditioning_scale', None)
        kwargs_copy.pop('image_guidance_scale', None)
        kwargs_copy.pop('adapter_conditioning_factor', None)

        # SDXL T2I-Adapter specifically expects RGB inputs due to PixelUnshuffle layers
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
                adapter_conditioning_scale=adapter_conditioning_scale,
                generator=generator,
                width=new_width,
                height=new_height,
                **kwargs_copy,
            )
            
            return result.images[0]
            
        except Exception as e:
            raise RuntimeError(f"T2I-Adapter SDXL generation failed: {e}") from e