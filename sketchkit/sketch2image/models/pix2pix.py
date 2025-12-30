"""
Pix2Pix Model Implementation for Sketch-to-Image Conversion
"""

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
from typing import Tuple, Union, Dict, Any, Optional
import warnings


class Pix2PixModel:
    """InstructPix2Pix model for direct sketch-to-image translation."""
    
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "timbrooks/instruct-pix2pix",
        torch_dtype: torch.dtype = torch.float16,
        safety_checker: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize Pix2Pix model.
        
        Args:
            device: Device to use ('cuda' or 'cpu')
            model_name: InstructPix2Pix model name
            torch_dtype: Data type for model weights
            safety_checker: Whether to use safety checker
            **kwargs: Additional parameters for model loading
        """
        print(f"Loading Pix2Pix model: {model_name}")
        
        try:
            if safety_checker:
                self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    **kwargs
                )
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                        model_name,
                        torch_dtype=torch_dtype,
                        safety_checker=None,
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
            
            print(f"✅ Pix2Pix model loaded successfully")
            print(f"   - Model: {model_name}")
            
        except ImportError as e:
            raise ImportError(
                "InstructPix2Pix requires diffusers>=0.14.0. "
                "Install with: pip install diffusers transformers accelerate"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load Pix2Pix model: {e}") from e
    
    def generate(
        self,
        control_image: Image.Image,
        prompt: str,
        size: Union[int, Tuple[int, int]] = 512,
        negative_prompt: str = "",
        num_inference_steps: int = 10,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.0,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Generate image from sketch using instruction-based translation."""
        # Remove parameters that are already explicitly handled
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('num_steps', None)
        kwargs_copy.pop('guidance_scale', None)
        kwargs_copy.pop('image_guidance_scale', None)
        
        # Convert image to RGB if needed
        if control_image.mode != 'RGB':
            control_image = control_image.convert('RGB')
        
        generator = None
        if seed is not None:
            device = self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(seed)
        
        try:
            result = self.pipeline(
                prompt=prompt,
                image=control_image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                generator=generator,
                height=size if isinstance(size, int) else size[1],
                width=size if isinstance(size, int) else size[0],
                **kwargs_copy,
            )
            
            return result.images[0]
            
        except Exception as e:
            raise RuntimeError(f"Pix2Pix generation failed: {e}") from e