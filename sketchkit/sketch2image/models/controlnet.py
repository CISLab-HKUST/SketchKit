"""
ControlNet Model Implementation for Sketch-to-Image Conversion
"""

import torch
from diffusers import StableDiffusionControlNetPipeline
from diffusers import ControlNetModel as DiffusersControlNetModel
from PIL import Image
from typing import Tuple, Union, Dict, Any, Optional
import warnings


class ControlNetModel:
    """ControlNet model for scribble-conditioned sketch-to-image conversion."""
    
    def __init__(
        self,
        device: str = "cuda",
        controlnet_model: str = "lllyasviel/sd-controlnet-scribble",
        sd_model: str = "runwayml/stable-diffusion-v1-5",
        torch_dtype: torch.dtype = torch.float16,
        safety_checker: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize ControlNet model.
        
        Args:
            device: Device to use ('cuda' or 'cpu')
            controlnet_model: ControlNet model name
            sd_model: Stable Diffusion base model
            torch_dtype: Data type for model weights
            safety_checker: Whether to use safety checker
            **kwargs: Additional parameters for model loading
        """
        print(f"Loading ControlNet model: {controlnet_model}")
        
        try:
            # Import ControlNetModel with alias to avoid conflict
            controlnet = DiffusersControlNetModel.from_pretrained(
                controlnet_model,
                torch_dtype=torch_dtype,
                **kwargs
            )
            
            if safety_checker:
                self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    sd_model,
                    controlnet=controlnet,
                    torch_dtype=torch_dtype,
                    **kwargs
                )
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                        sd_model,
                        controlnet=controlnet,
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
            
            self.controlnet_model = controlnet_model
            self.sd_model = sd_model
            self.torch_dtype = torch_dtype
            self.device = device
            
            print(f"✅ ControlNet model loaded successfully")
            print(f"   - ControlNet: {controlnet_model}")
            print(f"   - Base model: {sd_model}")
            
        except ImportError as e:
            raise ImportError(
                "ControlNet requires diffusers>=0.14.0. "
                "Install with: pip install diffusers transformers accelerate"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load ControlNet model: {e}") from e
    
    def generate(
        self,
        control_image: Image.Image,
        prompt: str,
        size: Union[int, Tuple[int, int]] = 512,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Generate image from scribble-conditioned sketch."""
        # Remove parameters that are already explicitly handled
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('num_steps', None)
        kwargs_copy.pop('guidance_scale', None)
        kwargs_copy.pop('controlnet_scale', None)
        
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
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
                height=size if isinstance(size, int) else size[1],
                width=size if isinstance(size, int) else size[0],
                **kwargs_copy,
            )
            
            return result.images[0]
            
        except Exception as e:
            raise RuntimeError(f"ControlNet generation failed: {e}") from e